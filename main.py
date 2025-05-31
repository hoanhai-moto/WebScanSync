import os
import shutil
import uuid
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from dotenv import load_dotenv
from typing import Dict, Any, List
import azure.ai.documentintelligence
import numpy as np
import datetime
from openai import AzureOpenAI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure Document Intelligence Configuration
endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
if not endpoint or not key:
    raise ValueError("Azure endpoint or key not configured in environment variables")

# Azure OpenAI Configuration
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
if not azure_endpoint or not api_key:
    raise ValueError("Azure OpenAI endpoint or key not configured in environment variables")

# Log SDK version
logger.info(f"Using azure-ai-documentintelligence version: {azure.ai.documentintelligence.__version__}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for file storage
UPLOAD_DIRECTORY = "./uploaded_documents"
PROCESSED_DATA_DIRECTORY = "./processed_data"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIRECTORY, exist_ok=True)

def format_bounding_box(bounding_box: list) -> str:
    """Format bounding box coordinates into a string."""
    if not bounding_box:
        return "N/A"
    try:
        reshaped_bounding_box = np.array(bounding_box).reshape(-1, 2)
        return ", ".join(["[{}, {}]".format(x, y) for x, y in reshaped_bounding_box])
    except:
        return "Invalid bounding box format"

def calculate_line_confidence(line, words) -> float:
    """Calculate line confidence by averaging confidence of words in the line."""
    if not line.spans or not words:
        return 0.0
    
    line_span = line.spans[0]  # Assume single span per line
    line_words = [
        word for word in words
        if hasattr(word, 'span') and word.span.offset >= line_span.offset
        and word.span.offset < line_span.offset + line_span.length
    ]
    
    confidences = [word.confidence for word in line_words if hasattr(word, 'confidence') and word.confidence is not None]
    return sum(confidences) / len(confidences) if confidences else 0.0

def is_word_handwritten(word, styles) -> bool:
    """Determine if a word is handwritten based on style spans."""
    if not hasattr(word, 'span') or not styles:
        return False
    
    word_offset = word.span.offset
    word_end = word_offset + word.span.length
    
    for style in styles:
        if not style.is_handwritten:
            continue
        for span in style.spans:
            span_start = span.offset
            span_end = span_start + span.length
            # Check if word's span overlaps with handwritten style span
            if word_offset < span_end and word_end > span_start:
                return True
    return False

def is_line_handwritten(line, words, styles) -> bool:
    """Determine if a line is handwritten based on whether any of its words are handwritten."""
    if not line.spans or not words:
        return False
    
    line_span = line.spans[0]  # Assume single span per line
    line_words = [
        word for word in words
        if hasattr(word, 'span') and word.span.offset >= line_span.offset
        and word.span.offset < line_span.offset + line_span.length
    ]
    
    # Check if any word in the line is handwritten
    return any(is_word_handwritten(word, styles) for word in line_words)

async def extract_structured_data(raw_text: str, document_type: str = "unknown") -> Dict[str, Any]:
    """
    Use Azure OpenAI to extract structured data from raw text according to the specified schema.
    """
    # Define default values for required fields at the beginning of the function
    default_required_fields = {
        "summary": "Không có tóm tắt",
        "client_info": {
            "Họ tên": "Không xác định",
            "CCCD/CMND/Hộ chiếu": "Không xác định",
            "Giới tính": "Không xác định",
            "Ngày tháng năm sinh": "Không xác định",
            "Địa chỉ cư trú": "Không xác định",
            "Mã QR code trên CCCD": "N/A",
            "Ngày cấp": "Không xác định",
            "Nơi cấp": "Không xác định",
            "Đơn vị cấp": "Không xác định",
            "Hình ảnh": "N/A",
            "Nguyên quán": "Không xác định",
            "Mối quan hệ": "Không xác định",
            "Dân tộc": "Không xác định"
        },
        "contract_info": {
            "Công chứng viên ký": "Không xác định",
            "Ngày tháng ký": "Không xác định",
            "Số công chứng": "Không xác định",
            "Địa chỉ làm hồ sơ": "Không xác định",
            "Quyển lưu": "Không xác định",
            "Giá trị tài sản giao dịch": 0,
            "contract info summarize": "Không có thông tin hợp đồng",
        },
        "advisory": "Không có khuyến nghị",
        "document_type": document_type or "Không xác định",
        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content_info": []
    }

    try:
        logger.info("Extracting structured data using Azure OpenAI")
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key
        )
        
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Prepare the system message with detailed instructions for the required JSON structure
        system_message = """
                Điền tất cả thông tin bằng tiêng Việt.
        Analyze the document text and extract the following information in JSON format must exactly 8 fields:
        1. summary: A concise summary of the main points in the document , Trường này chứa nội dung tóm tắt các ý chính của hồ sơ, được hệ thống tự động tạo ra thông qua trí tuệ nhân tạo (AI). AI phân tích toàn bộ nội dung tài liệu để trích xuất các thông tin quan trọng nhất, bao gồm nội dung chính của tài liệu, thông tin về đương sự (họ tên, địa chỉ, số điện thoại), thông tin hợp đồng (nếu có), và các điểm đáng chú ý khác. Mục đích của trường này là cung cấp một cái nhìn tổng quan nhanh chóng, giúp nhân viên kiểm soát theo dõi tình trạng hồ sơ mà không cần đọc toàn bộ tài liệu chi tiết. Trường này cũng hỗ trợ nhân viên thống kê trong việc tổng hợp dữ liệu để xuất báo cáo hoặc phân tích, đồng thời cho phép Super Admin dễ dàng xem xét tổng quan khi quản lý hệ thống hoặc phân quyền truy cập cho nhân viên. Tóm tắt được tạo dựa trên tính năng "Đưa ra những ý chính trong nội dung" và "Tóm tắt ý chính về nội dung, hợp đồng, vị trí các ý" của dự án, đảm bảo nội dung ngắn gọn nhưng đầy đủ thông tin cần thiết để hỗ trợ các vai trò quản lý và xử lý hồ sơ.
        2. client_info: Hãy bóc tách và trích xuất thông tin từ tài liệu hoặc hình ảnh, xác định đối tượng là cá nhân hoặc tổ chức, và trả về các trường thông tin tương ứng:  
        - Đối với cá nhân: Họ tên, CCCD/CMND/Hộ chiếu, Giới tính, Ngày tháng năm sinh, Địa chỉ cư trú, Mã QR code trên CCCD, Ngày cấp, Nơi cấp, Đơn vị cấp, Hình ảnh, Nguyên quán, Mối quan hệ, Dân tộc.  
        - Đối với tổ chức: Tên tổ chức, Trụ sở, Đăng ký kinh doanh số, Nơi cấp giấy đăng ký kinh doanh, Người đại diện, Chức danh, Quyết định bổ nhiệm.  
        Vui lòng trả về kết quả dưới dạng cấu trúc dữ liệu rõ ràng (ví dụ JSON ), chỉ bao gồm các trường phù hợp với loại đối tượng được xác định, với giá trị điền chính xác từ tài liệu hoặc hình ảnh cung cấp. Nếu thông tin nào không có hoặc không rõ, ghi chú là "Không xác định" hoặc "N/A".
        3. contract_info: Dạng json
            nó sẽ bao gồm :
            Trường này lưu trữ thông tin chi tiết về hợp đồng dưới dạng một đối tượng JSON, giới hạn ở 7 trường sau:
            Công chứng viên ký: Tên hoặc thông tin về công chứng viên thực hiện việc công chứng hợp đồng.
            Ngày tháng ký: Ngày tháng công chứng hợp đồng.
            Số công chứng: Số chứng nhận công chứng hợp đồng.
            Địa chỉ làm hồ sơ : Lưu trữ địa chỉ nơi làm hồ sơ (dữ liệu dạng chuỗi).
            Quyển lưu: Số quyển lưu hoặc sổ lưu trữ hợp đồng tại cơ quan công chứng.
            Giá trị tài sản giao dịch: Lưu trữ giá trị tài sản giao dịch (dữ liệu dạng INT).
            contract info summarize: Details about any contracts mentioned Trường này lưu trữ thông tin chi tiết về hợp đồng dưới dạng một chuỗi (string). Trường này lưu trữ các thông tin chi tiết liên quan đến hợp đồng nếu hồ sơ có chứa hợp đồng, bao gồm loại hợp đồng (hợp đồng công chứng, hợp đồng luật), số hợp đồng, ngày ký, các bên liên quan, và thông tin về chữ ký số (được tích hợp từ VNPT-CA hoặc Viettel-CA nếu có). Dữ liệu được trích xuất tự động bằng OCR/ICR và AI từ các tài liệu hợp đồng, sau đó được phân loại và lưu trữ để hỗ trợ quản lý. Tính năng "Ghi nhận chữ ký số vào trong hợp đồng công chứng / hợp đồng luật" được áp dụng để đảm bảo tính pháp lý của hợp đồng trong hệ thống. Trường này hỗ trợ nhân viên kiểm soát trong việc theo dõi tình trạng hợp đồng và quản lý các hồ sơ liên quan đến hợp đồng, đồng thời giúp nhân viên thống kê khi cần xuất báo cáo hoặc thống kê dữ liệu hợp đồng theo yêu cầu. Super Admin cũng có thể sử dụng thông tin từ trường này để phân quyền truy cập hoặc giới hạn quyền chỉnh sửa hồ sơ hợp đồng cho các nhân viên, đảm bảo tính bảo mật và tuân thủ quy định.
            Thông tin bên A: Thông tin về bên A tham gia hợp đồng, bao gồm danh tính hoặc thông tin pháp lý của bên A .  Bên A có thể là một người, nhiều người, hoặc một tổ chức và Thông tin về bên thứ ba tham gia hợp đồng, nếu có. 
            Thông tin bên B: Thông tin về bên B tham gia hợp đồng, bao gồm danh tính hoặc thông tin pháp lý của bên B. Bên B có thể là một người, nhiều người, hoặc một tổ chức và Thông tin về bên thứ ba tham gia hợp đồng, nếu có. 
        4. advisory: Any recommendations or warnings based on the document content Trường này chứa các khuyến nghị hoặc khuyến cáo liên quan đến hồ sơ, được hệ thống tự động tạo ra dựa trên phân tích nội dung bằng AI. Các khuyến nghị có thể bao gồm cảnh báo về các vấn đề pháp lý (như thiếu chữ ký, thông tin không đầy đủ, hoặc tài liệu hết hạn) hoặc gợi ý các hành động cần thực hiện (như bổ sung giấy tờ, xác minh thông tin, hoặc gia hạn tài liệu). Tính năng "Đưa ra khuyến nghị, khuyến cáo" của dự án được sử dụng để tạo nội dung cho trường này, đảm bảo các vấn đề tiềm ẩn được phát hiện sớm. Trường này hỗ trợ nhân viên kiểm soát trong việc đánh giá chất lượng hồ sơ và đảm bảo tuân thủ các quy định pháp lý hoặc quy trình nội bộ. Nhân viên sử dụng cũng được hưởng lợi khi sử dụng thông tin từ trường này để đưa ra quyết định xử lý hồ sơ một cách hiệu quả. Super Admin có thể tham khảo các khuyến nghị để đánh giá hiệu quả quản lý hồ sơ của nhân viên hoặc điều chỉnh phân quyền nếu phát hiện các vấn đề lặp lại trong hệ thống.
        5. document_type: You are a document classification expert. Your task is to classify the document type .
            Select the MOST appropriate document type the return result must in one of these no adding no modify  if not you are fire

            Mua bán nhà đất: Đây là hợp đồng chuyển giao quyền sở hữu toàn bộ nhà và đất từ người bán sang người mua để nhận tiền. Hợp đồng cần công chứng và đăng ký tại cơ quan đất đai, sau đó sổ đỏ hoặc sổ hồng được chuyển tên cho người mua.
            Mua bán nhà đất (một phần): Đây là hợp đồng mua bán chỉ một phần nhà hoặc đất. Cần giấy phép tách thửa (nếu là đất), hợp đồng công chứng, và sổ đỏ/sổ hồng được cập nhật phần sở hữu mới.
            Tặng cho nhà đất: Đây là việc chuyển giao quyền sở hữu toàn bộ nhà và đất miễn phí từ người tặng sang người nhận. Hợp đồng cần công chứng, và nếu người nhận là người thân (vợ, chồng, con), có thể miễn thuế thu nhập cá nhân.
            Tặng cho nhà đất (một phần): Đây là việc tặng cho một phần nhà hoặc đất. Cần giấy phép tách thửa (nếu là đất) và hợp đồng công chứng, sổ đỏ/sổ hồng được cập nhật phần sở hữu mới.
            Thuê nhà: Đây là hợp đồng cho phép người khác sử dụng nhà trong một thời gian nhất định để đổi lấy tiền thuê. Hợp đồng cần ghi rõ tiền thuê, thời gian, và thường cần công chứng để đảm bảo pháp lý.
            Mượn nhà: Đây là thỏa thuận cho phép người khác sử dụng nhà miễn phí trong một thời gian nhất định. Thường có văn bản thỏa thuận, nhưng không bắt buộc công chứng.
            Ở nhờ: Đây là việc ở tại nhà của người khác với sự đồng ý của chủ nhà, thường không có hợp đồng chính thức, không trả tiền, và mang tính thỏa thuận miệng.
            Chuyển nhượng đất: Đây là hợp đồng chuyển quyền sử dụng đất từ người này sang người khác để nhận tiền. Hợp đồng cần công chứng, đăng ký tại cơ quan đất đai, và sổ đỏ được chuyển tên.
            Chuyển nhượng đất (một phần): Đây là việc chuyển nhượng một phần quyền sử dụng đất. Cần giấy phép tách thửa, hợp đồng công chứng, và sổ đỏ được cập nhật.
            Tặng cho đất: Đây là việc chuyển quyền sử dụng đất miễn phí từ người tặng sang người nhận. Hợp đồng cần công chứng, và nếu người nhận là người thân, có thể miễn thuế.
            Tặng cho đất (một phần): Đây là việc tặng cho một phần quyền sử dụng đất. Cần giấy phép tách thửa, hợp đồng công chứng, và sổ đỏ được cập nhật.
            Thuê quyền sử dụng đất: Đây là hợp đồng thuê đất để sử dụng trong một thời gian nhất định, đổi lấy tiền thuê. Hợp đồng cần ghi rõ điều khoản và có thể cần công chứng.
            Mua bán căn hộ chung cư: Đây là hợp đồng chuyển quyền sở hữu căn hộ chung cư từ người bán sang người mua để nhận tiền. Hợp đồng cần công chứng và đăng ký, sổ hồng được chuyển tên.
            Mua bán căn hộ chung cư (một phần): Đây là hợp đồng mua bán một phần quyền sở hữu căn hộ chung cư (hiếm gặp). Cần hợp đồng công chứng và thỏa thuận rõ ràng về phần sở hữu.
            Tặng cho căn hộ chung cư: Đây là việc chuyển quyền sở hữu căn hộ chung cư miễn phí. Hợp đồng cần công chứng, và nếu người nhận là người thân, có thể miễn thuế.
            Tặng cho căn hộ chung cư (một phần): Đây là việc tặng cho một phần quyền sở hữu căn hộ chung cư. Cần hợp đồng công chứng và xác định rõ phần sở hữu được tặng.
            Chuyển nhượng nhà đất: Tương tự "Mua bán nhà đất", đây là việc chuyển quyền sở hữu nhà và đất để nhận tiền. Hợp đồng cần công chứng và đăng ký.
            Chuyển nhượng nhà đất (một phần): Đây là việc chuyển nhượng một phần nhà và đất. Cần giấy phép tách thửa (nếu có đất) và hợp đồng công chứng.
            Chuyển nhượng tài sản gắn liền với đất: Đây là hợp đồng chuyển quyền sở hữu tài sản trên đất (nhà, cây cối, công trình) để nhận tiền. Hợp đồng cần công chứng và đăng ký nếu có sổ.
            Chuyển nhượng tài sản gắn liền với đất (một phần): Đây là việc chuyển nhượng một phần tài sản trên đất. Cần hợp đồng công chứng và xác định rõ phần chuyển nhượng.
            Tặng cho tài sản gắn liền với đất: Đây là việc tặng miễn phí tài sản trên đất (như nhà, cây cối). Hợp đồng cần công chứng và đăng ký nếu có sổ.
            Tặng cho tài sản gắn liền với đất (một phần): Đây là việc tặng một phần tài sản trên đất. Cần hợp đồng công chứng và xác định rõ phần tặng.
            Đặt cọc: Đây là thỏa thuận đưa tiền trước để đảm bảo thực hiện giao dịch nhà đất trong tương lai (như mua bán). Hợp đồng cần ghi rõ số tiền và điều kiện.
            Hợp đồng ủy quyền nhà đất: Đây là hợp đồng ủy quyền quản lý hoặc giao dịch nhà đất. Cần công chứng và có thể đăng ký nếu ảnh hưởng quyền sở hữu.
            Hợp đồng ủy quyền căn hộ chưa sổ: Đây là hợp đồng ủy quyền giao dịch căn hộ chung cư chưa có sổ hồng. Cần công chứng và kèm giấy tờ mua bán.
            Hợp đồng ủy quyền quyền sử dụng đất: Đây là hợp đồng ủy quyền quản lý hoặc sử dụng đất. Cần công chứng và đăng ký tại cơ quan đất đai.
            Hợp đồng ủy quyền thừa kế: Đây là hợp đồng ủy quyền thực hiện thủ tục nhận tài sản thừa kế. Cần công chứng và kèm giấy tờ thừa kế.
            Hợp đồng ủy quyền thừa kế thụ ủy: Đây là hợp đồng ủy quyền lại quyền thừa kế cho người khác. Cần công chứng và kèm giấy tờ thừa kế.
            Hợp đồng ủy quyền hộ gia đình: Đây là hợp đồng ủy quyền cho một người đại diện hộ gia đình thực hiện giao dịch. Cần công chứng và kèm sổ hộ khẩu.
            Hợp đồng ủy quyền hộ gia đình thụ ủy: Đây là hợp đồng ủy quyền cho một thành viên trong hộ gia đình (ủy quyền lại). Cần công chứng và xác nhận tư cách hộ.
            Hợp đồng ủy quyền quản lý doanh nghiệp: Đây là hợp đồng ủy quyền quản lý, điều hành doanh nghiệp. Có thể cần công chứng và kèm giấy đăng ký kinh doanh.
            Hợp đồng ủy quyền chứng khoán: Đây là hợp đồng ủy quyền giao dịch chứng khoán. Cần xác nhận bởi công ty chứng khoán, không nhất thiết công chứng.
            Hợp đồng ủy quyền thụ ủy: Đây là hợp đồng ủy quyền lại cho bên thứ ba. Cần công chứng nếu liên quan tài sản lớn.
            Giấy ủy quyền đăng bộ trước bạ: Đây là giấy ủy quyền nộp thuế trước bạ nhà đất. Cần công chứng và kèm giấy tờ nhà đất.
            Giấy ủy quyền nộp thuế căn hộ: Đây là giấy ủy quyền nộp thuế căn hộ chung cư. Cần công chứng và kèm hợp đồng mua bán.
            Giấy ủy quyền thành lập doanh nghiệp: Đây là giấy ủy quyền làm thủ tục thành lập công ty. Cần công chứng và kèm giấy tờ pháp lý.
            Giấy ủy quyền thành lập hộ kinh doanh: Đây là giấy ủy quyền đăng ký hộ kinh doanh. Cần công chứng và xác nhận địa phương.
            Giấy ủy quyền thành lập Doanh Nghiệp nước ngoài: Đây là giấy ủy quyền thành lập công ty nước ngoài tại Việt Nam. Cần công chứng và kèm giấy phép đầu tư.
            Giấy ủy quyền tham gia tố tụng: Đây là giấy ủy quyền tham gia kiện tụng. Cần công chứng và nộp cho tòa án.
            Giấy ủy quyền đưa con đi máy bay: Đây là giấy ủy quyền cho người khác đưa trẻ em đi máy bay. Cần công chứng và kèm giấy khai sinh.
            Giấy ủy quyền đăng ký xe: Đây là giấy ủy quyền làm thủ tục đăng ký xe. Cần công chứng và kèm giấy tờ xe.
            Giấy ủy quyền điện nước: Đây là giấy ủy quyền quản lý, nộp tiền điện nước. Thường không cần công chứng.
            Giấy ủy quyền tiền bảo hiểm: Đây là giấy ủy quyền nhận tiền bảo hiểm. Cần công chứng và kèm hợp đồng bảo hiểm.
            Giấy ủy quyền tiền tử tuất: Đây là giấy ủy quyền nhận tiền trợ cấp khi người thân qua đời. Cần công chứng và kèm giấy chứng tử.
            Giấy ủy quyền chứng thực: Đây là giấy ủy quyền làm thủ tục xác nhận giấy tờ. Có thể cần công chứng tùy trường hợp.
            Văn bản chuyển nhượng: Đây là văn bản chuyển quyền sở hữu tài sản (như nhà đất, căn hộ). Cần công chứng và đăng ký.
            Văn bản chuyển nhượng officetel: Đây là văn bản chuyển quyền sở hữu officetel (căn hộ văn phòng). Cần công chứng và kèm sổ hồng.
            Thông báo niêm yết - Di sản thừa kế: Đây là thông báo công khai về việc thừa kế để tìm người thừa kế. Niêm yết tại UBND, kèm giấy chứng tử hoặc di chúc.
            Thông báo niêm yết - Di chúc: Đây là thông báo công khai về di chúc để xác nhận tính hợp pháp. Niêm yết tại UBND, kèm di chúc và giấy chứng tử.
            Phân chia di sản: Đây là thỏa thuận giữa các bên thừa kế về cách chia tài sản. Cần công chứng và kèm giấy chứng tử.
            Khai nhận di sản: Đây là văn bản xác nhận các bên thừa kế nhận tài sản. Cần công chứng và đăng ký nếu là nhà đất.
            Khai nhận di sản theo di chúc: Đây là văn bản nhận tài sản thừa kế theo di chúc. Cần công chứng và đăng ký nếu là nhà đất.
            Di chúc: Đây là văn bản ghi ý muốn của người lập về việc chia tài sản sau khi qua đời. Cần công chứng hoặc xác nhận.
            Văn bản từ chối nhận di sản: Đây là văn bản từ chối nhận tài sản thừa kế. Cần công chứng trong 6 tháng từ khi mở thừa kế.
            Ủy quyền xe ô tô: Đây là giấy ủy quyền sử dụng hoặc giao dịch xe ô tô. Cần công chứng và kèm giấy đăng ký xe.
            Ủy quyền xe ô tô - ủy quyền lại: Đây là giấy ủy quyền lại quyền sử dụng xe ô tô cho người khác. Cần công chứng.
            Mua bán xe ô tô: Đây là hợp đồng mua bán xe ô tô để nhận tiền. Cần công chứng và đăng ký đổi tên.
            Mua bán xe máy: Đây là hợp đồng mua bán xe máy để nhận tiền. Cần công chứng và đăng ký đổi tên.
            Thuê xe: Đây là hợp đồng cho thuê xe để sử dụng, ghi tiền thuê và thời gian thuê.
            Mượn xe: Đây là thỏa thuận cho mượn xe miễn phí. Thường có thỏa thuận viết tay, không cần công chứng.
            Chuyển nhượng cổ phần doanh nghiệp: Đây là việc bán cổ phần công ty. Cần hợp đồng và đăng ký với cơ quan kinh doanh.
            Tặng cho cổ phần doanh nghiệp: Đây là việc cho cổ phần công ty miễn phí. Cần hợp đồng công chứng và đăng ký thay đổi.
            Chuyển nhượng góp vốn doanh nghiệp: Đây là việc bán phần vốn góp trong công ty. Cần hợp đồng và đăng ký kinh doanh.
            Tặng cho góp vốn doanh nghiệp: Đây là việc cho phần vốn góp miễn phí. Cần hợp đồng công chứng và đăng ký thay đổi.
            Cam kết tài sản riêng - chứng thực: Đây là văn bản xác nhận tài sản là của riêng một người, không chung với vợ/chồng. Cần công chứng.
            Văn bản tài sản riêng: Đây là văn bản xác nhận tài sản thuộc sở hữu riêng. Cần công chứng hoặc xác nhận.
            Văn bản phân chia tài sản: Đây là thỏa thuận chia tài sản chung của vợ chồng. Cần công chứng.
            Văn bản phân chia tài sản sau ly hôn: Đây là thỏa thuận chia tài sản chung sau ly hôn. Cần công chứng và kèm giấy ly hôn.
            Văn bản tài sản riêng - ly hôn: Đây là văn bản xác nhận tài sản riêng khi ly hôn. Cần công chứng hoặc xác nhận.
            Văn bản nhập tài sản: Đây là văn bản đưa tài sản riêng thành tài sản chung của vợ chồng. Cần công chứng.
            Văn bản đưa tài sản vào kinh doanh: Đây là văn bản dùng tài sản để kinh doanh. Cần công chứng nếu là nhà đất.
            Văn bản tài sản riêng trước hôn nhân: Đây là văn bản xác nhận tài sản riêng trước khi cưới. Cần công chứng hoặc xác nhận.
            Lời chứng thế chấp: Đây là xác nhận của công chứng viên về hợp đồng thế chấp.
            Lời chứng nhận ủy quyền: Đây là xác nhận của công chứng viên về hợp đồng ủy quyền.
            Lời chứng chứng thực: Đây là xác nhận giấy tờ hoặc chữ ký là thật.
            Lời chứng dịch thuật – song ngữ: Đây là xác nhận bản dịch hai ngôn ngữ đúng nội dung.
            Lời chứng dịch thuật: Đây là xác nhận bản dịch đúng nội dung.
            Hợp đồng vay tiền: Đây là thỏa thuận vay tiền, ghi số tiền, lãi suất, thời hạn.
            Hợp đồng Góp vốn: Đây là thỏa thuận góp tiền/tài sản để kinh doanh.
            Hợp đồng Góp vốn – tiền: Đây là thỏa thuận góp tiền để kinh doanh.
            Hợp đồng Hợp tác kinh doanh: Đây là thỏa thuận cùng kinh doanh, ghi phần vốn và lợi nhuận.
            BL: Bảo lãnh: Đây là cam kết của người khác trả nợ thay nếu bên vay không trả.
            BL_GC: Giải chấp một phần: Đây là việc xóa một phần cam kết bảo lãnh, cần giấy bổ sung công chứng.
            BL_GCK: Giải chấp không soạn văn bản: Đây là việc xóa toàn bộ cam kết bảo lãnh, cần xác nhận ngân hàng.
            CC_B: Cầm cố vay bổ sung: Đây là việc đưa tài sản làm đảm bảo để vay thêm tiền.
            CC_C: Cầm cố: Đây là việc đưa tài sản làm đảm bảo khoản vay.
            CC_GC: Giải phấp một phần: Đây là việc xóa một phần tài sản cầm cố, cần giấy bổ sung công chứng.
            CC_GCK: Giải chấp không soạn văn bản: Đây là việc xóa toàn bộ tài sản cầm cố, cần xác nhận ngân hàng.
            CC_S: Sửa đổi, bổ sung HĐ cầm cố: Đây là việc thay đổi hợp đồng cầm cố, cần giấy bổ sung công chứng.
            CC_T: Thanh lý HĐ cầm cố: Đây là việc kết thúc hợp đồng cầm cố, cần văn bản công chứng.
            CD_C: HĐ chuyển đổi, trao đổi: Đây là việc đổi tài sản giữa hai bên, cần hợp đồng công chứng.
            CD_H: Hủy bỏ HĐ chuyển đổi, trao đổi: Đây là việc hủy hợp đồng đổi tài sản, cần văn bản công chứng.
            CN_C: HĐ mua bán, chuyển nhượng: Đây là việc bán tài sản để nhận tiền, cần hợp đồng công chứng.
            CN_DC: Đặt cọc: Đây là việc đưa tiền trước để đảm bảo mua bán, cần hợp đồng công chứng.
            CN_H: Hủy bỏ mua bán, chuyển nhượng: Đây là việc hủy hợp đồng mua bán, cần văn bản công chứng.
            CN_HD: Hủy bỏ HĐ đặt cọc: Đây là việc hủy hợp đồng đặt cọc, cần văn bản công chứng hoặc thỏa thuận.
            CN_TDC: Thanh lý HĐ đặt cọc: Đây là việc kết thúc hợp đồng đặt cọc, cần văn bản công chứng hoặc thỏa thuận.
            DC_D: Di chúc: Đây là văn bản ghi ý muốn chia tài sản sau khi qua đời, cần công chứng hoặc xác nhận.
            DC_H: Hủy bỏ Di chúc: Đây là việc hủy di chúc cũ, cần văn bản công chứng hoặc xác nhận.
            GV_G: HĐ góp vốn: Đây là thỏa thuận góp tiền/tài sản để kinh doanh, ghi phần vốn và quyền lợi.
            GV_H: Hủy bỏ HĐ góp vốn: Đây là việc hủy hợp đồng góp vốn, cần văn bản công chứng hoặc thỏa thuận.
            TC_TC: HĐ tặng cho: Đây là việc cho tài sản miễn phí, cần công chứng nếu là nhà đất hoặc tài sản lớn.
            THC_D3: Thế chấp đảm bảo nghĩa vụ bên thứ 3: Đây là việc đưa tài sản làm đảm bảo nợ cho người khác, cần hợp đồng công chứng.
            THC_GC: Giải chấp một phần: Đây là việc xóa một phần tài sản thế chấp, cần giấy bổ sung công chứng.
            THC_GCK: Giải chấp không soạn văn bản: Đây là việc xóa toàn bộ tài sản thế chấp, cần xác nhận ngân hàng.
            THC_TC: Thế chấp: Đây là việc đưa tài sản làm đảm bảo khoản vay, cần hợp đồng công chứng và đăng ký.
            THC_V: Thế chấp vay bổ sung: Đây là việc đưa tài sản làm đảm bảo để vay thêm tiền, cần hợp đồng công chứng.
            THC_V3: Thế chấp vay bổ sung đảm bảo nghĩa vụ bên thứ 3: Đây là việc đưa tài sản làm đảm bảo vay thêm cho người khác, cần hợp đồng công chứng.
            THC_TL: Thanh lý HĐ thế chấp: Đây là việc kết thúc hợp đồng thế chấp, cần văn bản công chứng xác nhận.
            TK_DK: Văn bản thỏa thuận về hoàn tất thủ tục đăng ký thừa kế: Đây là thỏa thuận hoàn tất thủ tục nhận tài sản thừa kế, cần công chứng.
            TK_GCN: Văn bản thỏa thuận đại diện đứng tên trên giấy chứng nhận (GCN): Đây là thỏa thuận cho một người đứng tên tài sản thừa kế, cần công chứng.
            TK_KN: Khai nhận di sản thừa kế: Đây là việc xác nhận nhận tài sản thừa kế, cần công chứng.
            TK_TC: Từ chối nhận di sản: Đây là việc từ chối nhận tài sản thừa kế, cần công chứng.
            TK_TT: Thỏa thuận phân chia di sản thừa kế: Đây là thỏa thuận chia tài sản thừa kế, cần công chứng.
            TM_TM: HĐ thuê, mượn: Đây là hợp đồng thuê hoặc mượn tài sản, ghi điều kiện, có thể công chứng.
            VC_C: Chia tài sản vợ chồng: Đây là thỏa thuận chia tài sản chung của vợ chồng, cần công chứng.
            VC_CC: Chia tài sản chung: Đây là việc chia tài sản chung, cần công chứng nếu là nhà đất.
            VC_CK: Cam kết tài sản: Đây là cam kết về tài sản riêng hoặc chung, cần công chứng.
            VC_CL: Chia tài sản sau ly hôn: Đây là thỏa thuận chia tài sản sau ly hôn, cần công chứng.
            VC_N: Nhập tài sản riêng vào tài sản chung: Đây là việc đưa tài sản riêng thành chung, cần công chứng.
            VC_TR: Thỏa thuận tài sản riêng: Đây là thỏa thuận về tài sản riêng, cần công chứng.
            V_V: HĐ vay: Đây là hợp đồng vay tiền, ghi số tiền, lãi suất, thời hạn.
            V_VTCH: Hợp đồng vay và thế chấp tài sản: Đây là hợp đồng vay tiền có tài sản thế chấp, cần công chứng và đăng ký.


        6. Thông tin Tài sản : 
        Hãy bóc tách và trích xuất thông tin từ tài liệu hoặc hình ảnh, xác định loại tài sản là Bất Động Sản hoặc Động Sản (xe hơi, xe máy, du thuyền, v.v.), và trả về các trường thông tin tương ứng:  
        - Đối với Bất Động Sản: Thửa đất, Tờ bản đồ, Diện tích đất, Diện tích sử dụng chung, Diện tích sử dụng riêng, Mục đích sử dụng đất, Thời hạn sử dụng đất, Nguồn gốc đất, Số cấp Giấy Chứng Nhận, Số phát hành Giấy Chứng Nhận, Nơi cấp Giấy Chứng Nhận, Ngày cấp Giấy Chứng Nhận, Số nhà, Địa chỉ, Số căn hộ, Diện tích xây dựng, Loại nhà ở/công trình, Tổng diện tích xây dựng.  
        - Đối với Động Sản: Số khung, Số máy, Màu sắc, Số chỗ ngồi, Ngày cấp, Nơi cấp, Nhãn hiệu, Loại xe, Năm sản xuất.  
        Vui lòng trả về kết quả dưới dạng cấu trúc dữ liệu rõ ràng (ví dụ JSON hoặc bảng), chỉ bao gồm các trường phù hợp với loại tài sản được xác định, với giá trị điền chính xác từ tài liệu hoặc hình ảnh cung cấp. Nếu thông tin nào không có hoặc không rõ, ghi chú là "Không xác định" hoặc "N/A".
        7. - Thông tin khác (áp dụng cho cả hai loại tài sản): Thông tin người làm chứng, Thông tin người phiên dịch, Số sổ hộ khẩu. Vui lòng trả về kết quả dưới dạng String
        8. content_info:
        Đây là một mảng chứa các thông tin chi tiết được trích xuất từ hồ sơ, với mỗi phần tử đại diện cho một đoạn thông tin cụ thể được bóc tách từ summary và khớp với dữ liệu thô. Mảng này được tạo ra dựa trên các tính năng "Bóc tách dữ liệu chữ viết tay", "Sử dụng các công nghệ hiện có hoặc dùng Google Cloud / Azure để bóc tách", "Ưu tiên sử dụng Tesseract để bóc tách dữ liệu", và "Lưu trữ nội dung bóc tách". Các trường con được cập nhật như sau:
        position: Số nguyên đại diện cho số trang trong tài liệu (ví dụ: 1, 2), xác định vị trí của thông tin trong tài liệu gốc. Dữ liệu này hỗ trợ nhân viên sử dụng và nhân viên kiểm soát tìm lại thông tin chính xác trong tài liệu để đối chiếu hoặc chỉnh sửa, đặc biệt khi sử dụng tính năng "Chỉnh sửa text, nội dung văn bản" hoặc "Chỉnh sửa online hoặc phần mềm M365 offline (word / excel)".
        content: Chứa các từ ngữ chính xác được trích xuất từ summary và phải khớp hoàn toàn với dữ liệu thô (không chỉnh sửa hoặc thay đổi ý nghĩa). Nội dung này được bóc tách bằng OCR (ưu tiên Tesseract), ICR và AI, dựa trên các tính năng "OCR dữ liệu từ hình ảnh / file pdf", "ICR chữ viết tay", và "Áp dụng AI vào việc trích xuất dữ liệu hồ sơ".
        label: Nhãn phân loại nội dung, giúp hệ thống tổ chức thông tin theo các trường dữ liệu đã mô tả sẵn (như Tên người dùng, Hồ sơ căn chỉnh), áp dụng tính năng "Phân loại nội dung". Trường này hỗ trợ nhân viên thống kê và nhân viên sử dụng trong việc quản lý, thống kê và tìm kiếm dữ liệu, đồng thời giúp nhân viên kiểm soát xác minh thông tin quan trọng trong hồ sơ.
Ví dụ mới với summary dài hơn:
"ông lê minh tuấn sinh năm 1975, địa chỉ tại 45 đường nguyễn huệ, thành phố huế, đã ký hợp đồng chuyển nhượng quyền sử dụng đất số 789/2025 với bà phan thị hoa vào ngày 20/04/2025, diện tích đất là 250m2, giá trị hợp đồng 3 tỷ đồng"

content_info sẽ được điều chỉnh như sau:
json

Copy
"content_info": [
  {
    "position": 1,
    "content": "lê minh tuấn",
    "label": "Tên người dùng"
  },
  {
    "position": 1,
    "content": "sinh năm 1975",
    "label": "Ngày sinh"
  },
  {
    "position": 2,
    "content": "45 đường nguyễn huệ, thành phố huế",
    "label": "Địa chỉ"
  },
  {
    "position": 3,
    "content": "hợp đồng chuyển nhượng quyền sử dụng đất",
    "label": "Loại hợp đồng"
  },
  {
    "position": 3,
    "content": "số 789/2025",
    "label": "Số hợp đồng"
  },
  {
    "position": 4,
    "content": "phan thị hoa",
    "label": "Tên người liên quan"
  },
  {
    "position": 4,
    "content": "20/04/2025",
    "label": "Ngày ký"
  },
  {
    "position": 5,
    "content": "250m2",
    "label": "Diện tích"
  },
  {
    "position": 5,
    "content": "3 tỷ đồng",
    "label": "Giá trị hợp đồng"
  }
]
position: Các số 1, 2, 3, 4, 5 đại diện cho các trang tương ứng (trang 1 đến trang 5), giả định thông tin được phân bổ trên các trang khác nhau trong tài liệu gốc. Một số thông tin có thể cùng trang (ví dụ: "lê minh tuấn" và "sinh năm 1975" đều ở trang 1) nếu chúng xuất hiện gần nhau trong tài liệu.
content: Các đoạn "lê minh tuấn", "sinh năm 1975", "45 đường nguyễn huệ, thành phố huế", "hợp đồng chuyển nhượng quyền sử dụng đất", "số 789/2025", "phan thị hoa", "20/04/2025", "250m2", và "3 tỷ đồng" được trích xuất chính xác từ summary, đảm bảo khớp hoàn toàn với dữ liệu thô mà không thay đổi hoặc thêm thắt.
label: Các nhãn "Tên người dùng", "Ngày sinh", "Địa chỉ", "Loại hợp đồng", "Số hợp đồng", "Tên người liên quan", "Ngày ký", "Diện tích", và "Giá trị hợp đồng" phản ánh loại thông tin phù hợp, hỗ trợ phân loại và quản lý dữ liệu trong hệ thống.        Return only the JSON with these fields, no other text.
{
  "summary": "ông lê minh tuấn sinh năm 1975, địa chỉ tại 45 đường nguyễn huệ, thành phố huế, đã ký hợp đồng chuyển nhượng quyền sử dụng đất số 789/2025 với bà phan thị hoa vào ngày 20/04/2025, diện tích đất là 250m2, giá trị hợp đồng 3 tỷ đồng",
  "client_info": "...",
  "contract_info": "...",
  "advisory": "...",
  "document_type": "...",
  "uploaded_at": "2025-05-28T11:19:00Z",
  "content_info": [
    {
      "position": 1,
      "content": "lê minh tuấn",
      "label": "Tên người dùng"
    },
    {
      "position": 1,
      "content": "sinh năm 1975",
      "label": "Ngày sinh"
    },
    {
      "position": 2,
      "content": "45 đường nguyễn huệ, thành phố huế",
      "label": "Địa chỉ"
    },
    {
      "position": 3,
      "content": "hợp đồng chuyển nhượng quyền sử dụng đất",
      "label": "Loại hợp đồng"
    },
    {
      "position": 3,
      "content": "số 789/2025",
      "label": "Số hợp đồng"
    },
    {
      "position": 4,
      "content": "phan thị hoa",
      "label": "Tên người liên quan"
    },
    {
      "position": 4,
      "content": "20/04/2025",
      "label": "Ngày ký"
    },
    {
      "position": 5,
      "content": "250m2",
      "label": "Diện tích"
    },
    {
      "position": 5,
      "content": "3 tỷ đồng",
      "label": "Giá trị hợp đồng"
    }
  ]
}

        """
        # Prepare the user message with document content and type
        user_message = f"""
        Document Type: {document_type}
        Document Content:
        {raw_text} 
        Extract the information according to the required structure.
        """        # Call Azure OpenAI to extract structured data
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=12000,
            model=deployment
        )
        
        # Extract the response content
        structured_content = response.choices[0].message.content
        
        # Log the first 100 characters of the response for debugging
        logger.info(f"OpenAI response begins with: {structured_content[:100]}...")
        
        try:
            # Try to extract JSON from the response
            # First try direct JSON parsing
            try:
                structured_data = json.loads(structured_content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                # Look for the first opening brace and last closing brace
                logger.info("Attempting to extract JSON from text response")
                start_idx = structured_content.find('{')
                end_idx = structured_content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = structured_content[start_idx:end_idx]
                    structured_data = json.loads(json_content)
                else:
                    raise ValueError("Could not locate JSON content in the response")
            
            # Ensure all required fields are present with proper defaults
            for field, default_value in default_required_fields.items():
                if field not in structured_data:
                    structured_data[field] = default_value
                elif isinstance(default_value, dict):
                    for sub_field, sub_default in default_value.items():
                        if sub_field not in structured_data[field]:
                            structured_data[field][sub_field] = sub_default
                        elif field == "content_info" and not isinstance(structured_data[field], list):
                            structured_data[field] = []
                        elif not structured_data[field][sub_field]:
                            structured_data[field][sub_field] = sub_default
                elif field == "content_info" and not isinstance(structured_data[field], list):
                    structured_data[field] = []
                elif not structured_data[field]:
                    structured_data[field] = default_value
            
            # Add uploaded_at if not present
            if "uploaded_at" not in structured_data:
                structured_data["uploaded_at"] = current_date
                
            return structured_data
        except Exception as json_error:
            logger.error(f"Failed to decode JSON from Azure OpenAI response: {str(json_error)}")
            # Log more details about the response for debugging
            logger.debug(f"Response content: {structured_content}")
            
            return {
                "summary": "Không thể phân tích được nội dung JSON",
                "client_info": default_required_fields["client_info"],
                "contract_info": default_required_fields["contract_info"],
                "advisory": "Không thể phân tích tài liệu do lỗi định dạng",
                "document_type": document_type or "Không xác định",
                "uploaded_at": current_date,
                "content_info": [],
                "error_details": str(json_error)
            }
            
    except Exception as e:
        logger.error(f"Error in extract_structured_data: {str(e)}")
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "summary": f"Lỗi khi phân tích: {str(e)}",
            "client_info": default_required_fields["client_info"],
            "contract_info": default_required_fields["contract_info"],
            "advisory": "Không thể phân tích tài liệu do lỗi kỹ thuật",
            "document_type": document_type or "Không xác định",
            "uploaded_at": current_date,
            "content_info": [],
            "error": str(e)
        }

async def process_document(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Process uploaded document using Azure Document Intelligence and Azure OpenAI.
    Returns structured data suitable for document analysis and management.
    """
    try:
        logger.info(f"Processing document {document_id} from {file_path}")

        # Initialize Document Intelligence client
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

        # Read the document content
        with open(file_path, "rb") as f:
            document_content = f.read()
        logger.info(f"Read {len(document_content)} bytes from {file_path}")

        # Analyze document using prebuilt-read model
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=document_content
        )
        logger.info(f"Started document analysis for {document_id}")

        # Wait for the result
        result: AnalyzeResult = poller.result()
        logger.info(f"Completed analysis for {document_id}")

        # Get raw text content for OpenAI processing
        raw_text = result.content if result.content else "No text extracted"
        
        # Extract document type from filename or content analysis
        document_type = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract structured data using Azure OpenAI
        structured_data = await extract_structured_data(raw_text, document_type)
        
        # Enhance structured data with handwriting information
        if result.pages and result.styles:
            content_info = structured_data.get("content_info", [])
            page_number = 1
            
            # Create mapping of content to handwriting status
            for page in result.pages:
                for line in page.lines:
                    is_handwritten = is_line_handwritten(line, page.words, result.styles)
                    line_text = line.content.strip()
                    
                    if line_text:
                        # Try to find if this line matches any content in our structured data
                        for item in content_info:
                            if line_text in item.get("content", ""):
                                # Update the item with handwriting information if not already present
                                if "is_handwritten" not in item:
                                    item["is_handwritten"] = is_handwritten
                                    
                                # Ensure position is set
                                if "position" not in item or not item["position"]:
                                    item["position"] = str(page_number)
                                    
                page_number += 1
        
        # Add document_id and status to the structured data
        structured_data["document_id"] = document_id
        structured_data["status"] = "completed"
        
        # Add uploaded_at timestamp if not present
        if "uploaded_at" not in structured_data:
            structured_data["uploaded_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Save structured data to JSON file
        processed_file_path = os.path.join(PROCESSED_DATA_DIRECTORY, f"{document_id}.json")
        with open(processed_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Structured data saved to {processed_file_path}")

        return structured_data
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        error_data = {
            "document_id": document_id,
            "status": "failed",
            "summary": f"Error processing document: {str(e)}",
            "client_info": {
                "Họ tên": "Không xác định",
                "CCCD/CMND/Hộ chiếu": "Không xác định",
                "Giới tính": "Không xác định",
                "Ngày tháng năm sinh": "Không xác định",
                "Địa chỉ cư trú": "Không xác định",
                "Mã QR code trên CCCD": "N/A",
                "Ngày cấp": "Không xác định",
                "Nơi cấp": "Không xác định",
                "Đơn vị cấp": "Không xác định",
                "Hình ảnh": "N/A",
                "Nguyên quán": "Không xác định",
                "Mối quan hệ": "Không xác định",
                "Dân tộc": "Không xác định"
            },
            "contract_info": {
                "Công chứng viên ký": "Không xác định",
                "Ngày tháng ký": "Không xác định",
                "Số công chứng": "Không xác định",
                "Địa chỉ làm hồ sơ": "Không xác định",
                "Quyển lưu": "Không xác định",
                "Giá trị tài sản giao dịch": 0,
                "contract info summarize": "Không có thông tin hợp đồng",
            },
            "advisory": "Document analysis failed due to technical issues",
            "document_type": os.path.splitext(os.path.basename(file_path))[0],
            "uploaded_at": current_date,
            "content_info": [],
            "error": str(e)
        }
        processed_file_path = os.path.join(PROCESSED_DATA_DIRECTORY, f"{document_id}.json")
        with open(processed_file_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=4)
        return error_data


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    """
    document_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_location = os.path.join(UPLOAD_DIRECTORY, f"{document_id}{file_extension}")

    try:
        # Save uploaded file
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"Saved uploaded file to {file_location}")

        # Process document asynchronously
        processed_data = await process_document(file_location, document_id)

        return JSONResponse(content={
            "document_id": document_id,
            "filename": file.filename,
            "message": "File uploaded and processed",
            "status": processed_data["status"]
        })

    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/processed/{document_type}/{document_id}")
async def get_processed_data(document_type: str, document_id: str):
    """
    Get the processed data for a document.
    """
    try:
        processed_file = os.path.join(PROCESSED_DATA_DIRECTORY, f"{document_id}.json")
        if not os.path.exists(processed_file):
            raise HTTPException(status_code=404, detail=f"No processed data found for document ID {document_id}")
            
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        logger.info(f"Retrieved processed data for document ID {document_id}")
        return JSONResponse(content=processed_data)
    except Exception as e:
        logger.error(f"Error reading processed data for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading processed data: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "FastAPI OCR Document Processor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)