import os
import shutil
import uuid
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
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
        5. document_type: The type of document (based on content analysis)Trường này xác định loại tài liệu của hồ sơ, dựa trên các danh mục được định nghĩa sẵn trong hệ thống phải giống với các loại sau không được tự ý chỉnh sửa tên các loại này: 
        01. NHÀ ĐẤT – CHUNG CƯ
        1. Mua bán nhà đất
        2. Mua bán nhà đất (một phần)
        3. Tặng cho nhà đất
        4. Tặng cho nhà đất (một phần)
        5. Thuê nhà
        6. Mượn nhà
        7. Ở nhờ
        8. Chuyển nhượng đất
        9. Chuyển nhượng đất (một phần)
        10. Tặng cho đất
        11. Tặng cho đất (một phần)
        12. Thuê quyền sử dụng đất
        13. Mua bán căn hộ chung cư
        14. Mua bán căn hộ chung cư (một phần)
        15. Tặng cho căn hộ chung cư
        16. Tặng cho căn hộ chung cư (một phần)
        17. Chuyển nhượng nhà đất
        18. Chuyển nhượng nhà đất (một phần)
        19. Tặng cho nhà đất
        20. Tặng cho nhà đất (một phần)
        21. Chuyển nhượng tài sản gắn liền với đất
        22. Chuyển nhượng tài sản gắn liền với đất (một phần)
        23. Tặng cho tài sản gắn liền với đất
        24. Tặng cho tài sản gắn liền với đất (một phần)
        25. Đặt cọc

        02. ỦY QUYỀN
        26. Hợp đồng ủy quyền (mẫu chung)
        27. Hợp đồng ủy quyền nhà đất
        28. Hợp đồng ủy quyền căn hộ chưa sổ
        29. Hợp đồng ủy quyền quyền sử dụng đất
        30. Hợp đồng ủy quyền thừa kế
        31. Hợp đồng ủy quyền thừa kế thụ ủy
        32. Hợp đồng ủy quyền hộ gia đình
        33. Hợp đồng ủy quyền hộ gia đình thụ ủy
        34. Hợp đồng ủy quyền quản lý doanh nghiệp
        35. Hợp đồng ủy quyền chứng khoán
        36. Hợp đồng ủy quyền thụ ủy
        37. Giấy ủy quyền (mẫu)
        38. Giấy ủy quyền đăng bộ trước bạ
        39. Giấy ủy quyền nộp thuế căn hộ
        40. Giấy ủy quyền thành lập doanh nghiệp
        41. Giấy ủy quyền thành lập hộ kinh doanh
        42. Giấy ủy quyền thành lập Doanh Nghiệp nước ngoài
        43. Giấy ủy quyền tham gia tố tụng
        44. Giấy ủy quyền đưa con đi máy bay
        45. Giấy ủy quyền đăng ký xe
        46. Giấy ủy quyền điện nước
        47. Giấy ủy quyền tiền bảo hiểm
        48. Giấy ủy quyền tiền tử tuất
        49. Giấy ủy quyền chứng thực (mẫu)

        03. VĂN BẢN CHUYỂN NHƯỢNG
        50. Văn bản chuyển nhượng
        51. Văn bản chuyển nhượng officetel

        04. DI SẢN THỪA KẾ
        52. Thông báo niêm yết - Di sản thừa kế
        53. Thông báo niêm yết - Di chúc
        54. Phân chia di sản
        55. Khai nhận di sản
        56. Khai nhận di sản theo di chúc
        57. Di chúc
        58. Văn bản từ chối nhận di sản

        05. XE
        59. Ủy quyền xe ô tô
        60. Ủy quyền xe ô tô - ủy quyền lại
        61. Mua bán xe ô tô
        62. Mua bán xe máy
        63. Thuê xe
        64. Mượn xe

        06. DOANH NGHIỆP
        65. Chuyển nhượng cổ phần doanh nghiệp
        66. Tặng cho cổ phần doanh nghiệp
        67. Chuyển nhượng góp vốn doanh nghiệp
        68. Tặng cho góp vốn doanh nghiệp

        07. TÀI SẢN VỢ CHỒNG
        69. Cam kết tài sản riêng - chứng thực
        70. Văn bản tài sản riêng
        71. Văn bản phân chia tài sản
        72. Văn bản phân chia tài sản sau ly hôn
        73. Văn bản tài sản riêng - ly hôn
        74. Văn bản nhập tài sản
        75. Văn bản đưa tài sản vào kinh doanh
        76. Văn bản tài sản riêng trước hôn nhân

        08. SỬA ĐỔI, HỦY BỎ
        77. Sửa đổi _ mẫu
        78. Hủy bỏ _ mẫu
        79. Chấm dứt _ mẫu

        09. LỜI CHỨNG
        80. Lời chứng _ mẫu
        81. Lời chứng thế chấp _ mẫu
        82. Lời chứng nhận ủy quyền _ mẫu
        83. Lời chứng chứng thực _ mẫu
        84. Lời chứng dịch thuật – song ngữ
        85. Lời chứng dịch thuật _ mẫu

        10. HỢP ĐỒNG GIAO DỊCH KHÁC
        86. Tặng cho tài sản _ mẫu
        87. Hợp đồng vay tiền
        88. Hợp đồng Góp vốn
        89. Hợp đồng Góp vốn – tiền _ mẫu
        90. Hợp đồng Hợp tác kinh doanh

        Bảo Lãnh
        BL: Bảo lãnh
        BL_GC: Giải chấp một phần (có soạn thảo hợp đồng sửa đổi bổ sung)
        BL_GCK: Giải chấp không soạn văn bản

        Cầm cố
        CC_B: Cầm cố vay bổ sung
        CC_C: Cầm cố
        CC_GC: Giải phấp 1 phần (có soạn thảo hợp đồng)
        CC_GCK: Giải chấp không soạn văn bản
        CC_S: Sửa đổi, bổ sung HĐ cầm cố
        CC_T: Thanh lý HĐ cầm cố

        Chuyển đổi - Trao đổi
        CD_C: HĐ chuyển đổi, trao đổi
        CD_H: Hủy bỏ HĐ chuyển đổi, trao đổi
        CD_S: Sửa đổi, bổ sung

        Chuyển nhượng - Mua bán
        CN_C: HĐ mua bán, chuyển nhượng
        CN_DC: Đặt cọc
        CN_H: Hủy bỏ mua bán, chuyển nhượng
        CN_HD: Hủy bỏ HĐ đặt cọc
        CN_HS: Hủy bỏ HĐ sửa đổi, bổ sung HĐ mua bán, chuyển nhượng
        CN_S: Sửa đổi, bổ sung
        CN_SC: Sửa đổi, bổ sung HĐ đặt cọc
        CN_TDC: Thanh lý HĐ đặt cọc

        Di chúc
        DC_D: Di chúc
        DC_H: Hủy bỏ Di chúc
        DC_S: Sửa đổi, bổ sung

        Góp vốn
        GV_G: HĐ góp vốn
        GV_H: Hủy bỏ HĐ góp vốn
        GV_S: Sửa đổi, bổ sung HĐ góp vốn

        Giao dịch khác
        K_H: Hủy bỏ, thanh lý
        K_K: HĐ, giao dịch khác
        K_S: Sửa đổi, bổ sung

        Tặng - cho
        TC_H: Hủy bỏ tặng cho
        TC_S: Sửa đổi, bổ sung
        TC_TC: HĐ tặng cho

        Thế chấp
        THC_D3: Thế chấp đảm bảo nghĩa vụ bên thứ 3
        THC_GC: Giải chấp một phần (có soạn thảo hợp đồng sửa đổi bổ sung)
        THC_GCK: Giải chấp không soạn văn bản
        THC_S: Sửa đổi, bổ sung HĐ thế chấp
        THC_S3: Sửa đổi bổ sung HĐ thế chấp đảm bảo nghĩa vụ bên thứ 3
        THC_TC: Thế chấp
        THC_V: Thế chấp vay bổ sung
        THC_V3: Thế chấp vay bổ sung đảm bảo nghĩa vụ bên thứ 3
        THC_TL: Thanh lý HĐ thế chấp

        Thừa kế
        TK_DK: Văn bản thỏa thuận về hoàn tất thủ tục đăng ký thừa kế
        TK_GCN: Văn bản thỏa thuận đại diện đứng tên trên giấy chứng nhận (GCN)
        TK_H: Hủy bỏ
        TK_KN: Khai nhận di sản thừa kế
        TK_S: Sửa đổi, bổ sung
        TK_TC: Từ chối nhận di sản
        TK_TT: Thỏa thuận phân chia di sản thừa kế

        Thuê mượn
        TM_S: Sửa đổi, bổ sung
        TM_TL: Thanh lý HĐ thuê mượn
        TM_TM: HĐ thuê, mượn

        Ủy quyền
        UQ_CD: Thỏa thuận chấm dứt Ủy quyền
        UQ_DPH: Đơn phương chấm dứt UQ
        UQ_H: Hủy bỏ ủy quyền
        UQ_S: Sửa đổi, bổ sung
        UQ_UQ: Ủy quyền

        Tài sản vợ chồng
        VC_C: Chia tài sản vợ chồng
        VC_CC: Chia tài sản chung
        VC_CK: Cam kết tài sản
        VC_CL: Chia tài sản sau ly hôn
        VC_HC: Hủy bỏ chia tài sản chung
        VC_HN: Hủy bỏ nhập tài sản riêng vào tài sản chung
        VC_HT: Hủy bỏ thỏa thuận tài sản riêng
        VC_KP: Khôi phục CĐTS chung
        VC_N: Nhập tài sản riêng vào tài sản chung
        VC_SC: Sửa đổi bổ sung chia tài sản chung
        VC_TR: Thỏa thuận tài sản riêng

        Vay
        V_S: Sửa đổi bổ sung
        V_TL: Thanh lý hợp đồng vay
        V_TLTCH: Thanh lý hợp đồng vay và thế chấp tài sản
        V_V: HĐ vay
        V_VTCH: Hợp đồng vay và thế chấp tài sản


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
            max_completion_tokens=10000,
            model=deployment
        )
        
        # Extract the response content
        structured_content = response.choices[0].message.content
        
        try:
            # Try to parse the response as JSON
            structured_data = json.loads(structured_content)
            
            # Define default values for required fields
            required_fields = {
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
                "Thông tin Tài sản": {
                    "Thửa đất": "Không xác định",
                    "Tờ bản đồ": "Không xác định",
                    "Diện tích đất": "Không xác định",
                    "Diện tích sử dụng chung": "N/A",
                    "Diện tích sử dụng riêng": "N/A",
                    "Mục đích sử dụng đất": "Không xác định",
                    "Thời hạn sử dụng đất": "Không xác định",
                    "Nguồn gốc đất": "Không xác định",
                    "Số cấp Giấy Chứng Nhận": "Không xác định",
                    "Số phát hành Giấy Chứng Nhận": "Không xác định",
                    "Nơi cấp Giấy Chứng Nhận": "Không xác định",
                    "Ngày cấp Giấy Chứng Nhận": "Không xác định",
                    "Số nhà": "Không xác định",
                    "Địa chỉ": "Không xác định",
                    "Số căn hộ": "N/A",
                    "Diện tích xây dựng": "Không xác định",
                    "Loại nhà ở/công trình": "Không xác định",
                    "Tổng diện tích xây dựng": "Không xác định"
                },
                "Thông tin khác": "Không xác định",
                "content_info": []
            }
            
            # Ensure all required fields are present with proper defaults
            for field, default_value in required_fields.items():
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
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from Azure OpenAI response")
            return {
                "summary": "Không thể phân tích được nội dung JSON",
                "client_info": required_fields["client_info"],
                "contract_info": required_fields["contract_info"],
                "advisory": "Không thể phân tích tài liệu do lỗi định dạng",
                "document_type": document_type or "Không xác định",
                "Thông tin Tài sản": required_fields["Thông tin Tài sản"],
                "Thông tin khác": "Không xác định",
                "uploaded_at": current_date,
                "content_info": []
            }
            
    except Exception as e:
        logger.error(f"Error in extract_structured_data: {str(e)}")
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "summary": f"Lỗi khi phân tích: {str(e)}",
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
            "advisory": "Không thể phân tích tài liệu do lỗi kỹ thuật",
            "document_type": document_type or "Không xác định",
            "Thông tin Tài sản": {
                "Thửa đất": "Không xác định",
                "Tờ bản đồ": "Không xác định",
                "Diện tích đất": "Không xác định",
                "Diện tích sử dụng chung": "N/A",
                "Diện tích sử dụng riêng": "N/A",
                "Mục đích sử dụng đất": "Không xác định",
                "Thời hạn sử dụng đất": "Không xác định",
                "Nguồn gốc đất": "Không xác định",
                "Số cấp Giấy Chứng Nhận": "Không xác định",
                "Số phát hành Giấy Chứng Nhận": "Không xác định",
                "Nơi cấp Giấy Chứng Nhận": "Không xác định",
                "Ngày cấp Giấy Chứng Nhận": "Không xác định",
                "Số nhà": "Không xác định",
                "Địa chỉ": "Không xác định",
                "Số căn hộ": "N/A",
                "Diện tích xây dựng": "Không xác định",
                "Loại nhà ở/công trình": "Không xác định",
                "Tổng diện tích xây dựng": "Không xác định"
            },
            "Thông tin khác": "Không xác định",
            "uploaded_at": current_date,
            "content_info": []
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
            "Thông tin Tài sản": {
                "Thử đất": "Không xác định",
                "Tờ bản đồ": "Không xác định",
                "Diện tích đất": "Không xác định",
                "Diện tích sử dụng chung": "N/A",
                "Diện tích sử dụng riêng": "N/A",
                "Mục đích sử dụng đất": "Không xác định",
                "Thời hạn sử dụng đất": "Không xác định",
                "Nguồn gốc đất": "Không xác định",
                "Số cấp Giấy Chứng Nhận": "Không xác định",
                "Số phát hành Giấy Chứng Nhận": "Không xác định",
                "Nơi cấp Giấy Chứng Nhận": "Không xác định",
                "Ngày cấp Giấy Chứng Nhận": "Không xác định",
                "Số nhà": "Không xác định",
                "Địa chỉ": "Không xác định",
                "Số căn hộ": "N/A",
                "Diện tích xây dựng": "Không xác định",
                "Loại nhà ở/công trình": "Không xác định",
                "Tổng diện tích xây dựng": "Không xác định"
            },
            "Thông tin khác": "Không xác định",
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
    Retrieve extracted data for a processed document.
    """
    processed_file_path = os.path.join(PROCESSED_DATA_DIRECTORY, f"{document_id}.json")

    if not os.path.exists(processed_file_path):
        logger.warning(f"Processed data not found for document ID {document_id}")
        raise HTTPException(status_code=404, detail="Processed data not found")

    try:
        with open(processed_file_path, "r", encoding="utf-8") as f:
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