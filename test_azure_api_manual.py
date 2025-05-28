#!/usr/bin/env python
"""
Script to test Azure API integrations with real credentials.
This script will perform live API calls to Azure services using your configured credentials.
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test Azure APIs with specific JSON file')
parser.add_argument('--file', type=str, help='Specific JSON file to test with (filename only)')
parser.add_argument('--test', type=int, choices=[1, 2, 3, 4, 5], help='Test number to run (1-5)')
args = parser.parse_args()

# Path to existing processed data
DEFAULT_DATA_FILENAME = "67296de8-df32-4adc-898e-ba2db7c237a2.json"
EXISTING_DATA_PATH = os.path.join("processed_data", args.file if args.file else DEFAULT_DATA_FILENAME)

# Load environment variables
load_dotenv()

# Check if sample test PDF exists, or create one
SAMPLE_PDF_PATH = "sample_test_document.pdf"

def create_sample_pdf():
    """Create a simple PDF for testing if one doesn't exist"""
    if os.path.exists(SAMPLE_PDF_PATH):
        logger.info(f"Using existing sample PDF: {SAMPLE_PDF_PATH}")
        return
        
    try:
        # Try to create a simple PDF using reportlab if available
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(SAMPLE_PDF_PATH)
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 730, "Chủ Sở Hữu: Nguyễn Văn A")
        c.drawString(100, 710, "Sinh năm: 1980")
        c.drawString(100, 690, "CMND/CCCD: 123456789012")
        c.drawString(100, 670, "Diện tích: 180m2")
        c.drawString(100, 650, "Địa chỉ: 123 Đường Lê Lợi, Phường Bến Nghé, Quận 1, TP.HCM")
        c.save()
        logger.info(f"Created sample PDF: {SAMPLE_PDF_PATH}")
    except ImportError:
        # If reportlab is not available, create a minimal PDF
        with open(SAMPLE_PDF_PATH, "wb") as f:
            f.write(b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n180\n%%EOF")
            logger.info(f"Created minimal sample PDF: {SAMPLE_PDF_PATH} (ReportLab not available)")

async def test_document_intelligence():
    """Test Azure Document Intelligence with a real API call"""
    logger.info("Testing Azure Document Intelligence...")
    
    # Get credentials from environment
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    
    if not endpoint or not key:
        logger.error("Azure Document Intelligence credentials not configured in environment variables")
        return False
    
    try:
        # Initialize Document Intelligence client
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # Read document content
        with open(SAMPLE_PDF_PATH, "rb") as f:
            document_content = f.read()
        
        # Analyze document
        logger.info(f"Sending document to Azure Document Intelligence ({len(document_content)} bytes)...")
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=document_content
        )
        
        # Wait for the result
        logger.info("Waiting for analysis to complete...")
        result = poller.result()
        
        # Output basic information about the result
        logger.info(f"Document analysis complete. Extracted content length: {len(result.content or '')}")
        logger.info(f"Number of pages detected: {len(result.pages or [])}")
        
        # Output sample of the content
        content_preview = (result.content or "")[:500]
        logger.info(f"Content preview: {content_preview}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Document Intelligence: {str(e)}", exc_info=True)
        return False

async def test_azure_openai_with_existing_data():
    """Test Azure OpenAI with real data from an existing processed file"""
    logger.info("Testing Azure OpenAI with existing processed data...")
    
    try:
        # Check if the existing data file exists
        if not os.path.exists(EXISTING_DATA_PATH):
            logger.error(f"Existing data file not found: {EXISTING_DATA_PATH}")
            return False
            
        # Load the existing processed data
        with open(EXISTING_DATA_PATH, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        if "raw_text" not in existing_data:
            logger.error("Raw text not found in existing data file")
            return False
            
        logger.info(f"Loaded existing data file. Document ID: {existing_data.get('document_id', 'unknown')}")
        logger.info(f"Text length: {len(existing_data['raw_text'])} characters")
        
        # Extract raw text from the existing data
        raw_text = existing_data["raw_text"]
        
        # Limit to first 2000 characters for logging
        text_preview = raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text
        logger.info(f"Document content preview:\n{text_preview}")        # Get credentials from environment
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not azure_endpoint or not api_key:
            logger.error("Azure OpenAI endpoint or key not configured in environment variables")
            return False
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key
        )
        
        # System message for structured data extraction
        system_message = """
        Điền tất cả thông tin bằng tiêng Việt.
        Analyze the document text and extract the following information in JSON format:
        
        1. summary: A concise summary of the main points in the document , Trường này chứa nội dung tóm tắt các ý chính của hồ sơ, được hệ thống tự động tạo ra thông qua trí tuệ nhân tạo (AI). AI phân tích toàn bộ nội dung tài liệu để trích xuất các thông tin quan trọng nhất, bao gồm nội dung chính của tài liệu, thông tin về đương sự (họ tên, địa chỉ, số điện thoại), thông tin hợp đồng (nếu có), và các điểm đáng chú ý khác. Mục đích của trường này là cung cấp một cái nhìn tổng quan nhanh chóng, giúp nhân viên kiểm soát theo dõi tình trạng hồ sơ mà không cần đọc toàn bộ tài liệu chi tiết. Trường này cũng hỗ trợ nhân viên thống kê trong việc tổng hợp dữ liệu để xuất báo cáo hoặc phân tích, đồng thời cho phép Super Admin dễ dàng xem xét tổng quan khi quản lý hệ thống hoặc phân quyền truy cập cho nhân viên. Tóm tắt được tạo dựa trên tính năng "Đưa ra những ý chính trong nội dung" và "Tóm tắt ý chính về nội dung, hợp đồng, vị trí các ý" của dự án, đảm bảo nội dung ngắn gọn nhưng đầy đủ thông tin cần thiết để hỗ trợ các vai trò quản lý và xử lý hồ sơ.
        2. client_info: Information about the client or subject Trường này lưu trữ thông tin chi tiết về đương sự dưới dạng một chuỗi (string) Trường này lưu trữ thông tin chi tiết về đương sự hoặc khách hàng có liên quan đến hồ sơ. Thông tin bao gồm các dữ liệu nhận dạng cá nhân như họ tên, ngày sinh, địa chỉ, số điện thoại, số căn cước công dân, số hộ chiếu, hoặc bất kỳ thông tin nào khác được bóc tách từ các giấy tờ do nhà nước cấp (căn cước công dân, chứng minh nhân dân, hộ chiếu, giấy chứng nhận kết hôn, sổ đỏ, v.v.). Dữ liệu trong trường này được trích xuất tự động bằng công nghệ OCR (đối với tài liệu in) hoặc ICR (đối với chữ viết tay), sau đó AI phân loại và nhập vào các trường phù hợp dựa trên tính năng "Phân loại nội dung như Số điện thoại, Tên người, Địa chỉ". Trường này hỗ trợ nhân viên sử dụng trong việc upload và quản lý hồ sơ cá nhân theo từng nhân viên hoặc phòng ban, đồng thời giúp nhân viên kiểm soát xác minh thông tin đương sự khi theo dõi hồ sơ đã upload. Ngoài ra, Super Admin có thể sử dụng thông tin trong trường này để phân quyền truy cập hồ sơ cho nhân viên, đảm bảo chỉ những người được phép mới có thể xem hoặc chỉnh sửa thông tin nhạy cảm.
        3. contract_info: Dạng json
        nó sẽ bao gồm :
            Trường này lưu trữ thông tin chi tiết về hợp đồng dưới dạng một đối tượng JSON, giới hạn ở 7 trường sau:
            Công chứng viên ký: Tên hoặc thông tin về công chứng viên thực hiện việc công chứng hợp đồng.
            Ngày tháng ký: Ngày tháng công chứng hợp đồng.
            Số công chứng: Số chứng nhận công chứng hợp đồng.
            Quyển lưu: Số quyển lưu hoặc sổ lưu trữ hợp đồng tại cơ quan công chứng.
            contract info summarize: Details about any contracts mentioned Trường này lưu trữ thông tin chi tiết về hợp đồng dưới dạng một chuỗi (string). Trường này lưu trữ các thông tin chi tiết liên quan đến hợp đồng nếu hồ sơ có chứa hợp đồng, bao gồm loại hợp đồng (hợp đồng công chứng, hợp đồng luật), số hợp đồng, ngày ký, các bên liên quan, và thông tin về chữ ký số (được tích hợp từ VNPT-CA hoặc Viettel-CA nếu có). Dữ liệu được trích xuất tự động bằng OCR/ICR và AI từ các tài liệu hợp đồng, sau đó được phân loại và lưu trữ để hỗ trợ quản lý. Tính năng "Ghi nhận chữ ký số vào trong hợp đồng công chứng / hợp đồng luật" được áp dụng để đảm bảo tính pháp lý của hợp đồng trong hệ thống. Trường này hỗ trợ nhân viên kiểm soát trong việc theo dõi tình trạng hợp đồng và quản lý các hồ sơ liên quan đến hợp đồng, đồng thời giúp nhân viên thống kê khi cần xuất báo cáo hoặc thống kê dữ liệu hợp đồng theo yêu cầu. Super Admin cũng có thể sử dụng thông tin từ trường này để phân quyền truy cập hoặc giới hạn quyền chỉnh sửa hồ sơ hợp đồng cho các nhân viên, đảm bảo tính bảo mật và tuân thủ quy định.
            Thông tin bên A: Thông tin về bên A tham gia hợp đồng, bao gồm danh tính hoặc thông tin pháp lý của bên A. Bên A có thể là một người, nhiều người, hoặc một tổ chức.
            Thông tin bên B: Thông tin về bên B tham gia hợp đồng, bao gồm danh tính hoặc thông tin pháp lý của bên B. Bên B có thể là một người, nhiều người, hoặc một tổ chức. Trường này lưu trữ các thông tin chi tiết liên quan đến hợp đồng nếu hồ sơ có chứa hợp đồng, bao gồm loại hợp đồng (hợp đồng công chứng, hợp đồng luật), số hợp đồng, ngày ký, các bên liên quan (Bên A và Bên B), và thông tin về chữ ký số (được tích hợp từ VNPT-CA hoặc Viettel-CA nếu có). Dữ liệu được trích xuất tự động bằng OCR/ICR và AI từ các tài liệu hợp đồng, sau đó được phân loại và lưu trữ để hỗ trợ quản lý. Tính năng "Ghi nhận chữ ký số vào trong hợp đồng công chứng / hợp đồng luật" được áp dụng để đảm bảo tính pháp lý của hợp đồng trong hệ thống. Trường này hỗ trợ nhân viên kiểm soát trong việc theo dõi tình trạng hợp đồng và quản lý các hồ sơ liên quan đến hợp đồng, đồng thời giúp nhân viên thống kê khi cần xuất báo cáo hoặc thống kê dữ liệu hợp đồng theo yêu cầu. Super Admin cũng có thể sử dụng thông tin từ trường này để phân quyền truy cập hoặc giới hạn quyền chỉnh sửa hồ sơ hợp đồng cho các nhân viên, đảm bảo tính bảo mật và tuân thủ quy định.

        4. advisory: Any recommendations or warnings based on the document content Trường này chứa các khuyến nghị hoặc khuyến cáo liên quan đến hồ sơ, được hệ thống tự động tạo ra dựa trên phân tích nội dung bằng AI. Các khuyến nghị có thể bao gồm cảnh báo về các vấn đề pháp lý (như thiếu chữ ký, thông tin không đầy đủ, hoặc tài liệu hết hạn) hoặc gợi ý các hành động cần thực hiện (như bổ sung giấy tờ, xác minh thông tin, hoặc gia hạn tài liệu). Tính năng "Đưa ra khuyến nghị, khuyến cáo" của dự án được sử dụng để tạo nội dung cho trường này, đảm bảo các vấn đề tiềm ẩn được phát hiện sớm. Trường này hỗ trợ nhân viên kiểm soát trong việc đánh giá chất lượng hồ sơ và đảm bảo tuân thủ các quy định pháp lý hoặc quy trình nội bộ. Nhân viên sử dụng cũng được hưởng lợi khi sử dụng thông tin từ trường này để đưa ra quyết định xử lý hồ sơ một cách hiệu quả. Super Admin có thể tham khảo các khuyến nghị để đánh giá hiệu quả quản lý hồ sơ của nhân viên hoặc điều chỉnh phân quyền nếu phát hiện các vấn đề lặp lại trong hệ thống.
        5. document_type: The type of document (based on content analysis)Trường này xác định loại tài liệu của hồ sơ, dựa trên các danh mục được định nghĩa sẵn trong hệ thống, chẳng hạn như công văn đi, công văn đến, tra cứu dữ liệu, mẫu dấu, mẫu chữ ký, công văn nội bộ, hồ sơ cá nhân, hoặc các giấy tờ do nhà nước cấp (căn cước công dân, sổ đỏ, giấy chứng nhận kết hôn, bằng lái xe, giấy xác nhận độc thân, v.v.). Dữ liệu được phân loại tự động bởi hệ thống thông qua việc sử dụng OCR và AI để nhận diện các đặc điểm của tài liệu, áp dụng tính năng "Phân loại hồ sơ theo phòng ban, loại hồ sơ". Trường này hỗ trợ nhân viên sử dụng trong việc upload và tổ chức hồ sơ theo từng loại tài liệu hoặc phòng ban, giúp nhân viên thống kê dễ dàng thực hiện các tác vụ thống kê dữ liệu hồ sơ hoặc xuất báo cáo theo yêu cầu. Nhân viên kiểm soát sử dụng trường này để theo dõi số lượng và tình trạng hồ sơ theo từng loại tài liệu, đảm bảo quản lý hiệu quả. Super Admin có thể dựa vào thông tin này để phân quyền truy cập hoặc quản lý tài khoản nhân viên dựa trên loại tài liệu mà họ được phép xử lý. còn nữa thêm note phía sau nếu có giấy tờ pháp lý liên quan Giấy tờ pháp lý liên quan", liệt kê các loại giấy tờ pháp lý cụ thể mà hệ thống nhận diện, phân tách bằng dấu phẩy. Trường này hỗ trợ nhân viên thống kê trong việc quản lý dữ liệu hồ sơ
        6. content_info:
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
        
        # Use the actual document content from the existing data
        user_message = f"""
        Document Type: Figure out the type of document based on the content.
        Document Raw data:
        {raw_text}  

        Extract the information according to the required structure.
        """        # Call Azure OpenAI
        logger.info(f"Sending request to Azure OpenAI deployment '{deployment}'...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=10000,
            model=deployment
        )
        # Output the result
        content = response.choices[0].message.content
        logger.info(f"Azure OpenAI response received. Length: {len(content)}")
        
        # Try to parse as JSON and pretty print
        try:
            data = json.loads(content)
            
            # Save the response to a file for analysis
            output_file = f"openai_response_{existing_data.get('document_id', 'test')}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Response saved to {output_file}")
            
            # Log a pretty-printed version of the first part of the response
            pretty_json = json.dumps({
                "summary": data.get("summary", ""),
                "client_info": data.get("client_info", ""),
                "document_type": data.get("document_type", ""),
                "content_info_count": len(data.get("content_info", [])) 
            }, indent=4, ensure_ascii=False)
            
            logger.info(f"Parsed response overview:\n{pretty_json}")
            
            # Check if response has expected structure
            required_fields = ["summary", "client_info", "contract_info", "advisory", "document_type", "content_info"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                logger.warning(f"Response is missing the following fields: {', '.join(missing_fields)}")
            
        except json.JSONDecodeError:
            logger.error("Failed to parse response as JSON. Raw response:")
            logger.error(content)
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Azure OpenAI with existing data: {str(e)}", exc_info=True)
        return False

async def test_azure_openai():
    """Test Azure OpenAI with a simple message (without document analysis)"""
    logger.info("Testing basic Azure OpenAI connectivity...")    # Get credentials from environment
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not azure_endpoint or not api_key:
        logger.error("Azure OpenAI endpoint or key not configured in environment variables")
        return False
    
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key
        )        # Simple test message
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Verify that you can connect to the Azure OpenAI API."}
            ],
            max_completion_tokens=100,
            model=deployment
        )
        
        # Output the result
        content = response.choices[0].message.content
        logger.info(f"Azure OpenAI basic connectivity test response: {content}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Azure OpenAI basic connectivity: {str(e)}", exc_info=True)
        return False

async def test_end_to_end_with_existing_data():
    """Test end-to-end processing using existing raw data"""
    logger.info("Testing end-to-end processing with existing data...")
    
    try:
        # Check if the existing data file exists
        if not os.path.exists(EXISTING_DATA_PATH):
            logger.error(f"Existing data file not found: {EXISTING_DATA_PATH}")
            return False
            
        # Load the existing processed data
        with open(EXISTING_DATA_PATH, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        if "raw_text" not in existing_data:
            logger.error("Raw text not found in existing data file")
            return False
            
        # Extract raw text from the existing data
        raw_text = existing_data["raw_text"]
        document_id = existing_data.get("document_id", "test_id")
        document_type = "Giấy ủy quyền"  # Based on content analysis        # Get Azure OpenAI credentials
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        
        if not azure_endpoint or not api_key:
            logger.error("Azure OpenAI credentials not configured in environment variables")
            return False
            
        # Import the extract_structured_data function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from main import extract_structured_data
        
        # Call the extract_structured_data function
        logger.info("Calling extract_structured_data function...")
        result = await extract_structured_data(raw_text, document_type)
        
        # Save the result to a file
        output_file = f"end_to_end_result_{document_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.info(f"End-to-end processing result saved to {output_file}")
        
        # Log the result structure
        logger.info("End-to-end processing completed with the following structure:")
        logger.info(f"- Document ID: {document_id}")
        logger.info(f"- Document Type: {result.get('document_type', 'N/A')}")
        logger.info(f"- Summary length: {len(result.get('summary', ''))}")
        logger.info(f"- Content info items: {len(result.get('content_info', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in end-to-end processing test: {str(e)}", exc_info=True)
        return False

async def main():
    """Main function to run API tests"""
    logger.info("Starting Azure API tests with existing data...")
    logger.info(f"Using data file: {EXISTING_DATA_PATH}")
    
    # Use command-line argument or prompt for test choice
    if args.test:
        choice = args.test
        logger.info(f"Running test #{choice} from command line argument")
    else:
        # Offer test options to user
        print("\nAvailable Tests:")
        print("1. Basic Azure OpenAI connectivity test")
        print("2. Azure OpenAI test with existing document data")
        print("3. End-to-end processing with existing data")
        print("4. Document Intelligence test (requires sample PDF)")
        print("5. Run all tests")
        
        try:
            choice = int(input("\nSelect test to run (1-5): "))
        except ValueError:
            choice = 5  # Default to all tests
    
    # Initialize success flags
    openai_basic_success = False
    openai_doc_success = False
    end_to_end_success = False
    di_success = False
    
    # Run selected tests
    if choice in [1, 5]:
        openai_basic_success = await test_azure_openai()
        
    if choice in [2, 5]:
        openai_doc_success = await test_azure_openai_with_existing_data()
        
    if choice in [3, 5]:
        end_to_end_success = await test_end_to_end_with_existing_data()
        
    if choice in [4, 5]:
        create_sample_pdf()
        di_success = await test_document_intelligence()
      # Print summary
    logger.info("\n--- Test Results ---")
    if choice in [1, 5]:
        logger.info(f"Basic OpenAI: {'✅ PASSED' if openai_basic_success else '❌ FAILED'}")
    if choice in [2, 5]:
        logger.info(f"OpenAI with Document: {'✅ PASSED' if openai_doc_success else '❌ FAILED'}")
    if choice in [3, 5]:
        logger.info(f"End-to-end Processing: {'✅ PASSED' if end_to_end_success else '❌ FAILED'}")
    if choice in [4, 5]:
        logger.info(f"Document Intelligence: {'✅ PASSED' if di_success else '❌ FAILED'}")
    logger.info("-------------------\n")
    
    # Return overall status based on which tests were run
    if choice == 1:
        return openai_basic_success
    elif choice == 2:
        return openai_doc_success
    elif choice == 3:
        return end_to_end_success
    elif choice == 4:
        return di_success
    else:
        return all([t for t in [openai_basic_success, openai_doc_success, end_to_end_success, di_success] if t is not None])

if __name__ == "__main__":
    asyncio.run(main())
