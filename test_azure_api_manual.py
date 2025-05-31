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
            temperature=0.2,
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
