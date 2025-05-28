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
    Use Azure OpenAI to extract structured data from raw text according to our schema.
    """
    try:
        logger.info("Extracting structured data using Azure OpenAI")        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key
        )
        
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Prepare the system message with instructions
        system_message = """
        Analyze the document text and extract the following information in JSON format:
        
        1. summary: A concise summary of the main points in the document
        2. client_info: Information about the client or subject (name, contact details, ID numbers)
        3. contract_info: Details about any contracts mentioned (type, number, date, parties)
        4. advisory: Any recommendations or warnings based on the document content
        5. document_type: The type of document (based on content analysis)
        6. content_info: An array of key information points with their positions and labels
        
        Return only the JSON with these fields, no other text.
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
              # Ensure the response has the required structure with proper default values
            required_fields = {
                "summary": "Không có tóm tắt",
                "client_info": "Không có thông tin khách hàng",
                "contract_info": "Không có thông tin hợp đồng",
                "advisory": "Không có khuyến nghị",
                "document_type": document_type or "Không xác định",
                "content_info": []
            }
            
            for field, default_value in required_fields.items():
                if field not in structured_data:
                    structured_data[field] = default_value
                elif field == "content_info" and not isinstance(structured_data[field], list):
                    # Ensure content_info is always a list
                    structured_data[field] = []
                elif field != "content_info" and not structured_data[field]:
                    # Replace empty strings with default values
                    structured_data[field] = default_value
                    
            # Add uploaded_at if not present
            if "uploaded_at" not in structured_data:
                structured_data["uploaded_at"] = current_date
                
            return structured_data
              except json.JSONDecodeError:
            logger.error("Failed to decode JSON from Azure OpenAI response")
            # Return a basic structure if JSON parsing fails
            return {
                "summary": "Không thể phân tích được nội dung JSON",
                "client_info": "Không có thông tin khách hàng",
                "contract_info": "Không có thông tin hợp đồng",
                "advisory": "Không thể phân tích tài liệu do lỗi định dạng",
                "document_type": document_type or "Không xác định",
                "uploaded_at": current_date,
                "content_info": []
            }
            
    except Exception as e:
        logger.error(f"Error in extract_structured_data: {str(e)}")
        # Return a basic structure on error
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "summary": f"Lỗi khi phân tích: {str(e)}",
            "client_info": "Không có thông tin khách hàng",
            "contract_info": "Không có thông tin hợp đồng",
            "advisory": "Không thể phân tích tài liệu do lỗi kỹ thuật",
            "document_type": document_type or "Không xác định",
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
        # This is a simplified approach, you might want a more sophisticated method
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
                                    item["position"] = f"page {page_number}"
                                    
                page_number += 1
        
        # Add document_id to the structured data
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

        return structured_data    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        error_data = {
            "document_id": document_id,
            "status": "failed",
            "summary": f"Error processing document: {str(e)}",
            "client_info": "",
            "contract_info": "",
            "advisory": "Document analysis failed due to technical issues",
            "document_type": os.path.splitext(os.path.basename(file_path))[0],
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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