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
from typing import Dict, Any
import azure.ai.documentintelligence
import numpy as np

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

async def process_document(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Process uploaded document using Azure Document Intelligence.
    Returns processed data with line and word handwritten flags and line confidence.
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

        # Extract relevant information
        processed_data = {
            "document_id": document_id,
            "status": "completed",
            "raw_text": result.content if result.content else "No text extracted",
            "styles": [
                {
                    "is_handwritten": style.is_handwritten,
                    "index": idx,
                    "spans": [
                        {"offset": span.offset, "length": span.length}
                        for span in style.spans
                    ]
                } for idx, style in enumerate(result.styles)
            ] if result.styles else [],
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit,
                    "lines": [
                        {
                            "text": line.content,
                            "bounding_box": format_bounding_box(line.polygon),
                            "confidence": calculate_line_confidence(line, page.words) if page.words else 0.0,
                            "is_handwritten": is_line_handwritten(line, page.words, result.styles)
                        } for line in page.lines
                    ],
                    "words": [
                        {
                            "text": word.content,
                            "confidence": word.confidence,
                            "bounding_box": format_bounding_box(word.polygon),
                            "is_handwritten": is_word_handwritten(word, result.styles)
                        } for word in page.words
                    ] if page.words else []
                } for page in result.pages
            ] if result.pages else []
        }

        # Save processed data to JSON file
        processed_file_path = os.path.join(PROCESSED_DATA_DIRECTORY, f"{document_id}.json")
        with open(processed_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Processed data saved to {processed_file_path}")

        return processed_data

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        error_data = {
            "document_id": document_id,
            "status": "failed",
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