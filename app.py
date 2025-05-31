from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio._client import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uuid
import os

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Document Intelligence credentials (replace with your actual endpoint and key)
# Consider using environment variables for production
endpoint = "YOUR_FORM_RECOGNIZER_ENDPOINT"
key = "YOUR_FORM_RECOGNIZER_KEY"

# In-memory storage for processed documents (for demonstration purposes)
# In a real application, you would use a database
processed_documents = {}

def format_bounding_box(bounding_box):
    if not bounding_box:
        return "N/A"
    reshaped_bounding_box = np.array(bounding_box).reshape(-1, 2)
    return ", ".join(["[{}, {}]".format(x, y) for x, y in reshaped_bounding_box])

async def analyze_document_intelligence(document_content):
    """
    Analyzes document content using Azure Document Intelligence.
    Returns the raw text extracted.
    """
    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    async with document_intelligence_client:
        poller = await document_intelligence_client.begin_analyze_document(
            "prebuilt-read", AnalyzeDocumentRequest(bytes_source=document_content)
        )
        result = await poller.result()

    raw_text = ""
    if result.content:
        raw_text = result.content

    return raw_text

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document for processing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    document_id = str(uuid.uuid4())
    document_type = file.filename.split('.')[-1].lower() # Simple type based on extension

    try:
        # Read file content
        document_content = await file.read()

        # Perform OCR/AI processing
        raw_data = await analyze_document_intelligence(document_content)

        # Store processed data (in-memory for now)
        processed_documents[document_id] = {
            "document_id": document_id,
            "document_type": document_type,
            "raw_data": raw_data,
            "status": "completed",
            "filename": file.filename,
            # Add content_info field expected by frontend
            "content_info": [],
            # Add other extracted data here later
        }

        return JSONResponse(content={"document_id": document_id, "filename": file.filename, "message": "Document uploaded and processing initiated", "status": "completed"})

    except Exception as e:
        # Log the exception in a real application
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")

@app.get("/processed/{document_type}/{document_id}")
async def get_processed_data(document_type: str, document_id: str):
    """
    Endpoint to retrieve extracted data for a processed document.
    """
    if document_id not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")

    document_data = processed_documents[document_id]

    # Optional: Add a check for document_type if needed, though document_id should be unique
    if document_data["document_type"] != document_type:
         # This might indicate a mismatch or incorrect URL, depending on how you want to handle it
         # For simplicity, we'll return the data if document_id matches
         pass


    return JSONResponse(content=document_data)

# Example of how to run the app (for local development)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)