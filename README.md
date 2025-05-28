# WebScanSync

WebScanSync is a document processing application that leverages Azure Document Intelligence and Azure OpenAI to extract, analyze, and structure information from uploaded documents.

## Features

- Upload documents (PDF, images) for processing
- Extract text from documents using Azure Document Intelligence (formerly Form Recognizer)
- Detect handwritten vs printed text
- Generate structured data from document content using Azure OpenAI
- Extract key information such as:
  - Document summary
  - Client information
  - Contract details
  - Advisory/recommendations
  - Document type
  - Key content points with positions and labels

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your Azure credentials
6. Run the application: `uvicorn main:app --reload`

## API Endpoints

- `POST /upload`: Upload a document for processing
- `GET /processed/{document_type}/{document_id}`: Retrieve processed data for a document
- `GET /`: Health check endpoint

## Output Format

The JSON output follows this structure:

```json
{
  "document_id": "uuid-string",
  "status": "completed",
  "summary": "Brief summary of document content",
  "client_info": "Information about the client/subject",
  "contract_info": "Details about any contracts mentioned",
  "advisory": "Recommendations or warnings based on content",
  "document_type": "Type of document detected",
  "uploaded_at": "2025-05-28T14:00:00Z",
  "content_info": [
    {
      "position": "page 1",
      "content": "Example text content",
      "label": "Name",
      "is_handwritten": true
    },
    ...
  ]
}
```

## Environment Variables

- `AZURE_FORM_RECOGNIZER_ENDPOINT`: Azure Document Intelligence endpoint
- `AZURE_FORM_RECOGNIZER_KEY`: Azure Document Intelligence API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name (e.g., "gpt-4")
