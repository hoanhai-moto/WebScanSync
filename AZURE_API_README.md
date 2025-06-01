# Azure API Integration

This document provides instructions for running and using the Azure OpenAI API endpoints integrated with the WebScanSync application.

## Available Azure API Endpoints

The following Azure OpenAI API endpoints have been implemented:

1. **Chat Completion API**: `/api/azure/chat`
   - Allows interaction with Azure OpenAI's chat completions API
   - Uses model-router deployment for flexible completions

2. **Document Summarization API**: `/api/azure/summarize`
   - Summarizes document content using Azure OpenAI
   - Works with documents processed through the application

## Running the API

### Prerequisites

- Python 3.8+
- Required Python packages (install with `pip install -r requirements.txt`)

### Steps to Run

1. Make sure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI application:
   ```
   python app.py
   ```

3. The API will be available at `http://14.225.210.30:8000`

### Production Deployment

For production deployment, we recommend using a proper WSGI server like Uvicorn with Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

## Using the API Endpoints

### Chat Completion Endpoint

**Endpoint:** `/api/azure/chat`

**Method:** POST

**Request Body:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What are the benefits of using Azure OpenAI services?"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Azure OpenAI services offer several benefits...",
  "usage": {
    "completion_tokens": 150,
    "prompt_tokens": 25,
    "total_tokens": 175
  }
}
```

### Document Summarization Endpoint

**Endpoint:** `/api/azure/summarize`

**Method:** POST

**Request Body:**
```json
{
  "document_id": "98069ef1-5f37-49d8-85e5-0f68ad66fa42"
}
```

**Response:**
```json
{
  "summary": "This document discusses..."
}
```

## Testing the API

A test script is provided to verify the API endpoints are working correctly.

```bash
# Test the chat endpoint
python test_azure_api_endpoints.py

# Test the summarization endpoint (provide a document ID)
python test_azure_api_endpoints.py 98069ef1-5f37-49d8-85e5-0f68ad66fa42
```

## Error Handling

The API includes proper error handling:

- 400 Bad Request: Returned when the request is malformed or missing required fields
- 404 Not Found: Returned when a document ID is not found
- 500 Internal Server Error: Returned when there's an error with the Azure OpenAI API

## Security Considerations

In a production environment, consider the following security best practices:

1. Store API keys in environment variables, not in the code
2. Implement proper authentication for API endpoints
3. Restrict CORS to only allow trusted domains
4. Use HTTPS for all API communication 