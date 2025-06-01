import requests
import json
import sys

BASE_URL = "http://14.225.210.30:8000"  # Updated server address

def test_azure_chat():
    """Test the Azure OpenAI chat completion endpoint"""
    print("Testing Azure chat completion endpoint...")
    
    url = f"{BASE_URL}/api/azure/chat"
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What are the top 3 benefits of using Azure OpenAI services?"
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        print("Response from Azure OpenAI:")
        print(result["response"])
        print(f"Usage: {result['usage']}")
        print("Test successful!\n")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return False

def test_summarize_document(document_id):
    """Test the document summarization endpoint"""
    print(f"Testing document summarization for document ID: {document_id}...")
    
    url = f"{BASE_URL}/api/azure/summarize"
    payload = {
        "document_id": document_id
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("Summary:")
        print(result["summary"])
        print("Test successful!\n")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    # If a document ID is provided as a command line argument, test summarization
    if len(sys.argv) > 1:
        document_id = sys.argv[1]
        test_summarize_document(document_id)
    else:
        # Otherwise just test the chat endpoint
        test_azure_chat() 