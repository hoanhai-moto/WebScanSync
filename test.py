import requests
import os
import time

# Assuming your FastAPI app is running locally on port 8000
BASE_URL = "http://127.0.0.1:8000"

def upload_document(file_path: str):
    """
    Uploads a document to the FastAPI /upload endpoint.
    """
    url = f"{BASE_URL}/upload"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        response = requests.post(url, files=files)
    return response.json()

def get_processed_data(document_type: str, document_id: str):
    """
    Retrieves processed data from the FastAPI /processed endpoint.
    """
    url = f"{BASE_URL}/processed/{document_type}/{document_id}"
    response = requests.get(url)
    return response.json()

if __name__ == "__main__":
    # --- Test Workflow ---
    # Replace 'path/to/your/test/document.pdf' with the actual path to a test file
    # You can use the 'mau 01 - HD uy quyen.pdf' file if it exists in uploaded_documents
    # For this test, let's assume you have a test file named 'test_document.pdf'
    # in the same directory as test.py, or provide a full path.
    test_file_path = "./uploaded_documents/mau 01 - HD uy quyen.pdf" # Example using the file from environment_details

    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
    else:
        print(f"Uploading document: {test_file_path}")
        upload_response = upload_document(test_file_path)
        print("Upload Response:", upload_response)

        if "document_id" in upload_response:
            document_id = upload_response["document_id"]
            print(f"Document uploaded with ID: {document_id}")

            # In a real application, you would poll the status or use webhooks
            # For this simple test, we'll wait a few seconds for processing to complete
            print("Waiting for processing to complete...")
            time.sleep(10) # Adjust the sleep time based on expected processing duration

            print(f"Attempting to retrieve processed data for document ID: {document_id}")
            # Note: The document_type is currently not used by the /processed endpoint
            # in main.py, but we include it as per the API definition.
            processed_data_response = get_processed_data("general", document_id)
            print("Processed Data Response:", processed_data_response)
        else:
            print("Failed to get document_id from upload response.")