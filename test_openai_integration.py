import pytest
import os
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app, process_document

client = TestClient(app)

@pytest.mark.asyncio
async def test_extract_structured_data():
    """Test the extract_structured_data function with mock Azure OpenAI response"""
    
    # Sample raw text
    raw_text = "Contract Agreement\nBetween Nguyễn Văn A (ID: 123456789)\nAnd Company XYZ\nProperty size: 180m2"
    document_type = "contract"
    
    # Mock response from Azure OpenAI
    mock_content = json.dumps({
        "summary": "Contract agreement between Nguyen Van A and Company XYZ for property",
        "client_info": "Nguyễn Văn A, ID: 123456789",
        "contract_info": "Contract between Nguyễn Văn A and Company XYZ",
        "advisory": "Verify property details and signatures",
        "document_type": "contract",
        "content_info": [
            {
                "position": "page 1",
                "content": "Nguyễn Văn A, ID: 123456789",
                "label": "Client name and ID"
            },
            {
                "position": "page 1",
                "content": "Property size: 180m2",
                "label": "Property size"
            }
        ]
    })
    
    mock_choice = MagicMock()
    mock_choice.message.content = mock_content
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Create a mock for the Azure OpenAI client
    with patch('main.AzureOpenAI') as mock_azure_openai:
        # Configure the mock to return our mock response
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client
        
        # Call the extract_structured_data function
        from main import extract_structured_data
        result = await extract_structured_data(raw_text, document_type)
        
        # Assertions
        assert isinstance(result, dict)
        assert "summary" in result
        assert "client_info" in result
        assert "contract_info" in result
        assert "advisory" in result
        assert "document_type" in result
        assert "content_info" in result
        assert isinstance(result["content_info"], list)
        assert len(result["content_info"]) == 2

@pytest.mark.asyncio
async def test_process_document():
    """Test the process_document function with mocked Azure services"""
    
    # Create temporary test files
    test_document_id = "test-document-id"
    test_file_path = f"test_document_{test_document_id}.pdf"
    
    # Write an empty file for testing
    with open(test_file_path, "wb") as f:
        f.write(b"Test PDF content")
    
    try:
        # Mock the Document Intelligence client
        with patch('main.DocumentIntelligenceClient') as mock_di_client, \
             patch('main.extract_structured_data') as mock_extract:
            
            # Create mock analyze result
            mock_result = MagicMock()
            mock_result.content = "Test document content"
            mock_result.pages = []
            mock_result.styles = []
            
            # Configure poller mock
            mock_poller = MagicMock()
            mock_poller.result.return_value = mock_result
            
            # Configure client mock
            mock_client_instance = MagicMock()
            mock_client_instance.begin_analyze_document.return_value = mock_poller
            mock_di_client.return_value = mock_client_instance
            
            # Mock extract_structured_data function
            mock_structured_data = {
                "summary": "Test summary",
                "client_info": "Test client",
                "contract_info": "Test contract",
                "advisory": "Test advisory",
                "document_type": "Test type",
                "uploaded_at": "2025-05-28T00:00:00Z",
                "content_info": []
            }
            mock_extract.return_value = mock_structured_data
            
            # Call the process_document function
            result = await process_document(test_file_path, test_document_id)
            
            # Assertions
            assert isinstance(result, dict)
            assert result["document_id"] == test_document_id
            assert result["status"] == "completed"
            assert "summary" in result
            assert "client_info" in result
            assert "contract_info" in result
            assert "advisory" in result
            assert "document_type" in result
            assert "uploaded_at" in result
            assert "content_info" in result
    
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
