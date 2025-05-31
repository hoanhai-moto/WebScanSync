// Test file to verify the API is working correctly
// Run with Node.js: node test_api.js

const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

async function testUploadEndpoint() {
  try {
    console.log('Testing /upload endpoint...');
    
    // Create a form with a test file
    const formData = new FormData();
    const testFilePath = path.join(__dirname, 'test.pdf'); // Replace with an actual test file path
    
    // Check if file exists
    if (!fs.existsSync(testFilePath)) {
      console.error(`Test file not found: ${testFilePath}`);
      console.log('Please create a test.pdf file in the project root directory');
      return;
    }
    
    formData.append('file', fs.createReadStream(testFilePath));
    
    // Make the API call
    const response = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders(),
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Upload response:', data);
    
    // Test the processed endpoint
    if (data.document_id) {
      await testProcessedEndpoint(data.document_id, data.filename.split('.')[0] || 'unknown');
    }
  } catch (error) {
    console.error('Error testing upload endpoint:', error);
  }
}

async function testProcessedEndpoint(documentId, documentType) {
  try {
    console.log(`Testing /processed/${documentType}/${documentId} endpoint...`);
    
    // Poll for processed data
    const maxAttempts = 5;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      console.log(`Attempt ${attempts + 1}/${maxAttempts}...`);
      
      const response = await fetch(`http://127.0.0.1:8000/processed/${documentType}/${documentId}`, {
        headers: {
          'Accept': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Processed data:', data);
        
        if (data.status === 'completed') {
          console.log('Document processing completed successfully!');
          return;
        }
      } else {
        console.log(`Failed to get processed data: ${response.statusText}`);
      }
      
      console.log('Waiting before next attempt...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      attempts++;
    }
    
    console.log('Document processing timed out or failed');
  } catch (error) {
    console.error('Error testing processed endpoint:', error);
  }
}

// Run the test
testUploadEndpoint(); 