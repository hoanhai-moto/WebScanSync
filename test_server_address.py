import requests
import sys

def test_server_connection():
    """Test if the server is running at http://14.225.210.30:8000"""
    url = "http://14.225.210.30:8000"
    
    try:
        response = requests.get(url)
        print(f"Server is running at {url}")
        print(f"Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to server at {url}")
        print("Make sure the server is running and bound to 14.225.210.30")
        return False

if __name__ == "__main__":
    test_server_connection() 