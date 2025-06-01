#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting Azure API deployment on 14.225.210.30..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Installing gunicorn..."
    pip install gunicorn
fi

# Check if the port is already in use
PORT=8000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Stopping the process..."
    kill $(lsof -t -i:$PORT) || true
    sleep 2
fi

# Start the API server in production mode with gunicorn
echo "Starting API server on port $PORT..."
nohup gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 14.225.210.30:$PORT > api_server.log 2>&1 &

# Check if server started successfully
sleep 5
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "API server is running on http://14.225.210.30:$PORT"
    echo "To test the API, run: python test_azure_api_endpoints.py"
else
    echo "Failed to start the API server. Check api_server.log for details."
    exit 1
fi 