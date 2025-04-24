#!/bin/bash
cd "/Users/Martins/Desktop/test/Contents/Resources/xautron_frontend/design"
echo "Starting Xautron frontend..."
echo "Working directory: $(pwd)"
echo "Installing dependencies..."
rm -rf node_modules
npm install
echo "Installing react-scripts..."
npm install react-scripts@5.0.1 --save-dev
echo "Starting frontend server..."
export NODE_OPTIONS=--openssl-legacy-provider
npm start
echo "Frontend server stopped."
echo "Press Enter to close this terminal..."
read
