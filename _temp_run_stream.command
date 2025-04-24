#!/bin/bash
cd "/Users/Martins/Desktop/test/Contents/Resources/xautron_backend"
echo "Starting Xautron stream processor..."
echo "Working directory: $(pwd)"
echo "Using Python path: $(which python3)"
echo "Python version: $(python3 --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Contents of current directory:"
ls -la
echo "Contents of xautron_backend directory:"
ls -la xautron_backend
echo "Contents of tpqoa directory:"
ls -la tpqoa
python3 "start_stream.py"
echo "Process completed or terminated."
echo "Press Enter to close this terminal..."
read
