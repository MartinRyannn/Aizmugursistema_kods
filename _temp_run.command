#!/bin/bash

# Get the directory this script is in
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Check if start_stream.py exists
if [ ! -f "$SCRIPT_DIR/start_stream.py" ]; then
    echo "ERROR: start_stream.py not found in $SCRIPT_DIR"
    echo "Press Enter to exit..."
    read
    exit 1
fi

# Run the stream processor script
echo "Starting Xautron stream processor..."
echo "Working directory: $(pwd)"
python3 "$SCRIPT_DIR/start_stream.py"

# Keep terminal open when done
echo "Process completed or terminated."
echo "Press Enter to close this terminal..."
read