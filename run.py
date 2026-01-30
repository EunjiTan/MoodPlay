"""
MoodPlay Application Runner
Starts the Flask API server with optional demo mode.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for demo mode (no heavy ML dependencies)
DEMO_MODE = os.environ.get('DEMO_MODE', '0') == '1'

import uvicorn
from backend.api_gateway.app import app

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MoodPlay Video Object Segmentation System")
    print("=" * 50)
    print(f"\nServer starting at: http://localhost:8000")
    print("\nPress Ctrl+C to stop\n")
    
    # Run with Uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
