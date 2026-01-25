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

if DEMO_MODE:
    print("=" * 50)
    print("Running in DEMO MODE (no ML processing)")
    print("=" * 50)

from backend.app import app

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MoodPlay Video Colorization Pipeline")
    print("=" * 50)
    print(f"\nServer starting at: http://localhost:5000")
    print(f"Demo Mode: {DEMO_MODE}")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
