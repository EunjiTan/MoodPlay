import uvicorn
import os
import sys

def main():
    print("Starting MoodPlay Local Development Server...")
    print("API Gateway: http://localhost:8000")
    
    # Ensure project root is in path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run FastAPI (Reload disabled for stability with heavy ML models)
    uvicorn.run("backend.api_gateway.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
