from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

# Import Service (Local Monolith style)
# Ensure backend is in path if running from root
from backend.services.sam3_service import sam3_service

from fastapi.staticfiles import StaticFiles

# ... imports ...

app = FastAPI(title="MoodPlay Local API", version="1.0.0")

# Mount Uploads for Static Access (Critical for Frontend Video Player)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ClickRequest(BaseModel):
    video_path: str
    frame_idx: int
    object_id: int
    points: list # [[x,y], [x,y]]
    labels: list # [1, 0]

@app.get("/")
def read_root():
    return {"status": "MoodPlay Local API Running", "model_device": sam3_service.device}

@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Auto-initialize session on upload for now? Or wait for explicit call?
    # Let's just return path.
    return {"status": "uploaded", "path": file_location}

@app.post("/segment/init")
def init_segmentation(video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    try:
        res = sam3_service.init_session(video_path)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Import ControlNet Service
from backend.services.controlnet_service import controlnet_service

class GenerateRequest(BaseModel):
    video_path: str
    prompt: str

@app.post("/generate")
def generate_video(req: GenerateRequest):
    # This should be async in release, but for local dev we can run one frame to test
    # or spawn a thread. For Step 2.1, let's verify loading.
    
    try:
        # Just trigger model verify/load
        controlnet_service.load_model()
        return {"status": "ControlNet Loaded", "message": "Ready to process"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

