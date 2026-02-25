from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import logging

logger = logging.getLogger(__name__)

# Import Service (Local Monolith style)
# Ensure backend is in path if running from root
try:
    from backend.services.sam3_service import sam3_service
except ImportError:
    sam3_service = None
    logger.warning("sam3_service not available; /segment/init and / endpoints will be limited")

try:
    from backend.services.video_pipeline import video_pipeline
except ImportError:
    video_pipeline = None
    logger.warning("video_pipeline not available; WebSocket /ws/process endpoint will be limited")

try:
    from backend.services.ffmpeg_colorization import ffmpeg_colorization
except ImportError:
    ffmpeg_colorization = None
    logger.warning("ffmpeg_colorization not available; WebSocket /ws/colorize endpoint will be limited")

import uuid

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
    device = sam3_service.device if sam3_service else "unavailable"
    return {"status": "MoodPlay Local API Running", "model_device": device}

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
    if sam3_service is None:
        raise HTTPException(status_code=503, detail="Segmentation service not available")
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

@app.post("/process/{session_id}")
async def start_processing(session_id: str, video_path: str):
    """Start video processing pipeline."""
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return {"status": "ready", "session_id": session_id}

@app.websocket("/ws/process/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        # Wait for "start" message with video_path
        data = await websocket.receive_json()
        if data.get("command") == "start":
            video_path = data.get("video_path")
            if video_pipeline is None:
                await websocket.send_json({"error": "video_pipeline service not available"})
                return
            # Start pipeline
            await video_pipeline.process_video(video_path, session_id, websocket)

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WS Error: {e}")

# Colorization Endpoints
@app.post("/colorize/{session_id}")
async def start_colorization(session_id: str, video_path: str, prompt: str = "vibrant colors, natural lighting"):
    """
    Start colorization process for uploaded video.
    """
    return {"session_id": session_id, "status": "started", "message": "Connect to WebSocket for progress"}

@app.websocket("/ws/colorize/{session_id}")
async def colorize_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time colorization progress.
    Uses FFmpeg-based pipeline for VRAM optimization.
    """
    await websocket.accept()
    try:
        # Receive start command with parameters
        data = await websocket.receive_json()
        if data.get("command") == "start":
            video_path = data.get("video_path")
            prompt = data.get("prompt", "vibrant colors, natural lighting")
            if ffmpeg_colorization is None:
                await websocket.send_json({"error": "ffmpeg_colorization service not available"})
                return
            # Start FFmpeg-based colorization
            await ffmpeg_colorization.colorize_video(
                video_path=video_path,
                session_id=session_id,
                prompt=prompt,
                websocket=websocket
            )
    except WebSocketDisconnect:
        print(f"Colorization client disconnected: {session_id}")
    except Exception as e:
        print(f"Colorization WS Error: {e}")
        import traceback
        traceback.print_exc()


# --------------------------------------------------------------------------------
# NEW STAGED PIPELINE ENDPOINT (SD1.5 + ControlNext)
# --------------------------------------------------------------------------------
try:
    from backend.pipelines.orchestrator_v2 import staged_pipeline
except ImportError:
    staged_pipeline = None
    logger.warning("staged_pipeline not available; /colorize/staged endpoint will return 503")
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for long-running pipeline tasks
pipeline_executor = ThreadPoolExecutor(max_workers=1)

class StagedColorizeRequest(BaseModel):
    video_path: str
    style: str = "cinematic"
    interval: int = 5
    job_id: str = "web_job"

@app.post("/colorize/staged")
async def start_staged_colorization(req: StagedColorizeRequest):
    """
    Triggers the new VRAM-safe staged pipeline.
    Runs in background thread to avoid blocking API.
    """
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Video path not found")
    if staged_pipeline is None:
        raise HTTPException(status_code=503, detail="Staged pipeline service not available")

    loop = asyncio.get_event_loop()
    
    try:
        await loop.run_in_executor(
            pipeline_executor,
            lambda: staged_pipeline.run(
                input_video=req.video_path,
                output_name="web_colorized.mp4",
                style_name=req.style,
                keyframe_interval=req.interval,
                job_id=req.job_id,
                clean_start=True
            )
        )
        return {
            "status": "completed", 
            "output_path": "results/web_colorized.mp4",
            "download_url": "/uploads/../results/web_colorized.mp4" # Path hack or need mount
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# We need to expose 'results' dir too
app.mount("/results", StaticFiles(directory="results"), name="results")
