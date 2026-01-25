from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from celery import Celery
import os
import shutil
import asyncio
import json
from typing import List

# Configuration
UPLOAD_DIR = "/shared/uploads"
RESULTS_DIR = "/shared/results"
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Ensure shared dirs exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# App Setup
app = FastAPI(title="MoodPlay v3.0 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Celery Client
celery_app = Celery("moodplay", broker=REDIS_URL, backend=REDIS_URL)

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.get("/")
def read_root():
    return {"status": "MoodPlay v3.0 Gateway Online"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    return {"filename": file.filename, "path": file_location, "status": "uploaded"}

@app.post("/segment/click")
async def segment_click(video_path: str, points: list):
    """
    Trigger SAM-3 Segmentation Task.
    """
    # Send to Celery Queue (sam3_queue)
    task = celery_app.send_task(
        "services.sam3.worker.segment_click_task",
        args=[video_path, points],
        queue="sam3_queue"
    )
    return {"task_id": task.id, "status": "queued"}

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # In a real app, client might ping/pong or ask for specific task updates
            # For now, we just echo or broadcast mock updates
            # Real updates come from Celery backend polling (implemented in background task usually)
            pass 
    except WebSocketDisconnect:
        manager.disconnect(websocket)
