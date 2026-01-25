import os
import time
from celery import Celery
from backend.processing.tracking import SAM2Tracker
from backend.workers.pipeline import RenderPipeline

# Configure Celery
# Assumes Redis is running on localhost:6379
celery_app = Celery('moodplay', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Global instances to avoid reloading on every task (Lazy loading handled in classes)
tracker = SAM2Tracker("checkpoints/sam2.pt", "configs/sam2.yaml")
pipeline = RenderPipeline()

@celery_app.task(bind=True)
def process_video_task(self, video_path, output_path, prompt, style_lora=None):
    """
    Background task to process video.
    """
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Initializing...'})
    
    print(f"Processing video: {video_path}")
    
    # 1. Initialize Tracker
    tracker.init_video(video_path)
    
    # 2. Iterate frames (Mock loop for now since we don't have video reader util here yet)
    # real impl would use cv2.VideoCapture(video_path)
    
    total_frames = 100 # Mock
    for i in range(total_frames):
        # 3. Get Mask (Mock)
        # _, _, masks = next(tracker.propagate()) 
        
        # 4. Render Frame (Mock)
        # pipeline.process_frame(..., prompt=prompt)
        
        # Update progress
        if i % 10 == 0:
            self.update_state(state='PROGRESS', 
                              meta={'current': i, 'total': total_frames, 'status': f'Rendering frame {i}'})
            time.sleep(0.1) # Simulate work

    # 5. Save Output
    # For demo mode: Copy original video to output so it's a valid MP4 file
    import shutil
    print(f"Saving output to {output_path}")
    shutil.copy2(video_path, output_path)
        
    return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': output_path}
