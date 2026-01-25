import os
from celery import Celery

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

def make_celery(name):
    return Celery(name, broker=REDIS_URL, backend=REDIS_URL)

celery_app = make_celery('moodplay')

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Prevents memory issues with GPU tasks
)

# Route tasks to specific queues
celery_app.conf.task_routes = {
    # Segmentation tasks
    'backend.workers.tasks_seg.segment_video_task': {'queue': 'sam3_queue'},
    'backend.workers.tasks_seg.refine_segmentation_task': {'queue': 'sam3_queue'},
    
    # Generation tasks
    'backend.workers.tasks_gen.generate_video_task': {'queue': 'generation_queue'},
    'backend.workers.tasks_gen.colorize_frame_task': {'queue': 'generation_queue'},
    'backend.workers.tasks_gen.batch_colorize_task': {'queue': 'generation_queue'},
}

# Task result expiration (24 hours)
celery_app.conf.result_expires = 86400
