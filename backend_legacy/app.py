from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
from backend import config
from celery.result import AsyncResult
from backend.workers.celery_config import celery_app
from backend.workers.tasks_seg import segment_video_task, refine_segmentation_task
from backend.workers.tasks_gen import generate_video_task, colorize_frame_task, batch_colorize_task

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    allowed_exts = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'} 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

@app.route('/')
def index():
    return render_template_string(open('frontend/index.html', encoding='utf-8').read())

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        base_name = f"{timestamp}_{filename}"
        upload_path = os.path.join(config.UPLOAD_FOLDER, base_name)
        file.save(upload_path)
        
        # Return video ID/path so frontend can load it and user can click
        return jsonify({'success': True, 'video_path': upload_path, 'video_id': base_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/interact/segment', methods=['POST'])
def segment_video():
    """
    Start segmentation task with multi-modal prompting.
    
    Request JSON:
        video_path: Path to video
        prompts: List of prompt objects with:
            - object_id: Unique ID
            - type: "click" | "box" | "text" | "negative"
            - frame_idx: Frame index (default 0)
            - data: Prompt-specific data
        output_dir: Optional output directory for masks
    """
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video path'}), 400
    
    # Convert legacy click format to new prompt format
    prompts = data.get('prompts', [])
    
    # Support legacy 'clicks' format for backward compatibility
    legacy_clicks = data.get('clicks', [])
    if legacy_clicks and not prompts:
        prompts = []
        for click in legacy_clicks:
            prompts.append({
                'object_id': click.get('object_id', 1),
                'type': 'click',
                'frame_idx': click.get('frame', 0),
                'data': {
                    'points': [[click.get('x', 0), click.get('y', 0)]],
                    'labels': [1]  # Positive click
                }
            })
    
    output_dir = data.get('output_dir')
    
    task = segment_video_task.delay(video_path, prompts, output_dir)
    return jsonify({'success': True, 'task_id': task.id})

@app.route('/interact/segment/refine', methods=['POST'])
def refine_segment():
    """
    Refine an existing segmentation with additional prompts.
    """
    data = request.json
    video_path = data.get('video_path')
    object_id = data.get('object_id', 1)
    frame_idx = data.get('frame_idx', 0)
    refinement_prompts = data.get('prompts', [])
    
    task = refine_segmentation_task.delay(
        video_path, object_id, frame_idx, refinement_prompts
    )
    return jsonify({'success': True, 'task_id': task.id})

@app.route('/generate', methods=['POST'])
def generate_video():
    """
    Start video colorization/generation task.
    
    Request JSON:
        video_path: Path to input video
        mask_path: Path to mask directory (from segmentation)
        prompt: Colorization prompt
        negative_prompt: Optional negative prompt
        style_lora: Optional path to style LoRA
        colorization_lora: Optional path to colorization LoRA
        guidance_scale: Optional CFG scale
        controlnet_weight: Optional ControlNet strength
    """
    data = request.json
    video_path = data.get('video_path')
    mask_path = data.get('mask_path')
    prompt = data.get('prompt', 'vibrant colors, high quality')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video path'}), 400
    
    # Optional parameters
    kwargs = {}
    if 'negative_prompt' in data:
        kwargs['negative_prompt'] = data['negative_prompt']
    if 'style_lora' in data:
        kwargs['style_lora'] = data['style_lora']
    if 'colorization_lora' in data:
        kwargs['colorization_lora'] = data['colorization_lora']
    if 'guidance_scale' in data:
        kwargs['guidance_scale'] = data['guidance_scale']
    if 'controlnet_weight' in data:
        kwargs['controlnet_weight'] = data['controlnet_weight']
    
    task = generate_video_task.delay(video_path, mask_path, prompt, **kwargs)
    return jsonify({'success': True, 'task_id': task.id})

@app.route('/generate/frame', methods=['POST'])
def generate_frame():
    """
    Colorize a single frame/image.
    """
    data = request.json
    image_path = data.get('image_path')
    prompt = data.get('prompt', 'vibrant colors')
    mask = data.get('mask')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400
    
    task = colorize_frame_task.delay(image_path, prompt, mask)
    return jsonify({'success': True, 'task_id': task.id})

@app.route('/generate/batch', methods=['POST'])
def generate_batch():
    """
    Colorize a batch of frames with temporal consistency.
    """
    data = request.json
    frame_paths = data.get('frame_paths', [])
    prompt = data.get('prompt', 'vibrant colors')
    masks_dir = data.get('masks_dir')
    
    if not frame_paths:
        return jsonify({'error': 'No frames provided'}), 400
    
    task = batch_colorize_task.delay(frame_paths, prompt, masks_dir)
    return jsonify({'success': True, 'task_id': task.id})

@app.route('/config/colorization', methods=['GET', 'POST'])
def colorization_config():
    """
    Get or update colorization configuration.
    """
    from backend.workers import colorization_config as cfg
    
    if request.method == 'GET':
        return jsonify({
            'batch_size': cfg.BATCH_SIZE,
            'overlap_frames': cfg.OVERLAP_FRAMES,
            'controlnet_weight': cfg.CONTROLNET_WEIGHT,
            'guidance_scale': cfg.GUIDANCE_SCALE,
            'lora_colorization_weight': cfg.LORA_COLORIZATION_WEIGHT,
            'lora_style_weight': cfg.LORA_STYLE_WEIGHT,
            'warp_strength': cfg.WARP_STRENGTH,
            'clip_threshold': cfg.CLIP_SIMILARITY_THRESHOLD,
            'sam2_model_size': cfg.SAM2_MODEL_SIZE,
            'sam2_confidence': cfg.SAM2_CONFIDENCE_THRESHOLD,
        })
    
    # POST - update configuration (runtime only, not persistent)
    data = request.json
    if 'batch_size' in data:
        cfg.BATCH_SIZE = int(data['batch_size'])
    if 'overlap_frames' in data:
        cfg.OVERLAP_FRAMES = int(data['overlap_frames'])
    if 'controlnet_weight' in data:
        cfg.CONTROLNET_WEIGHT = float(data['controlnet_weight'])
    if 'guidance_scale' in data:
        cfg.GUIDANCE_SCALE = float(data['guidance_scale'])
    if 'lora_colorization_weight' in data:
        cfg.LORA_COLORIZATION_WEIGHT = float(data['lora_colorization_weight'])
    if 'lora_style_weight' in data:
        cfg.LORA_STYLE_WEIGHT = float(data['lora_style_weight'])
    if 'warp_strength' in data:
        cfg.WARP_STRENGTH = float(data['warp_strength'])
    if 'clip_threshold' in data:
        cfg.CLIP_SIMILARITY_THRESHOLD = float(data['clip_threshold'])
    
    return jsonify({'success': True, 'message': 'Configuration updated'})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task_result = AsyncResult(task_id, app=celery_app)
    result_val = task_result.result
    
    # Handle progress updates
    meta = {}
    if task_result.state == 'PROGRESS':
        meta = result_val if isinstance(result_val, dict) else {}
        result_val = None
    elif isinstance(result_val, Exception):
        result_val = str(result_val)
    
    return jsonify({
        "task_id": task_id,
        "status": task_result.status,
        "result": result_val,
        "meta": meta,
    })

@app.route('/result/<filename>')
def get_result_file(filename):
    return send_from_directory(config.RESULTS_FOLDER, filename)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(config.UPLOAD_FOLDER, filename)

@app.route('/masks/<path:filepath>')
def get_mask_file(filepath):
    """Serve mask files."""
    mask_base = os.path.join(config.UPLOAD_FOLDER, 'masks')
    return send_from_directory(mask_base, filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
