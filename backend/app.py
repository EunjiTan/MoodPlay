from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
from backend import config
from backend import processing
from backend.model import model

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template_string(open('frontend/index.html', encoding='utf-8').read())


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        base_name = f"{timestamp}_{filename}"
        
        upload_path = os.path.join(config.UPLOAD_FOLDER, base_name)
        gray_path = os.path.join(config.RESULTS_FOLDER, f"gray_{base_name}")
        seg_path = os.path.join(config.RESULTS_FOLDER, f"seg_{base_name}")
        color_path = os.path.join(config.RESULTS_FOLDER, f"color_{base_name}")
        
        file.save(upload_path)
        
        print(f"Processing: {filename}")
        processing.convert_to_grayscale(upload_path, gray_path)
        processing.generate_segmentation(gray_path, seg_path)
        
        if model.loaded:
            model.colorize(gray_path, seg_path, color_path)
        else:
            processing.colorize_placeholder(gray_path, seg_path, color_path)
        
        return jsonify({
            'success': True,
            'original': 'data:image/png;base64,' + processing.image_to_base64(upload_path),
            'grayscale': 'data:image/png;base64,' + processing.image_to_base64(gray_path),
            'segmentation': 'data:image/png;base64,' + processing.image_to_base64(seg_path),
            'colorized': 'data:image/png;base64,' + processing.image_to_base64(color_path),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        'online': True,
        'model_loaded': model.loaded
    })
