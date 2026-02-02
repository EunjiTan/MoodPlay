import React, { useState, useRef, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

const STEPS = {
    UPLOAD: 0,
    SEGMENTATION: 1,
    PROMPT: 2,
    PROCESSING: 3,
    RESULT: 4
};

export default function UnifiedPipeline() {
    const [currentStep, setCurrentStep] = useState(STEPS.UPLOAD);
    const [videoFile, setVideoFile] = useState(null);
    const [uploadedPath, setUploadedPath] = useState('');
    const [sessionId, setSessionId] = useState('');

    // Segmentation State
    const [detections, setDetections] = useState([]);
    const [isSegmenting, setIsSegmenting] = useState(false);
    const [segmentationProgress, setSegmentationProgress] = useState(0);

    // Colorization State
    const [prompt, setPrompt] = useState('vibrant colors, natural lighting, high quality');
    const [isColorizing, setIsColorizing] = useState(false);
    const [colorizationProgress, setColorizationProgress] = useState(0);
    const [outputPath, setOutputPath] = useState('');

    const canvasRef = useRef(null);
    const wsRef = useRef(null);

    useEffect(() => {
        // Generate session ID on mount
        setSessionId(Date.now().toString());
    }, []);

    // STEP 1: Upload
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setVideoFile(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            setUploadedPath(data.path);
            setCurrentStep(STEPS.SEGMENTATION); // Auto-advance
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed. Is the backend running?');
        }
    };

    // STEP 2: Segmentation (YOLO + SAM)
    const startSegmentation = () => {
        setIsSegmenting(true);

        wsRef.current = new WebSocket(`${WS_BASE}/ws/process/${sessionId}`);

        wsRef.current.onopen = () => {
            wsRef.current.send(JSON.stringify({
                command: 'start',
                video_path: uploadedPath
            }));
        };

        wsRef.current.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                if (data.status === 'completed') {
                    setIsSegmenting(false);
                    // Don't auto-advance yet, let user review detections
                }
                if (data.progress) setSegmentationProgress(data.progress);
                if (data.detections) setDetections(data.detections);
            } else {
                // Draw frame
                const bitmap = await createImageBitmap(event.data);
                const canvas = canvasRef.current;
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    ctx.drawImage(bitmap, 0, 0);
                }
            }
        };
    };

    const confirmSegmentation = () => {
        if (wsRef.current) wsRef.current.close();
        setCurrentStep(STEPS.PROMPT);
    };

    // STEP 3: Colorization
    const startColorization = () => {
        setCurrentStep(STEPS.PROCESSING);
        setIsColorizing(true);

        wsRef.current = new WebSocket(`${WS_BASE}/ws/colorize/${sessionId}`);

        wsRef.current.onopen = () => {
            wsRef.current.send(JSON.stringify({
                command: 'start',
                video_path: uploadedPath,
                prompt: prompt,
                num_steps: 15
            }));
        };

        wsRef.current.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                if (data.status === 'completed') {
                    setOutputPath(data.output_path);
                    setIsColorizing(false);
                    setCurrentStep(STEPS.RESULT);
                }
                if (data.progress) setColorizationProgress(data.progress);
            } else {
                // Draw preview
                const bitmap = await createImageBitmap(event.data);
                const canvas = canvasRef.current;
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    ctx.drawImage(bitmap, 0, 0);
                }
            }
        };
    };

    return (
        <div className="unified-container">
            {/* Progress Stepper */}
            <div className="stepper">
                {['Upload', 'Segmentation', 'Description', 'Processing', 'Result'].map((label, idx) => (
                    <div key={idx} className={`step ${currentStep >= idx ? 'active' : ''}`}>
                        <div className="step-circle">{idx + 1}</div>
                        <span className="step-label">{label}</span>
                    </div>
                ))}
            </div>

            <div className="content-area">
                {currentStep === STEPS.UPLOAD && (
                    <div className="step-card">
                        <h2>📤 Upload Video</h2>
                        <div className="upload-box">
                            <input type="file" accept="video/*" onChange={handleFileUpload} />
                            <p>Drag & drop or click to upload</p>
                        </div>
                    </div>
                )}

                {currentStep === STEPS.SEGMENTATION && (
                    <div className="step-card">
                        <h2>🎯 Object Segmentation</h2>
                        <p>We'll detect objects to ensure consistent coloring.</p>

                        <div className="preview-container">
                            <canvas ref={canvasRef} className="video-canvas" />
                        </div>

                        <div className="action-row">
                            {!isSegmenting && segmentationProgress === 0 && (
                                <button onClick={startSegmentation} className="btn-primary">
                                    Start Analysis
                                </button>
                            )}

                            {isSegmenting && (
                                <div className="progress-bar">
                                    <div className="fill" style={{ width: `${segmentationProgress}%` }}></div>
                                    <span>Analyzing... {segmentationProgress.toFixed(0)}%</span>
                                </div>
                            )}

                            {segmentationProgress === 100 && (
                                <button onClick={confirmSegmentation} className="btn-success">
                                    Confirm & Next ➜
                                </button>
                            )}
                        </div>
                    </div>
                )}

                {currentStep === STEPS.PROMPT && (
                    <div className="step-card">
                        <h2>🎨 Color Description</h2>
                        <div className="form-group">
                            <label>Describe the scene colors:</label>
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                rows={4}
                            />
                        </div>
                        <button onClick={startColorization} className="btn-primary">
                            Generate Colors 🚀
                        </button>
                    </div>
                )}

                {currentStep === STEPS.PROCESSING && (
                    <div className="step-card">
                        <h2>⚡ Colorizing Video...</h2>
                        <div className="preview-container">
                            <canvas ref={canvasRef} className="video-canvas" />
                        </div>
                        <div className="progress-bar">
                            <div className="fill" style={{ width: `${colorizationProgress}%` }}></div>
                            <span>Generating... {colorizationProgress.toFixed(0)}%</span>
                        </div>
                    </div>
                )}

                {currentStep === STEPS.RESULT && (
                    <div className="step-card">
                        <h2>✨ Colorization Complete!</h2>
                        <video controls src={`${API_BASE}/${outputPath}`} className="result-video" />
                        <a href={`${API_BASE}/${outputPath}`} download className="btn-primary">
                            Download Video 📥
                        </a>
                    </div>
                )}
            </div>

            <style jsx>{`
        .unified-container {
          max-width: 1000px;
          margin: 0 auto;
          color: white;
        }
        
        .stepper {
          display: flex;
          justify-content: space-between;
          margin-bottom: 3rem;
          position: relative;
        }
        
        .stepper::before {
          content: '';
          position: absolute;
          top: 15px;
          left: 0;
          right: 0;
          height: 2px;
          background: #334155;
          z-index: 0;
        }
        
        .step {
          position: relative;
          z-index: 1;
          background: #0f172a;
          padding: 0 10px;
          text-align: center;
          opacity: 0.5;
          transition: 0.3s;
        }
        
        .step.active {
          opacity: 1;
        }
        
        .step-circle {
          width: 32px;
          height: 32px;
          background: #334155;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 8px;
          font-weight: bold;
        }
        
        .step.active .step-circle {
          background: #8b5cf6;
          box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
        }
        
        .step-card {
          background: #1e293b;
          border-radius: 16px;
          padding: 2rem;
          text-align: center;
        }
        
        .upload-box {
          border: 2px dashed #475569;
          padding: 4rem;
          border-radius: 12px;
          margin-top: 1rem;
          cursor: pointer;
        }
        
        .video-canvas, .result-video {
          max-width: 100%;
          border-radius: 8px;
          background: black;
          margin: 1rem 0;
        }
        
        .btn-primary {
          background: #8b5cf6;
          color: white;
          padding: 0.75rem 2rem;
          border-radius: 8px;
          font-weight: 600;
          border: none;
          cursor: pointer;
          font-size: 1.1rem;
        }
        
        textarea {
          width: 100%;
          background: #0f172a;
          border: 1px solid #334155;
          color: white;
          padding: 1rem;
          border-radius: 8px;
          margin: 1rem 0;
        }
        
        .progress-bar {
          height: 8px;
          background: #334155;
          border-radius: 4px;
          overflow: hidden;
          margin: 1rem 0;
          position: relative;
        }
        
        .fill {
          height: 100%;
          background: #8b5cf6;
          transition: width 0.3s;
        }
      `}</style>
        </div>
    );
}
