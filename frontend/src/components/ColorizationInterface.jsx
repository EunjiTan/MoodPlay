import React, { useState, useRef, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

export default function ColorizationInterface() {
    const [videoFile, setVideoFile] = useState(null);
    const [uploadedPath, setUploadedPath] = useState('');
    const [prompt, setPrompt] = useState('vibrant colors, natural lighting, high quality');
    const [numSteps, setNumSteps] = useState(15);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [stats, setStats] = useState(null);
    const [outputPath, setOutputPath] = useState('');

    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const sessionIdRef = useRef(Date.now().toString());

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setVideoFile(file);

        // Upload to backend
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            setUploadedPath(data.path);
            console.log('Video uploaded:', data.path);
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload video');
        }
    };

    const startColorization = () => {
        if (!uploadedPath) {
            alert('Please upload a video first');
            return;
        }

        setIsProcessing(true);
        setProgress(0);
        setOutputPath('');

        const sessionId = sessionIdRef.current;

        // Connect to WebSocket
        wsRef.current = new WebSocket(`${WS_BASE}/ws/colorize/${sessionId}`);

        wsRef.current.onopen = () => {
            console.log('WebSocket connected');
            // Send start command
            wsRef.current.send(JSON.stringify({
                command: 'start',
                video_path: uploadedPath,
                prompt: prompt,
                num_steps: numSteps,
            }));
        };

        wsRef.current.onmessage = async (event) => {
            if (typeof event.data === 'string') {
                // JSON stats
                const data = JSON.parse(event.data);

                if (data.error) {
                    alert(`Error: ${data.error}`);
                    setIsProcessing(false);
                    return;
                }

                if (data.status === 'completed') {
                    setOutputPath(data.output_path);
                    setIsProcessing(false);
                    setProgress(100);
                    console.log('Colorization complete!', data.output_path);
                    return;
                }

                setStats(data);
                if (data.progress) {
                    setProgress(data.progress);
                }
            } else {
                // Binary image data (Blob)
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

        wsRef.current.onclose = () => {
            console.log('WebSocket closed');
            setIsProcessing(false);
        };

        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setIsProcessing(false);
        };
    };

    const stopProcessing = () => {
        if (wsRef.current) {
            wsRef.current.close();
        }
        setIsProcessing(false);
    };

    return (
        <div className="colorization-container">
            <div className="header">
                <h1>🎨 Video Colorization</h1>
                <p>Transform grayscale videos into vibrant color using AI</p>
            </div>

            <div className="main-content">
                {/* Upload Section */}
                <div className="upload-section card">
                    <h2>1. Upload Grayscale Video</h2>
                    <input
                        type="file"
                        accept="video/*"
                        onChange={handleFileUpload}
                        className="file-input"
                        disabled={isProcessing}
                    />
                    {videoFile && (
                        <div className="file-info">
                            ✓ {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
                        </div>
                    )}
                </div>

                {/* Prompt Section */}
                <div className="prompt-section card">
                    <h2>2. Describe the Desired Colors</h2>
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g., vibrant colors, warm sunset tones, natural lighting..."
                        className="prompt-input"
                        rows={3}
                        disabled={isProcessing}
                    />

                    <div className="settings">
                        <label>
                            Quality Steps: {numSteps}
                            <input
                                type="range"
                                min="10"
                                max="30"
                                value={numSteps}
                                onChange={(e) => setNumSteps(parseInt(e.target.value))}
                                disabled={isProcessing}
                            />
                            <span className="hint">Higher = better quality but slower</span>
                        </label>
                    </div>
                </div>

                {/* Control Section */}
                <div className="control-section card">
                    <h2>3. Start Colorization</h2>
                    {!isProcessing ? (
                        <button
                            onClick={startColorization}
                            className="btn-primary"
                            disabled={!uploadedPath}
                        >
                            🚀 Start Colorization
                        </button>
                    ) : (
                        <button onClick={stopProcessing} className="btn-danger">
                            ⏹ Stop
                        </button>
                    )}
                </div>

                {/* Preview Section */}
                {isProcessing && (
                    <div className="preview-section card">
                        <h2>Real-time Preview</h2>
                        <canvas ref={canvasRef} className="preview-canvas" />

                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }} />
                        </div>

                        {stats && (
                            <div className="stats">
                                <span>Frame: {stats.frame + 1} / {stats.total_frames}</span>
                                <span>Progress: {progress.toFixed(1)}%</span>
                                <span>Objects: {stats.objects}</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Output Section */}
                {outputPath && (
                    <div className="output-section card">
                        <h2>✅ Colorization Complete!</h2>
                        <p>Output saved to: <code>{outputPath}</code></p>
                        <video controls className="output-video">
                            <source src={`${API_BASE}/${outputPath}`} type="video/mp4" />
                        </video>
                    </div>
                )}
            </div>

            <style jsx>{`
        .colorization-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .header {
          text-align: center;
          margin-bottom: 3rem;
        }

        .header h1 {
          font-size: 3rem;
          font-weight: 700;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 0.5rem;
        }

        .header p {
          color: #64748b;
          font-size: 1.125rem;
        }

        .main-content {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .card {
          background: white;
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          border: 1px solid #e2e8f0;
        }

        .card h2 {
          font-size: 1.5rem;
          font-weight: 600;
          color: #1e293b;
          margin-bottom: 1rem;
        }

        .file-input {
          width: 100%;
          padding: 1rem;
          border: 2px dashed #cbd5e1;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .file-input:hover {
          border-color: #667eea;
          background: #f8fafc;
        }

        .file-info {
          margin-top: 0.75rem;
          padding: 0.75rem;
          background: #f0fdf4;
          border-radius: 8px;
          color: #166534;
          font-weight: 500;
        }

        .prompt-input {
          width: 100%;
          padding: 1rem;
          border: 2px solid #e2e8f0;
          border-radius: 8px;
          font-size: 1rem;
          font-family: inherit;
          resize: vertical;
          transition: border-color 0.2s;
        }

        .prompt-input:focus {
          outline: none;
          border-color: #667eea;
        }

        .settings {
          margin-top: 1rem;
        }

        .settings label {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          font-weight: 500;
          color: #475569;
        }

        .settings input[type="range"] {
          width: 100%;
        }

        .hint {
          font-size: 0.875rem;
          color: #94a3b8;
          font-weight: 400;
        }

        .btn-primary, .btn-danger {
          padding: 1rem 2rem;
          font-size: 1.125rem;
          font-weight: 600;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          width: 100%;
        }

        .btn-primary {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 10px 20px -10px rgba(102, 126, 234, 0.5);
        }

        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-danger {
          background: #ef4444;
          color: white;
        }

        .btn-danger:hover {
          background: #dc2626;
        }

        .preview-canvas {
          width: 100%;
          max-width: 640px;
          height: auto;
          border-radius: 8px;
          margin: 0 auto;
          display: block;
          background: #000;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: #e2e8f0;
          border-radius: 4px;
          overflow: hidden;
          margin: 1rem 0;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
          transition: width 0.3s ease;
        }

        .stats {
          display: flex;
          justify-content: space-around;
          padding: 1rem;
          background: #f8fafc;
          border-radius: 8px;
          font-weight: 500;
          color: #475569;
        }

        .output-video {
          width: 100%;
          max-width: 800px;
          margin: 1rem auto;
          display: block;
          border-radius: 8px;
        }

        code {
          background: #f1f5f9;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-family: 'Fira Code', monospace;
          font-size: 0.875rem;
        }
      `}</style>
        </div>
    );
}
