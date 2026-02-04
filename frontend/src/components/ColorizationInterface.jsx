import React, { useState, useRef, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

export default function ColorizationInterface() {
  const [videoFile, setVideoFile] = useState(null);
  const [uploadedPath, setUploadedPath] = useState('');
  const [selectedStyle, setSelectedStyle] = useState('sunny_day');
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

  const startColorization = async () => {
    if (!uploadedPath) {
      alert('Please upload a video first');
      return;
    }

    setIsProcessing(true);
    setProgress(0); // Indeterminate or just 0
    setOutputPath('');
    setStats({ message: `Applying '${selectedStyle.toUpperCase()}' Palette... This may take a few minutes.` });

    try {
      // New Staged Pipeline (REST API)
      // This is a long-running request (minutes). The browser "Processing" state will hold.
      // Ideally we'd implement WebSocket progress, but for V1 we used the blocked API.

      // Import dynamically or assume it's available via api.js import
      const { startStagedColorization } = await import('../api');

      const data = await startStagedColorization(uploadedPath, selectedStyle);

      if (data.status === 'completed') {
        setOutputPath(data.output_path);
        setProgress(100);
      }
    } catch (error) {
      console.error(error);
      alert("Colorization failed (or timed out). Check server logs.");
    } finally {
      setIsProcessing(false);
    }
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
          <h2>2. Choose the Mood</h2>

          <div className="style-grid">
            {[
              { id: 'sunny_day', label: '☀️ Sunny Day', desc: 'Bright, vibrant, warm golden light' },
              { id: 'winter', label: '❄️ Winter', desc: 'Cold, snowy, crisp blue tones' },
              { id: 'golden_hour', label: '🌅 Golden Hour', desc: 'Sunset glow, dramatic shadows' },
              { id: 'overcast', label: '☁️ Overcast', desc: 'Soft diffused light, neutral tones' },
              { id: 'night_scene', label: '🌙 Night Scene', desc: 'Dark, cinematic, artificial lights' },
              { id: 'autumn', label: '🍂 Autumn', desc: 'Rich earth tones, orange leaves' },
              { id: 'spring', label: '🌸 Spring', desc: 'Fresh greens, pastel flowers' },
              { id: 'desert', label: '🏜️ Desert', desc: 'Hot, dry, sandy yellow tones' },
              { id: 'tropical', label: '🌴 Tropical', desc: 'Vibrant turquoise, lush greens' },
              { id: 'vintage', label: '📽️ Vintage Film', desc: 'Faded, classic analog look' }
            ].map(style => (
              <button
                key={style.id}
                className={`style-card ${selectedStyle === style.id ? 'active' : ''}`}
                onClick={() => setSelectedStyle(style.id)}
                disabled={isProcessing}
              >
                <div className="style-label">{style.label}</div>
                <div className="style-desc">{style.desc}</div>
              </button>
            ))}
          </div>

          <div className="settings">
            {/* Steps hidden for simplicity/magic feel */}
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

        .style-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .style-card {
            background: #f1f5f9;
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }

        .style-card:hover:not(:disabled) {
            background: #e2e8f0;
        }

        .style-card.active {
            border-color: #667eea;
            background: #e0e7ff;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }

        .style-label {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.25rem;
        }

        .style-desc {
            font-size: 0.8rem;
            color: #64748b;
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
