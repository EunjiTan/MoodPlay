
import React, { useState, useRef, useEffect } from 'react';

const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

const VideoProcessor = () => {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState("idle"); // idle, uploading, processing, completed
    const [progress, setProgress] = useState(0);
    const [stats, setStats] = useState(null);
    const [videoPath, setVideoPath] = useState(null);

    const canvasRef = useRef(null);
    const wsRef = useRef(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const uploadVideo = async () => {
        if (!file) return;
        setStatus("uploading");

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            setVideoPath(data.path);
            setStatus("uploaded");
        } catch (err) {
            console.error(err);
            setStatus("error");
        }
    };

    const startProcessing = () => {
        if (!videoPath) return;
        setStatus("processing");
        const sessionId = Date.now().toString();

        // Connect WS
        wsRef.current = new WebSocket(`${WS_BASE}/ws/process/${sessionId}`);

        wsRef.current.onopen = () => {
            // Send start command
            wsRef.current.send(JSON.stringify({
                command: "start",
                video_path: videoPath
            }));
        };

        wsRef.current.onmessage = async (event) => {
            if (typeof event.data === "string") {
                // JSON stats
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error(data.error);
                    setStatus("error");
                } else {
                    setStats(data);
                    if (data.frame && data.total_frames) {
                        setProgress((data.frame / data.total_frames) * 100);
                    }
                }
            } else {
                // Binary Image Data (Blob)
                const bitmap = await createImageBitmap(event.data);
                const canvas = canvasRef.current;
                if (canvas) {
                    const ctx = canvas.getContext("2d");
                    // Resize canvas to match image or fixed
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    ctx.drawImage(bitmap, 0, 0);
                }
            }
        };

        wsRef.current.onclose = () => {
            console.log("WS Closed");
            if (status === "processing") setStatus("completed");
        };
    };

    return (
        <div style={{ padding: 20, maxWidth: 800, margin: "0 auto" }}>
            <h2>Video Object Segmentation (YOLOv12 + SAM)</h2>

            <div style={{ marginBottom: 20 }}>
                <input type="file" accept="video/*" onChange={handleFileChange} />
                <button onClick={uploadVideo} disabled={!file || status === "uploading" || status === "processing"}>
                    {status === "uploading" ? "Uploading..." : "Upload"}
                </button>
            </div>

            {videoPath && (
                <div style={{ marginBottom: 20 }}>
                    <button onClick={startProcessing} disabled={status === "processing"}>
                        {status === "processing" ? "Processing..." : "Start Processing"}
                    </button>
                </div>
            )}

            {status === "processing" && (
                <div style={{ marginBottom: 10 }}>
                    <div style={{ width: "100%", backgroundColor: "#ddd", height: 20 }}>
                        <div style={{ width: `${progress}%`, backgroundColor: "green", height: "100%" }} />
                    </div>
                    <div>FPS: {stats?.fps?.toFixed(1)} | Objects: {stats?.objects}</div>
                </div>
            )}

            <div style={{ border: "1px solid #333", minHeight: 400, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#000' }}>
                <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />
            </div>

            <div style={{ marginTop: 20 }}>
                <h3>Detections Log</h3>
                <div style={{ height: 100, overflowY: "scroll", border: "1px solid #ccc", padding: 5 }}>
                    {stats?.detections?.map((d, i) => (
                        <div key={i}>{d.track_id ? `ID ${d.track_id}: ` : ''}{d.label} ({d.confidence.toFixed(2)})</div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default VideoProcessor;
