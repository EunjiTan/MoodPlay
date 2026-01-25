import React, { useRef, useState } from 'react';
import axios from 'axios';

const Canvas = ({ videoPath }) => {
    const videoRef = useRef(null);
    const [clicks, setClicks] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleClick = async (e) => {
        if (!videoPath || loading) return;

        const rect = e.target.getBoundingClientRect();
        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;

        // Calculate Scale
        const video = videoRef.current;
        const scaleX = video.videoWidth / rect.width;
        const scaleY = video.videoHeight / rect.height;

        const trueX = Math.round(clientX * scaleX);
        const trueY = Math.round(clientY * scaleY);

        const newPoint = { x: clientX, y: clientY, trueX, trueY, label: 1 };
        const newClicks = [...clicks, newPoint];
        setClicks(newClicks);

        try {
            setLoading(true);
            const res = await axios.post('/api/segment/click', {
                video_path: videoPath.path || videoPath,
                frame_idx: 0,
                object_id: 1,
                points: [[trueX, trueY]],
                labels: [1]
            });
            console.log("Segmented:", res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // Correct URL construction for proxy
    // If videoPath is object: {filename: 'foo.mp4', path: '...'}
    const filename = videoPath?.filename || (typeof videoPath === 'string' ? videoPath.split(/[\\/]/).pop() : null);
    const videoSrc = filename ? `/api/uploads/${filename}` : null;

    return (
        <div className="relative w-full max-w-4xl mx-auto bg-black rounded-lg overflow-hidden border border-gray-700">
            {videoSrc ? (
                <>
                    <video
                        ref={videoRef}
                        src={videoSrc}
                        className="w-full h-auto cursor-crosshair"
                        controls={false}
                        onClick={handleClick}
                    />
                    <div className="absolute inset-0 pointer-events-none">
                        {clicks.map((c, i) => (
                            <div
                                key={i}
                                className="absolute w-4 h-4 rounded-full bg-green-500 border-2 border-white transform -translate-x-1/2 -translate-y-1/2"
                                style={{ left: c.x, top: c.y }}
                            />
                        ))}
                    </div>
                    {loading && (
                        <div className="absolute top-4 right-4 bg-black/80 text-white px-3 py-1 rounded-full text-sm">
                            Processing...
                        </div>
                    )}
                </>
            ) : (
                <div className="h-64 flex items-center justify-center text-gray-500">
                    Upload a video to start
                </div>
            )}
        </div>
    );
};

export default Canvas;
