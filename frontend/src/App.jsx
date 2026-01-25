import React, { useState } from 'react';
import axios from 'axios';
import Canvas from './components/Canvas';

function App() {
    const [video, setVideo] = useState(null);
    const [uploading, setUploading] = useState(false);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('/api/upload', formData);
            setVideo(res.data);

            // Init Session
            await axios.post('/api/segment/init', null, { params: { video_path: res.data.path } });
        } catch (err) {
            console.error(err);
            alert("Upload failed");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
            <header className="max-w-6xl mx-auto flex justify-between items-center mb-12">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    MoodPlay v3.0
                </h1>
                <div className="flex gap-4">
                    <label className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded cursor-pointer transition">
                        {uploading ? 'Uploading...' : 'Upload Video'}
                        <input type="file" className="hidden" accept="video/*" onChange={handleUpload} disabled={uploading} />
                    </label>
                </div>
            </header>

            <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                    <div className="bg-slate-800 p-1 rounded-xl shadow-xl border border-slate-700">
                        <Canvas videoPath={video} />
                    </div>

                    <div className="mt-6 flex justify-between items-center bg-slate-800 p-4 rounded-lg border border-slate-700">
                        <div>
                            <h3 className="text-lg font-semibold">Timeline</h3>
                            <p className="text-slate-400 text-sm">Drag to seek (Coming Soon)</p>
                        </div>
                        <div className="flex gap-2">
                            <button className="p-2 bg-slate-700 rounded hover:bg-slate-600">Prev</button>
                            <button className="p-2 bg-slate-700 rounded hover:bg-slate-600">Play</button>
                            <button className="p-2 bg-slate-700 rounded hover:bg-slate-600">Next</button>
                        </div>
                    </div>
                </div>

                <div className="space-y-6">
                    {/* Control Panel */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <h2 className="text-xl font-bold mb-4">Style Engine</h2>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Prompt</label>
                                <textarea
                                    className="w-full bg-slate-900 border border-slate-600 rounded p-3 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                                    placeholder="e.g. A cyberpunk city with neon lights..."
                                    rows="3"
                                />
                            </div>

                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Style Preset</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {['Cinematic', 'Anime', 'Vintage', 'Noir'].map(style => (
                                        <button key={style} className="p-2 bg-slate-900 border border-slate-600 rounded text-sm hover:border-blue-500 transition">
                                            {style}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button className="w-full bg-gradient-to-r from-purple-600 to-blue-600 py-3 rounded font-bold hover:shadow-lg hover:shadow-blue-500/20 transition">
                                Generate Video
                            </button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}

export default App
