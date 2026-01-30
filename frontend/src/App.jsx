import { useState } from 'react';
import VideoProcessor from './components/VideoProcessor';
import ColorizationInterface from './components/ColorizationInterface';

function App() {
    const [activeTab, setActiveTab] = useState('colorization');

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
            <header className="max-w-6xl mx-auto mb-8">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-6">
                    🎬 MoodPlay v3.0
                </h1>

                <nav className="flex gap-4">
                    <button
                        onClick={() => setActiveTab('colorization')}
                        className={`px-6 py-3 rounded-lg font-semibold transition-all ${activeTab === 'colorization'
                                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                            }`}
                    >
                        🎨 Video Colorization
                    </button>
                    <button
                        onClick={() => setActiveTab('segmentation')}
                        className={`px-6 py-3 rounded-lg font-semibold transition-all ${activeTab === 'segmentation'
                                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                            }`}
                    >
                        🎯 Object Segmentation
                    </button>
                </nav>
            </header>

            <main className="max-w-6xl mx-auto">
                {activeTab === 'colorization' ? (
                    <ColorizationInterface />
                ) : (
                    <div className="bg-slate-800 p-6 rounded-xl shadow-xl border border-slate-700">
                        <VideoProcessor />
                    </div>
                )}
            </main>
        </div>
    )
}

export default App

