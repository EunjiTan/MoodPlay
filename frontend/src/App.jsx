import VideoProcessor from './components/VideoProcessor';

function App() {
    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
            <header className="max-w-6xl mx-auto flex justify-between items-center mb-12">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    MoodPlay v3.0 - Object Segmentation
                </h1>
            </header>

            <main className="max-w-6xl mx-auto">
                <div className="bg-slate-800 p-6 rounded-xl shadow-xl border border-slate-700">
                    <VideoProcessor />
                </div>
            </main>
        </div>
    )
}

export default App

