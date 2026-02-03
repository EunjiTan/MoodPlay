import UnifiedPipeline from './components/UnifiedPipeline';

function App() {
    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
            <header className="max-w-4xl mx-auto mb-8 text-center">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-2">
                    🎬 MoodPlay
                </h1>
                <p className="text-slate-400">Next-Gen Video Colorization & Segmentation Pipeline</p>
            </header>

            <main className="max-w-4xl mx-auto">
                <UnifiedPipeline />
            </main>
        </div>
    )
}

export default App
