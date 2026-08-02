import { Link } from 'react-router-dom';

const DEMOS = [
  {
    id: 'emotion',
    title: 'Facial Emotion Classifier',
    description: 'Real-time deep CNN inference analyzing webcam feeds natively in-browser.',
    path: '/emotion-detector',
    status: 'Live',
  },
  {
    id: 'digits',
    title: 'Handwritten Digit Recognizer',
    description: 'Draw a number and let the engine evaluate the feedforward MLP graph.',
    path: '/digit-detector',
    status: 'Live',
  }
];

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="text-center mb-16">
        <h1 className="text-5xl md:text-6xl font-extrabold text-white tracking-tight mb-6">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">NNEngine</span> Web
        </h1>
        <p className="text-xl text-slate-400 max-w-2xl mx-auto">
          High-performance C++ neural networks compiled to WebAssembly. Zero allocations. Zero Python GIL. Running entirely on your device.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        {DEMOS.map((demo) => (
          <Link 
            key={demo.id} 
            to={demo.path}
            className={`group relative bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 transition-all duration-300 flex flex-col ${demo.status === 'Live' ? 'hover:border-cyan-500/50 hover:shadow-[0_0_40px_-10px_rgba(34,211,238,0.2)]' : 'opacity-75 cursor-not-allowed'}`}
            onClick={(e) => demo.status !== 'Live' && e.preventDefault()}
          >
            <div className="flex justify-between items-start mb-4">
              <h3 className={`text-2xl font-bold ${demo.status === 'Live' ? 'text-white group-hover:text-cyan-400' : 'text-slate-300'} transition-colors`}>
                {demo.title}
              </h3>
              <span className={`px-3 py-1 text-xs font-semibold rounded-full ${demo.status === 'Live' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'bg-slate-800 text-slate-400'}`}>
                {demo.status}
              </span>
            </div>
            <p className="text-slate-400 text-base leading-relaxed">
              {demo.description}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}