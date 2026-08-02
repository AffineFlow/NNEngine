import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/80 border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="text-xl font-bold text-cyan-400 tracking-tight hover:text-cyan-300 transition-colors">
            NNEngine Hub
          </Link>
          <div className="flex space-x-4">
            <Link to="/" className="text-slate-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Demos
            </Link>
            <a href="https://github.com" target="_blank" rel="noreferrer" className="text-slate-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
              GitHub
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}