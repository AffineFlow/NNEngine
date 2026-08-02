import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import EmotionDetector from './pages/EmotionDetector';
import DigitDetector from './pages/DigitDetector';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans selection:bg-cyan-500/30">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/emotion-detector" element={<EmotionDetector />} />
          <Route path="/digit-detector" element={<DigitDetector />} />
        </Routes>
      </main>
    </div>
  );
}