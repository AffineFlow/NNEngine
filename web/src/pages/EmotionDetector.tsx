import { useEffect, useRef, useState } from 'react';
import initNNEngine from '../../public/nnengine.js';

// Standard FER-2013 label order matching the mapping above
const EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

export default function EmotionDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const classifierRef = useRef<any>(null);
  
  const [isReady, setIsReady] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [streamActive, setStreamActive] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function initialize() {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}emotion_weights.nne`);
        const buffer = await response.arrayBuffer();
        const Module = await initNNEngine({
          locateFile: (path: string) => `${import.meta.env.BASE_URL}${path}`
        });

        try {
          Module.FS.writeFile('emotion_weights.nne', new Uint8Array(buffer));
        } catch (e) { /* file exists */ }

        if (isMounted) {
          classifierRef.current = new Module.WasmClassifier(7);
          setIsReady(true);
        }
      } catch (err) {
        console.error("Error init:", err);
      }
    }
    initialize();
    
    return () => { isMounted = false; };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 320, height: 240, facingMode: 'user' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreamActive(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStreamActive(false);
    }
  };

  const runInference = () => {
    if (!classifierRef.current || !videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
      // 1. Crop center of webcam
      const minDim = Math.min(video.videoWidth, video.videoHeight);
      const startX = (video.videoWidth - minDim) / 2;
      const startY = (video.videoHeight - minDim) / 2;
      ctx.drawImage(video, startX, startY, minDim, minDim, 0, 0, 64, 64);
      
      const imgData = ctx.getImageData(0, 0, 64, 64);
      const pixels = imgData.data;
      const grayInput = new Float32Array(64 * 64);
      
      // 2. Grayscale conversion
      for (let i = 0; i < 64 * 64; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];
        const gray = (0.299 * r + 0.587 * g + 0.114 * b);
        
        grayInput[i] = gray / 255.0; // Scaled to 0.0 -> 1.0

        // 3. Paint the grayscale back to the canvas buffer for us to verify
        pixels[i * 4] = gray;     // R
        pixels[i * 4 + 1] = gray; // G
        pixels[i * 4 + 2] = gray; // B
        pixels[i * 4 + 3] = 255;  // Alpha
      }

      // 4. Update the debug canvas
      ctx.putImageData(imgData, 0, 0);

      // 5. Predict
      const preds = classifierRef.current.predict(grayInput);
      
      if (preds && Array.isArray(preds)) {
        let maxIdx = 0;
        let maxVal = preds[0];
        // Iterate through all 7 classes
        for (let i = 1; i < 7; i++) {
          if (preds[i] > maxVal) {
            maxVal = preds[i];
            maxIdx = i;
          }
        }
        setPrediction(EMOTION_LABELS[maxIdx]);
        setConfidence(maxVal * 100);
      }
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>Facial Emotion Detector</h2>
      <p>{isReady ? "Engine Loaded. Start the webcam to test." : "Loading Engine..."}</p>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '20px' }}>
        {/* Live Webcam Feed */}
        <div style={{ background: '#000', borderRadius: '8px', overflow: 'hidden', width: '320px', height: '240px' }}>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        </div>
        
        {/* Visible 64x64 Debug Canvas */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <p style={{ margin: '0 0 5px 0', fontSize: '12px', color: '#64748b' }}>What the CNN sees:</p>
          <canvas 
            ref={canvasRef} 
            width={64} 
            height={64} 
            style={{ border: '1px solid #0ea5e9', width: '128px', height: '128px', imageRendering: 'pixelated' }} 
          />
        </div>
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '15px' }}>
        {!streamActive ? (
          <button onClick={startWebcam} style={{ padding: '10px 20px', borderRadius: '6px', cursor: 'pointer' }}>
            Start Camera
          </button>
        ) : (
          <button onClick={stopWebcam} style={{ padding: '10px 20px', borderRadius: '6px', cursor: 'pointer' }}>
            Stop Camera
          </button>
        )}
        
        <button 
          onClick={runInference} 
          disabled={!isReady || !streamActive}
          style={{ 
            padding: '10px 20px', 
            cursor: (isReady && streamActive) ? 'pointer' : 'not-allowed', 
            borderRadius: '6px', 
            background: '#0ea5e9', 
            color: 'white',
            fontWeight: 'bold',
            border: 'none'
          }}>
          Predict Emotion
        </button>
      </div>

      {prediction !== null && (
        <div style={{ marginTop: '25px', padding: '15px', background: '#f8fafc', borderRadius: '8px', display: 'inline-block' }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#0f172a' }}>Detected: <span style={{ color: '#0ea5e9', fontSize: '1.5em' }}>{prediction}</span></h3>
          <p style={{ margin: 0, color: '#64748b' }}>Confidence: {confidence.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}