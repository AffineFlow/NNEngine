import { useEffect, useRef, useState } from 'react';
import initNNEngine from '../../public/nnengine.js';

export default function DigitDetector() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const classifierRef = useRef<any>(null);
  const [isReady, setIsReady] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function initialize() {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}digit_weights.nne`);
        const buffer = await response.arrayBuffer();
        const Module = await initNNEngine({
          locateFile: (path: string) => `${import.meta.env.BASE_URL}${path}`
        });

        try {
          Module.FS.writeFile('digit_weights.nne', new Uint8Array(buffer));
        } catch (e) { /* file exists */ }

        if (isMounted) {
          classifierRef.current = new Module.WasmDigitClassifier();
          setIsReady(true);
        }
      } catch (err) {
        console.error("Error init:", err);
      }
    }
    initialize();
    
    // Set up canvas drawing context
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, 280, 280);
      ctx.lineCap = 'round';
      ctx.lineWidth = 20;
      ctx.strokeStyle = 'white';
    }
    
    return () => { isMounted = false; };
  }, []);

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, 280, 280);
      ctx.beginPath(); // Reset path to prevent connecting strokes
      setPrediction(null);
      setConfidence(0);
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    // runInference() REMOVED FROM HERE to prevent Wasm memory overflow
  };

  const runInference = () => {
    if (!classifierRef.current || !canvasRef.current) return;
    
    // Create an offscreen 28x28 canvas to downscale the drawing
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext('2d');
    
    if (smallCtx) {
      // FIX 1: MNIST Bounding Box Padding
      // Fill with black first, then draw the user's canvas shrunk down 
      // into a 20x20 box, offset by 4 pixels to center it.
      smallCtx.fillStyle = 'black';
      smallCtx.fillRect(0, 0, 28, 28);
      smallCtx.drawImage(canvasRef.current, 4, 4, 20, 20);
      
      const imgData = smallCtx.getImageData(0, 0, 28, 28).data;
      const grayInput = new Float32Array(28 * 28);
      
      let maxIntensity = 0;

      // Extract the red channel[cite: 2] and track the maximum pixel brightness
      for (let i = 0; i < 28 * 28; i++) {
        let val = imgData[i * 4] / 255.0; 
        grayInput[i] = val;
        if (val > maxIntensity) {
          maxIntensity = val;
        }
      }

      // FIX 2: Intensity Normalization
      // Scale all pixels relative to the brightest pixel to counter interpolation dimming,
      // ensuring the peak intensity matches the 1.0 format used during training[cite: 3].
      if (maxIntensity > 0) {
        for (let i = 0; i < 28 * 28; i++) {
          grayInput[i] = grayInput[i] / maxIntensity;
        }
      }

      const preds = classifierRef.current.predict(grayInput);
      if (preds && Array.isArray(preds)) {
        let maxIdx = 0;
        let maxVal = preds[0];
        for (let i = 1; i < 10; i++) {
          if (preds[i] > maxVal) {
            maxVal = preds[i];
            maxIdx = i;
          }
        }
        setPrediction(maxIdx);
        setConfidence(maxVal * 100);
      }
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>MNIST Digit Recognizer</h2>
      <p>{isReady ? "Draw a number (0-9) then click Predict!" : "Loading Engine..."}</p>
      
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '2px solid #333', cursor: 'crosshair', margin: '0 auto', display: 'block' }}
        onMouseDown={(e) => { setIsDrawing(true); draw(e); }}
        onMouseMove={draw}
        onMouseUp={() => { setIsDrawing(false); canvasRef.current?.getContext('2d')?.beginPath(); }}
        onMouseLeave={() => { setIsDrawing(false); canvasRef.current?.getContext('2d')?.beginPath(); }}
      />
      
      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', gap: '15px' }}>
        <button onClick={clearCanvas} style={{ padding: '10px 20px', cursor: 'pointer', borderRadius: '6px' }}>
          Clear
        </button>
        <button 
          onClick={runInference} 
          disabled={!isReady}
          style={{ 
            padding: '10px 20px', 
            cursor: isReady ? 'pointer' : 'not-allowed', 
            borderRadius: '6px', 
            background: '#0ea5e9', 
            color: 'white',
            fontWeight: 'bold',
            border: 'none'
          }}>
          Predict Digit
        </button>
      </div>

      {prediction !== null && (
        <div style={{ marginTop: '25px', padding: '15px', background: '#f8fafc', borderRadius: '8px', display: 'inline-block' }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#0f172a' }}>Predicted Digit: <span style={{ color: '#0ea5e9', fontSize: '1.5em' }}>{prediction}</span></h3>
          <p style={{ margin: 0, color: '#64748b' }}>Confidence: {confidence.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}