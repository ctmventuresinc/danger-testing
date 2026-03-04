"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  ImageSegmenter,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

export default function PoseTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const glowCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const animationFrameRef = useRef<number>(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);

  const initSegmenter = useCallback(async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      segmenterRef.current = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });

      setIsLoading(false);
    } catch (err) {
      console.error("Failed to init ImageSegmenter:", err);
      setError("Failed to load segmentation model. Please refresh.");
      setIsLoading(false);
    }
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
      }
    } catch (err) {
      console.error("Camera error:", err);
      setError("Camera access denied. Please allow camera permissions.");
    }
  }, []);

  const renderFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const segmenter = segmenterRef.current;

    if (!video || !canvas || !segmenter || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const w = video.videoWidth;
    const h = video.videoHeight;
    canvas.width = w;
    canvas.height = h;

    // Init offscreen canvases once
    if (!maskCanvasRef.current) {
      maskCanvasRef.current = document.createElement("canvas");
      glowCanvasRef.current = document.createElement("canvas");
    }
    const maskCanvas = maskCanvasRef.current;
    const glowCanvas = glowCanvasRef.current!;
    maskCanvas.width = w;
    maskCanvas.height = h;
    glowCanvas.width = w;
    glowCanvas.height = h;

    const ctx = canvas.getContext("2d")!;
    const maskCtx = maskCanvas.getContext("2d")!;
    const glowCtx = glowCanvas.getContext("2d")!;

    const result = segmenter.segmentForVideo(video, performance.now());

    // Draw original video as background
    ctx.drawImage(video, 0, 0, w, h);

    if (result.confidenceMasks && result.confidenceMasks.length > 0) {
      const mask = result.confidenceMasks[0];
      const maskData = mask.getAsFloat32Array();

      // Build mask as alpha image on maskCanvas
      const imageData = maskCtx.createImageData(w, h);
      const pixels = imageData.data;

      for (let i = 0; i < maskData.length; i++) {
        const confidence = maskData[i];
        const idx = i * 4;
        // White pixel with alpha based on confidence
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        pixels[idx + 3] = confidence > 0.5 ? 255 : 0;
      }

      maskCtx.putImageData(imageData, 0, 0);

      // Draw gradient on glowCanvas, masked by body shape
      glowCtx.clearRect(0, 0, w, h);

      // First draw the mask
      glowCtx.drawImage(maskCanvas, 0, 0);

      // Then draw gradient only inside the mask using source-in
      glowCtx.globalCompositeOperation = "source-in";
      const gradient = glowCtx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "#0033FF");
      gradient.addColorStop(0.3, "#4400CC");
      gradient.addColorStop(0.55, "#7700AA");
      gradient.addColorStop(0.75, "#AA0088");
      gradient.addColorStop(1, "#FF00AA");
      glowCtx.fillStyle = gradient;
      glowCtx.fillRect(0, 0, w, h);
      glowCtx.globalCompositeOperation = "source-over";

      // Draw glow on main canvas
      ctx.save();
      ctx.shadowColor = "#FF00FF";
      ctx.shadowBlur = 25;
      ctx.globalCompositeOperation = "screen";
      ctx.drawImage(glowCanvas, 0, 0);
      ctx.drawImage(glowCanvas, 0, 0);
      ctx.restore();

      mask.close();
    }

    animationFrameRef.current = requestAnimationFrame(renderFrame);
  }, []);

  useEffect(() => {
    initSegmenter();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (segmenterRef.current) {
        segmenterRef.current.close();
      }
    };
  }, [initSegmenter]);

  useEffect(() => {
    if (!isLoading && !error) {
      startCamera();
    }
  }, [isLoading, error, startCamera]);

  useEffect(() => {
    if (cameraActive) {
      renderFrame();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [cameraActive, renderFrame]);

  if (error) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-black">
        <p className="text-red-500 text-lg px-6 text-center">{error}</p>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-full overflow-hidden bg-black">
      {isLoading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-purple-500 border-t-transparent" />
          <p className="text-purple-400 text-sm font-mono">
            Loading body tracker...
          </p>
        </div>
      )}

      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover -scale-x-100 opacity-0"
        playsInline
        muted
      />

      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-cover -scale-x-100"
      />

      <div className="absolute top-4 left-4 z-10">
        <div className="flex items-center gap-2 rounded-full bg-black/60 px-4 py-2 backdrop-blur-sm">
          <div
            className={`h-2.5 w-2.5 rounded-full ${
              cameraActive ? "bg-purple-500 animate-pulse" : "bg-red-500"
            }`}
          />
          <span className="text-white text-xs font-mono tracking-wider uppercase">
            {cameraActive ? "Tracking" : "Initializing"}
          </span>
        </div>
      </div>
    </div>
  );
}
