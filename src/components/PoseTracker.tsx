"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  ImageSegmenter,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

export default function PoseTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
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
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
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

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const offscreen = offscreenCanvasRef.current;
    offscreen.width = w;
    offscreen.height = h;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const offCtx = offscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx || !offCtx) return;

    const result = segmenter.segmentForVideo(video, performance.now());

    // Draw the original video as background
    ctx.drawImage(video, 0, 0, w, h);

    if (result.confidenceMasks && result.confidenceMasks.length > 0) {
      // The selfie_multiclass model returns multiple masks
      // Index 0 = background, others are body parts
      // We want any non-background (person) pixels
      // For selfie_multiclass: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others(accessories)
      const masks = result.confidenceMasks;

      // Build combined person mask from all non-background channels
      const personMask = new Float32Array(w * h);
      for (let i = 1; i < masks.length; i++) {
        const maskData = masks[i].getAsFloat32Array();
        for (let j = 0; j < maskData.length; j++) {
          personMask[j] = Math.min(1, personMask[j] + maskData[j]);
        }
      }

      // Create gradient on offscreen canvas
      offCtx.clearRect(0, 0, w, h);
      const gradient = offCtx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "#0033FF");    // deep blue at feet
      gradient.addColorStop(0.3, "#4400CC");  // blue-purple
      gradient.addColorStop(0.55, "#7700AA"); // purple
      gradient.addColorStop(0.75, "#AA0088"); // purple-pink
      gradient.addColorStop(1, "#FF00AA");    // hot pink at top
      offCtx.fillStyle = gradient;
      offCtx.fillRect(0, 0, w, h);

      // Get gradient pixels
      const gradientData = offCtx.getImageData(0, 0, w, h);
      const gradPixels = gradientData.data;

      // Apply mask to gradient — set alpha based on confidence
      for (let i = 0; i < personMask.length; i++) {
        const confidence = personMask[i];
        const pIdx = i * 4;
        if (confidence < 0.4) {
          gradPixels[pIdx + 3] = 0; // fully transparent
        } else {
          gradPixels[pIdx + 3] = Math.floor(confidence * 220); // semi-transparent overlay
        }
      }

      offCtx.putImageData(gradientData, 0, 0);

      // Draw glow effect — draw the silhouette multiple times with increasing shadow
      ctx.save();

      // Outer glow layer
      ctx.shadowColor = "#FF00FF";
      ctx.shadowBlur = 30;
      ctx.globalCompositeOperation = "screen";
      ctx.drawImage(offscreen, 0, 0);
      ctx.drawImage(offscreen, 0, 0);

      // Inner sharper glow
      ctx.shadowBlur = 10;
      ctx.shadowColor = "#CC44FF";
      ctx.drawImage(offscreen, 0, 0);

      ctx.restore();

      // Draw the gradient silhouette on top cleanly
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ctx.globalAlpha = 0.85;
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();

      // Close masks to free memory
      for (const mask of masks) {
        mask.close();
      }
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
