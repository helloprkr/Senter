"use client";

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Camera, RotateCcw, Check } from "lucide-react";
import { useCamera } from "@/hooks/use-camera";

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (imageData: string) => void;
}

export function CameraModal({ isOpen, onClose, onCapture }: CameraModalProps) {
  const { videoRef, isActive, error, startCamera, stopCamera, captureImage, switchCamera, facingMode } = useCamera();

  useEffect(() => {
    if (isOpen) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [isOpen, startCamera, stopCamera]);

  const handleCapture = async () => {
    const imageData = await captureImage();
    if (imageData) {
      onCapture(imageData);
      onClose();
    }
  };

  const handleClose = () => {
    stopCamera();
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative w-full max-w-2xl glass-panel overflow-hidden" data-ui-panel>
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-brand-light/10">
                <h2 className="text-lg font-medium text-brand-light">Camera</h2>
                <motion.button
                  onClick={handleClose}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/10 hover:bg-brand-light/20 transition-colors"
                >
                  <X className="w-4 h-4 text-brand-light/70" />
                </motion.button>
              </div>

              {/* Video Preview */}
              <div className="relative aspect-video bg-black">
                {error ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <Camera className="w-12 h-12 text-brand-light/30 mx-auto mb-3" />
                      <p className="text-brand-light/70 text-sm">{error}</p>
                      <motion.button
                        onClick={startCamera}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="mt-3 px-4 py-2 rounded-lg bg-brand-accent text-brand-light text-sm"
                      >
                        Try Again
                      </motion.button>
                    </div>
                  </div>
                ) : !isActive ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-12 h-12 rounded-full border-2 border-brand-light/30 border-t-brand-accent animate-spin mx-auto mb-3" />
                      <p className="text-brand-light/50 text-sm">Starting camera...</p>
                    </div>
                  </div>
                ) : null}
                
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${facingMode === 'user' ? 'scale-x-[-1]' : ''}`}
                  style={{ display: isActive && !error ? 'block' : 'none' }}
                />
                
                {/* Camera overlay guides */}
                {isActive && !error && (
                  <div className="absolute inset-0 pointer-events-none">
                    {/* Corner guides */}
                    <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-brand-light/50 rounded-tl-lg" />
                    <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-brand-light/50 rounded-tr-lg" />
                    <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-brand-light/50 rounded-bl-lg" />
                    <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-brand-light/50 rounded-br-lg" />
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center justify-center gap-6 p-6">
                {/* Switch Camera */}
                <motion.button
                  onClick={switchCamera}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  disabled={!isActive}
                  className="w-12 h-12 rounded-full flex items-center justify-center bg-brand-light/10 hover:bg-brand-light/20 transition-colors disabled:opacity-50"
                >
                  <RotateCcw className="w-5 h-5 text-brand-light/70" />
                </motion.button>

                {/* Capture Button */}
                <motion.button
                  onClick={handleCapture}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  disabled={!isActive}
                  className="w-16 h-16 rounded-full flex items-center justify-center bg-brand-accent hover:bg-brand-accent/90 transition-colors disabled:opacity-50 shadow-lg"
                >
                  <Camera className="w-7 h-7 text-brand-light" />
                </motion.button>

                {/* Placeholder for symmetry */}
                <div className="w-12 h-12" />
              </div>

              {/* Hint */}
              <div className="text-center pb-4">
                <p className="text-brand-light/40 text-xs">
                  Capture an image to include in your message
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

