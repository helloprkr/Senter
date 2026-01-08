"use client";

import { useState, useCallback, useRef } from 'react';
import {
  startSpeechRecognition,
  stopSpeechRecognition,
  isSpeechRecognitionSupported,
} from '@/lib/whisper';

interface UseAudioRecordingOptions {
  onTranscript?: (text: string, isFinal: boolean) => void;
  onError?: (error: string) => void;
}

interface UseAudioRecordingReturn {
  isRecording: boolean;
  isSupported: boolean;
  transcript: string;
  interimTranscript: string;
  startRecording: () => void;
  stopRecording: () => string;
  clearTranscript: () => void;
  error: string | null;
}

export function useAudioRecording(options: UseAudioRecordingOptions = {}): UseAudioRecordingReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  const transcriptRef = useRef('');
  const isSupported = isSpeechRecognitionSupported();

  const startRecording = useCallback(() => {
    if (!isSupported) {
      const errorMsg = 'Speech recognition is not supported in this browser';
      setError(errorMsg);
      options.onError?.(errorMsg);
      return;
    }

    setError(null);
    transcriptRef.current = '';
    setTranscript('');
    setInterimTranscript('');

    const started = startSpeechRecognition({
      onStart: () => {
        setIsRecording(true);
      },
      onResult: (result) => {
        if (result.isFinal) {
          transcriptRef.current += result.text + ' ';
          setTranscript(transcriptRef.current.trim());
          setInterimTranscript('');
          options.onTranscript?.(result.text, true);
        } else {
          setInterimTranscript(result.text);
          options.onTranscript?.(result.text, false);
        }
      },
      onError: (err) => {
        setError(err);
        setIsRecording(false);
        options.onError?.(err);
      },
      onEnd: () => {
        setIsRecording(false);
        setInterimTranscript('');
      },
    });

    if (!started) {
      setError('Failed to start speech recognition');
    }
  }, [isSupported, options]);

  const stopRecording = useCallback((): string => {
    stopSpeechRecognition();
    setIsRecording(false);
    setInterimTranscript('');
    return transcriptRef.current.trim();
  }, []);

  const clearTranscript = useCallback(() => {
    transcriptRef.current = '';
    setTranscript('');
    setInterimTranscript('');
  }, []);

  return {
    isRecording,
    isSupported,
    transcript,
    interimTranscript,
    startRecording,
    stopRecording,
    clearTranscript,
    error,
  };
}

