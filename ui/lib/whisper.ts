"use client";

// Whisper Speech-to-Text using Web Speech API with fallback
// Note: For a fully local solution, you would use @xenova/transformers with Whisper
// But that requires significant download (~40MB+) and complex setup
// This implementation uses the browser's built-in speech recognition as primary

let recognitionInstance: SpeechRecognition | null = null;

// Check if Web Speech API is available
export function isSpeechRecognitionSupported(): boolean {
  if (typeof window === 'undefined') return false;
  return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
}

// Initialize speech recognition
export function getSpeechRecognition(): SpeechRecognition | null {
  if (typeof window === 'undefined') return null;
  
  const SpeechRecognitionAPI = (window as typeof window & {
    SpeechRecognition?: typeof SpeechRecognition;
    webkitSpeechRecognition?: typeof SpeechRecognition;
  }).SpeechRecognition || (window as typeof window & {
    webkitSpeechRecognition?: typeof SpeechRecognition;
  }).webkitSpeechRecognition;
  
  if (!SpeechRecognitionAPI) return null;
  
  if (!recognitionInstance) {
    recognitionInstance = new SpeechRecognitionAPI();
    recognitionInstance.continuous = true;
    recognitionInstance.interimResults = true;
    recognitionInstance.lang = 'en-US';
  }
  
  return recognitionInstance;
}

interface TranscriptionResult {
  text: string;
  isFinal: boolean;
  confidence: number;
}

interface SpeechRecognitionCallbacks {
  onResult?: (result: TranscriptionResult) => void;
  onError?: (error: string) => void;
  onEnd?: () => void;
  onStart?: () => void;
}

let isListeningState = false;

export function startSpeechRecognition(callbacks: SpeechRecognitionCallbacks): boolean {
  const recognition = getSpeechRecognition();
  
  if (!recognition) {
    callbacks.onError?.('Speech recognition not supported in this browser');
    return false;
  }
  
  if (isListeningState) {
    return true; // Already listening
  }
  
  recognition.onstart = () => {
    isListeningState = true;
    callbacks.onStart?.();
  };
  
  recognition.onresult = (event: SpeechRecognitionEvent) => {
    let finalTranscript = '';
    let interimTranscript = '';
    
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const transcript = result[0].transcript;
      const confidence = result[0].confidence;
      
      if (result.isFinal) {
        finalTranscript += transcript;
        callbacks.onResult?.({
          text: transcript,
          isFinal: true,
          confidence: confidence || 0.9,
        });
      } else {
        interimTranscript += transcript;
        callbacks.onResult?.({
          text: transcript,
          isFinal: false,
          confidence: confidence || 0.5,
        });
      }
    }
  };
  
  recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
    isListeningState = false;
    callbacks.onError?.(event.error);
  };
  
  recognition.onend = () => {
    isListeningState = false;
    callbacks.onEnd?.();
  };
  
  try {
    recognition.start();
    return true;
  } catch (error) {
    callbacks.onError?.('Failed to start speech recognition');
    return false;
  }
}

export function stopSpeechRecognition(): void {
  const recognition = getSpeechRecognition();
  if (recognition && isListeningState) {
    recognition.stop();
    isListeningState = false;
  }
}

export function isListening(): boolean {
  return isListeningState;
}

// Alternative: Local audio recording for future Whisper integration
export interface AudioRecordingResult {
  blob: Blob;
  duration: number;
}

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordingStartTime: number = 0;

export async function startAudioRecording(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    });
    
    audioChunks = [];
    recordingStartTime = Date.now();
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    mediaRecorder.start(100); // Collect data every 100ms
    return true;
  } catch (error) {
    console.error('Failed to start audio recording:', error);
    return false;
  }
}

export async function stopAudioRecording(): Promise<AudioRecordingResult | null> {
  return new Promise((resolve) => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      resolve(null);
      return;
    }
    
    mediaRecorder.onstop = () => {
      const duration = Date.now() - recordingStartTime;
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      
      // Stop all tracks
      mediaRecorder?.stream.getTracks().forEach(track => track.stop());
      
      resolve({ blob, duration });
    };
    
    mediaRecorder.stop();
  });
}

export function isRecording(): boolean {
  return mediaRecorder?.state === 'recording';
}

