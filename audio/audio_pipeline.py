#!/usr/bin/env python3
"""
Audio Pipeline

Handles all audio I/O for Senter:
- Voice Activity Detection (VAD)
- Speech-to-Text (STT) via Whisper
- Text-to-Speech (TTS)
- Audio stream management

Works with gaze detection: only transcribes when user is paying attention.
"""

import time
import logging
import sys
import subprocess
import threading
from typing import Optional
from pathlib import Path
from multiprocessing import Event
from queue import Empty

# Check dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon.message_bus import MessageBus, MessageType, Message

logger = logging.getLogger('senter.audio')


class AudioBuffer:
    """
    Ring buffer for audio data.
    """

    def __init__(self, sample_rate: int = 16000, buffer_seconds: int = 30):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for AudioBuffer")

        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.speech_start_pos = None
        self._lock = threading.Lock()

    def write(self, data: np.ndarray):
        """Write audio data to buffer"""
        with self._lock:
            data = data.flatten()
            data_len = len(data)

            if data_len >= self.buffer_size:
                self.buffer[:] = data[-self.buffer_size:]
                self.write_pos = 0
            else:
                end_pos = self.write_pos + data_len
                if end_pos <= self.buffer_size:
                    self.buffer[self.write_pos:end_pos] = data
                else:
                    first_part = self.buffer_size - self.write_pos
                    self.buffer[self.write_pos:] = data[:first_part]
                    self.buffer[:data_len - first_part] = data[first_part:]
                self.write_pos = end_pos % self.buffer_size

    def get_recent(self, seconds: float) -> np.ndarray:
        """Get recent audio data"""
        with self._lock:
            samples = int(seconds * self.sample_rate)
            if samples > self.buffer_size:
                samples = self.buffer_size

            if self.write_pos >= samples:
                return self.buffer[self.write_pos - samples:self.write_pos].copy()
            else:
                return np.concatenate([
                    self.buffer[-(samples - self.write_pos):],
                    self.buffer[:self.write_pos]
                ])

    def mark_speech_start(self):
        """Mark current position as speech start"""
        with self._lock:
            self.speech_start_pos = self.write_pos

    def get_speech_segment(self) -> Optional[np.ndarray]:
        """Get audio from speech start to now"""
        with self._lock:
            if self.speech_start_pos is None:
                return None

            if self.write_pos > self.speech_start_pos:
                segment = self.buffer[self.speech_start_pos:self.write_pos].copy()
            else:
                segment = np.concatenate([
                    self.buffer[self.speech_start_pos:],
                    self.buffer[:self.write_pos]
                ])

            self.speech_start_pos = None
            return segment


class VoiceActivityDetector:
    """
    Detects voice activity in audio.
    Uses energy-based detection with optional Silero VAD.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._silero_model = None
        self._use_silero = False

        # Try to load Silero VAD
        try:
            import torch
            self._silero_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self._use_silero = True
            logger.info("Loaded Silero VAD")
        except Exception as e:
            logger.info(f"Using energy-based VAD (Silero unavailable: {e})")

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech"""
        if self._use_silero and self._silero_model is not None:
            try:
                import torch
                audio_tensor = torch.from_numpy(audio)
                speech_prob = self._silero_model(audio_tensor, 16000).item()
                return speech_prob > self.threshold
            except:
                pass

        # Fallback: energy-based detection
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > 0.01


class STTEngine:
    """
    Speech-to-Text engine using Whisper.
    """

    def __init__(self, model_name: str = "small"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None and WHISPER_AVAILABLE:
            try:
                # Map friendly names
                model_map = {
                    "whisper-tiny": "tiny",
                    "whisper-base": "base",
                    "whisper-small": "small",
                    "whisper-medium": "medium",
                }
                model_size = model_map.get(self.model_name, self.model_name)
                self._model = whisper.load_model(model_size)
                logger.info(f"Loaded Whisper model: {model_size}")
            except Exception as e:
                logger.error(f"Could not load Whisper: {e}")
        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text"""
        if self.model is None:
            return ""

        try:
            result = self.model.transcribe(
                audio,
                language="en",
                fp16=False
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


class TTSEngine:
    """
    Text-to-Speech engine.
    Uses system TTS as fallback (macOS 'say', etc.)
    """

    def __init__(self, model_name: str = "system"):
        self.model_name = model_name
        self._piper_model = None

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio"""
        # Try Piper first
        if self.model_name != "system":
            try:
                return self._synthesize_piper(text)
            except:
                pass

        # Fallback to system TTS
        return self._synthesize_system(text)

    def _synthesize_piper(self, text: str) -> np.ndarray:
        """Synthesize using Piper TTS"""
        try:
            from piper import PiperVoice
            # Would load piper model here
            raise NotImplementedError("Piper TTS not configured")
        except ImportError:
            raise

    def _synthesize_system(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using system TTS"""
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                    temp_path = f.name

                subprocess.run(
                    ["say", "-o", temp_path, text],
                    capture_output=True,
                    timeout=30
                )

                # Load and return audio
                try:
                    import soundfile as sf
                    audio, sr = sf.read(temp_path)
                    Path(temp_path).unlink()
                    return audio.astype(np.float32)
                except ImportError:
                    Path(temp_path).unlink()
                    return None

            except Exception as e:
                logger.warning(f"System TTS failed: {e}")
                return None
        else:
            logger.warning(f"System TTS not implemented for {system}")
            return None

    def speak(self, text: str):
        """Speak text directly (blocking)"""
        import platform
        system = platform.system()

        if system == "Darwin":
            subprocess.run(["say", text], capture_output=True)
        elif system == "Linux":
            subprocess.run(["espeak", text], capture_output=True)


class AudioPipeline:
    """
    Main audio pipeline controller.
    """

    def __init__(
        self,
        stt_model: str,
        tts_model: str,
        vad_threshold: float,
        message_bus: MessageBus,
        shutdown_event: Event
    ):
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.vad_threshold = vad_threshold
        self.message_bus = message_bus
        self.shutdown_event = shutdown_event

        # State
        self.is_listening = False
        self.has_attention = False

        # Components (lazy loaded)
        self._audio_buffer = None
        self._vad = None
        self._stt = None
        self._tts = None

        # Message queue
        self._queue = None

    def run(self):
        """Main pipeline loop"""
        logger.info("Audio pipeline starting...")

        # Check dependencies
        if not NUMPY_AVAILABLE:
            logger.error("numpy required for audio pipeline")
            return

        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available - audio capture disabled")

        # Initialize components
        self._audio_buffer = AudioBuffer()
        self._vad = VoiceActivityDetector(self.vad_threshold)
        self._stt = STTEngine(self.stt_model)
        self._tts = TTSEngine(self.tts_model)

        # Register with message bus
        self._queue = self.message_bus.register("audio")

        # Start capture thread if sounddevice available
        if SOUNDDEVICE_AVAILABLE:
            capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            capture_thread.start()

        logger.info("Audio pipeline started")

        while not self.shutdown_event.is_set():
            try:
                self._process_messages()
                self._process_audio()
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Audio pipeline error: {e}")

        logger.info("Audio pipeline stopped")

    def _capture_loop(self):
        """Continuous audio capture"""
        try:
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                self._audio_buffer.write(indata.copy())

            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=1600
            ):
                while not self.shutdown_event.is_set():
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Audio capture error: {e}")

    def _process_messages(self):
        """Process incoming messages"""
        try:
            while True:
                msg_dict = self._queue.get_nowait()
                message = Message.from_dict(msg_dict)
                self._handle_message(message)
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Message error: {e}")

    def _handle_message(self, message: Message):
        """Handle a message"""
        payload = message.payload

        if message.type == MessageType.ATTENTION_GAINED:
            self.has_attention = True
            logger.info("Attention gained - voice active")

        elif message.type == MessageType.ATTENTION_LOST:
            self.has_attention = False
            logger.info("Attention lost - voice inactive")

        elif message.type == MessageType.SPEAK:
            text = payload.get("text")
            if text:
                self._speak(text)

        elif message.type == MessageType.MODEL_RESPONSE:
            # Speak response if we have attention
            if self.has_attention:
                text = payload.get("response")
                if text:
                    self._speak(text)

    def _process_audio(self):
        """Process captured audio"""
        if not self.has_attention or self._audio_buffer is None:
            return

        # Get recent audio
        audio_data = self._audio_buffer.get_recent(seconds=2)
        if audio_data is None or len(audio_data) == 0:
            return

        # Check for voice activity
        if not self._vad.is_speech(audio_data):
            if self.is_listening:
                # Speech ended - transcribe
                self._transcribe_and_send()
            return

        # Voice detected
        if not self.is_listening:
            logger.debug("Speech detected")
            self.is_listening = True
            self._audio_buffer.mark_speech_start()

    def _transcribe_and_send(self):
        """Transcribe buffered speech and send"""
        self.is_listening = False

        speech_audio = self._audio_buffer.get_speech_segment()
        if speech_audio is None or len(speech_audio) < 1600:
            return

        # Transcribe
        text = self._stt.transcribe(speech_audio)
        if text and len(text.strip()) > 0:
            logger.info(f"Transcribed: {text}")

            # Send to message bus
            self.message_bus.send(
                MessageType.USER_VOICE,
                source="audio_pipeline",
                payload={
                    "text": text,
                    "audio_duration": len(speech_audio) / 16000
                }
            )

    def _speak(self, text: str):
        """Speak text"""
        if self._tts:
            self._tts.speak(text)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Audio Pipeline Test")
    print("=" * 40)
    print(f"numpy available: {NUMPY_AVAILABLE}")
    print(f"sounddevice available: {SOUNDDEVICE_AVAILABLE}")
    print(f"whisper available: {WHISPER_AVAILABLE}")

    if NUMPY_AVAILABLE:
        # Test VAD
        vad = VoiceActivityDetector()
        print(f"\nVAD initialized")

        # Test TTS
        tts = TTSEngine()
        print("Testing TTS...")
        tts.speak("Hello, this is Senter speaking.")
        print("TTS test complete")

    print("\nTest complete")
