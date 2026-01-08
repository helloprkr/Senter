"""
Voice interface using local Whisper model.

No wake word needed - activated by gaze detection or push-to-talk.
"""

from __future__ import annotations
import asyncio
import tempfile
import wave
from pathlib import Path
from typing import Callable

import numpy as np


class VoiceInterface:
    """
    Local speech-to-text using Whisper.

    No wake word needed - activated by gaze detection.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.is_listening = False
        self.audio_buffer = []

        # Voice activity detection settings
        self.vad_threshold = 0.02  # Energy threshold
        self.silence_duration = 1.5  # Seconds of silence to end utterance
        self.sample_rate = 16000

    def load(self) -> None:
        """Load Whisper model."""
        try:
            import whisper

            self.model = whisper.load_model(self.model_size)
            print(f"[VOICE] Whisper {self.model_size} loaded")
        except ImportError:
            print("[VOICE] Whisper not available. Install: pip install openai-whisper")
            print("[VOICE] Voice interface will be disabled")

    def is_available(self) -> bool:
        """Check if voice interface is available."""
        return self.model is not None

    async def start_listening(
        self, on_transcript: Callable[[str], None]
    ) -> None:
        """
        Start listening for speech.

        Uses voice activity detection to determine when user is speaking.
        """
        if not self.model:
            print("[VOICE] Model not loaded, cannot start listening")
            return

        try:
            import sounddevice as sd
        except ImportError:
            print("[VOICE] sounddevice not available. Install: pip install sounddevice")
            return

        self.is_listening = True
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(self.sample_rate * chunk_duration)

        silence_chunks = 0
        max_silence_chunks = int(self.silence_duration / chunk_duration)
        recording = False
        audio_data = []

        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_chunks, recording, audio_data

            if not self.is_listening:
                return

            # Calculate energy
            energy = np.sqrt(np.mean(indata ** 2))

            if energy > self.vad_threshold:
                # Voice detected
                if not recording:
                    print("[VOICE] Recording started...")
                recording = True
                silence_chunks = 0
                audio_data.append(indata.copy())
            elif recording:
                # Silence during recording
                silence_chunks += 1
                audio_data.append(indata.copy())

                if silence_chunks >= max_silence_chunks:
                    # End of utterance
                    recording = False
                    if audio_data:
                        # Process in background
                        audio_copy = np.concatenate(audio_data)
                        asyncio.create_task(
                            self._process_audio(audio_copy, on_transcript)
                        )
                    audio_data = []
                    silence_chunks = 0

        # Start audio stream
        try:
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                callback=audio_callback,
            )

            with stream:
                print("[VOICE] Listening...")
                while self.is_listening:
                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"[VOICE] Error starting audio stream: {e}")

    def stop_listening(self) -> None:
        """Stop listening."""
        self.is_listening = False
        print("[VOICE] Stopped listening")

    async def _process_audio(
        self,
        audio: np.ndarray,
        callback: Callable[[str], None],
    ) -> None:
        """Process recorded audio through Whisper."""
        if self.model is None:
            return

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

            # Write WAV
            with wave.open(f.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes((audio * 32767).astype(np.int16).tobytes())

        try:
            # Transcribe (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(temp_path, language="en", fp16=False),
            )

            text = result["text"].strip()
            if text:
                print(f"[VOICE] Heard: {text}")
                callback(text)

        except Exception as e:
            print(f"[VOICE] Transcription error: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file."""
        if not self.model:
            return ""

        result = self.model.transcribe(str(audio_path), language="en")
        return result["text"].strip()

    async def record_and_transcribe(self, duration: float = 5.0) -> str:
        """
        Record audio for a fixed duration and transcribe.

        Useful for push-to-talk mode.
        """
        if not self.model:
            return ""

        try:
            import sounddevice as sd

            print(f"[VOICE] Recording for {duration}s...")

            # Record
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()

            print("[VOICE] Processing...")

            # Save and transcribe
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

                with wave.open(f.name, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.sample_rate)
                    wav.writeframes(
                        (audio.flatten() * 32767).astype(np.int16).tobytes()
                    )

            try:
                result = self.model.transcribe(temp_path, language="en", fp16=False)
                return result["text"].strip()
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except ImportError:
            print("[VOICE] sounddevice not available")
            return ""
        except Exception as e:
            print(f"[VOICE] Recording error: {e}")
            return ""
