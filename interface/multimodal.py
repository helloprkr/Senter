"""
Integrated multimodal interface.

Gaze + Voice = Seamless interaction without wake words.
"""

from __future__ import annotations
import asyncio
from typing import Callable, Optional

from .gaze import GazeDetector
from .voice import VoiceInterface


class MultimodalInterface:
    """
    Combines gaze detection + voice input.

    Workflow:
    1. User looks at camera
    2. After 0.5s of sustained gaze -> start listening
    3. User speaks
    4. User looks away -> stop listening, process
    """

    def __init__(
        self,
        whisper_model: str = "base",
        camera_id: int = 0,
    ):
        self.gaze = GazeDetector()
        self.voice = VoiceInterface(whisper_model)
        self.camera_id = camera_id

        self._on_input: Optional[Callable[[str], None]] = None
        self._voice_task: Optional[asyncio.Task] = None
        self.running = False

    def load(self) -> bool:
        """
        Load all required models.

        Returns True if at least voice is available.
        """
        self.gaze.load()
        self.voice.load()

        if not self.voice.is_available():
            print("[MULTIMODAL] Voice not available - interface disabled")
            return False

        if not self.gaze.is_available():
            print("[MULTIMODAL] Gaze not available - using voice-only mode")

        return True

    def is_available(self) -> bool:
        """Check if multimodal interface is available."""
        return self.voice.is_available()

    async def start(self, on_input: Callable[[str], None]) -> None:
        """
        Start the multimodal interface.

        Args:
            on_input: Callback when user input is transcribed
        """
        self._on_input = on_input
        self.running = True

        if self.gaze.is_available():
            # Full multimodal: gaze activates voice
            print("[MULTIMODAL] Starting gaze-activated voice mode")
            await self._start_gaze_voice()
        else:
            # Voice-only: continuous listening with VAD
            print("[MULTIMODAL] Starting voice-only mode")
            await self._start_voice_only()

    async def _start_gaze_voice(self) -> None:
        """Start gaze-activated voice input."""

        def on_gaze_start():
            """Called when user starts looking at camera."""
            print("[MULTIMODAL] Gaze detected - starting voice input")
            if self._voice_task is None or self._voice_task.done():
                self._voice_task = asyncio.create_task(
                    self.voice.start_listening(self._on_transcript)
                )

        def on_gaze_end():
            """Called when user looks away."""
            print("[MULTIMODAL] Gaze ended - stopping voice input")
            self.voice.stop_listening()

        await self.gaze.start(on_gaze_start, on_gaze_end, self.camera_id)

    async def _start_voice_only(self) -> None:
        """Start voice-only continuous listening."""
        await self.voice.start_listening(self._on_transcript)

    def _on_transcript(self, text: str) -> None:
        """Called when speech is transcribed."""
        if self._on_input and text:
            self._on_input(text)

    def stop(self) -> None:
        """Stop the multimodal interface."""
        self.running = False
        self.gaze.stop()
        self.voice.stop_listening()
        print("[MULTIMODAL] Stopped")

    async def push_to_talk(self, duration: float = 5.0) -> str:
        """
        Push-to-talk mode: record for fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Transcribed text
        """
        return await self.voice.record_and_transcribe(duration)


class SimplifiedVoiceInput:
    """
    Simplified voice input for CLI integration.

    Just press Enter to start recording, speak, and get transcription.
    """

    def __init__(self, model_size: str = "base"):
        self.voice = VoiceInterface(model_size)
        self._loaded = False

    def load(self) -> bool:
        """Load voice model."""
        self.voice.load()
        self._loaded = self.voice.is_available()
        return self._loaded

    async def record_input(self, duration: float = 5.0) -> str:
        """
        Record and transcribe voice input.

        Args:
            duration: Max recording duration

        Returns:
            Transcribed text or empty string
        """
        if not self._loaded:
            if not self.load():
                return ""

        return await self.voice.record_and_transcribe(duration)

    def is_available(self) -> bool:
        """Check if voice input is available."""
        return self._loaded


# Convenience function for quick voice input
async def get_voice_input(
    model_size: str = "base", duration: float = 5.0
) -> str:
    """
    Quick one-shot voice input.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        duration: Recording duration in seconds

    Returns:
        Transcribed text
    """
    voice = SimplifiedVoiceInput(model_size)
    if voice.load():
        return await voice.record_input(duration)
    return ""
