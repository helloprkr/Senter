#!/usr/bin/env python3
"""
Senter Music Generator - Unified Music Generation Pipeline
Combines compose_music.py (ACE-Step integration) with Jammit's playback/queuing system

Features:
- Single track generation
- Infinite radio mode (continuous generation + playback)
- Silent operation (no console output)
- Graceful shutdown
"""

import argparse
import os
import sys
import time
import subprocess
import threading
import warnings
from collections import deque
from pathlib import Path
from typing import Optional, List

# Add ACE-Step to path BEFORE any imports to fix bus error
project_root = Path(__file__).parent.parent.parent.absolute()
ace_step_path = project_root / "Resources" / "ACE-Step"
if str(ace_step_path) not in sys.path:
    sys.path.insert(0, str(ace_step_path))

warnings.filterwarnings("ignore")

# Uncomment to see debug output
# Suppress all output
# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.devnull, "w")

# Restore stdout/stderr at the end
import atexit


def restore_output():
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


atexit.register(restore_output)

# Import ACE-Step composition
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Functions.compose_music import compose_music


class MusicGenerator:
    """Unified music generator with playback capabilities"""

    def __init__(self):
        self.playback_queue = deque()
        self.current_playing_process = None
        self.playback_thread = None
        self.infinite_mode_prompt = None
        self.generation_lock = threading.Lock()
        self._should_terminate = False
        self._duration_seconds = 213
        self._infer_steps = 60
        self._guidance_scale = 15.0

    def _play_audio(self, audio_path):
        """Play audio file using aplay"""
        if self._should_terminate:
            return

        if self.current_playing_process:
            try:
                self.current_playing_process.terminate()
                self.current_playing_process.wait()
            except Exception:
                pass

        try:
            self.current_playing_process = subprocess.Popen(
                ["aplay", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Trigger next generation in infinite mode
            if self.infinite_mode_prompt and not self.generation_lock.locked():
                threading.Thread(target=self._trigger_next_generation).start()

            while (
                self.current_playing_process.poll() is None
                and not self._should_terminate
            ):
                time.sleep(0.1)

            if (
                self.current_playing_process
                and self.current_playing_process.poll() is None
            ):
                self.current_playing_process.terminate()
                self.current_playing_process.wait()
            self.current_playing_process = None
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _playback_manager(self):
        """Background thread that manages playback queue"""
        while not self._should_terminate:
            if self.playback_queue:
                audio_path = self.playback_queue.popleft()
                self._play_audio(audio_path)
            else:
                time.sleep(1)

    def _perform_generation(self, prompt):
        """Generate music in background thread"""
        if self._should_terminate:
            return

        with self.generation_lock:
            if self.infinite_mode_prompt:
                generated_paths = compose_music(
                    prompt=prompt,
                    duration_seconds=self._duration_seconds,
                    infer_steps=self._infer_steps,
                    guidance_scale=self._guidance_scale,
                    n_gen=1,
                )

                if generated_paths:
                    self.playback_queue.extend(generated_paths)

    def _trigger_next_generation(self):
        """Trigger next generation without blocking"""
        if self.infinite_mode_prompt:
            generation_thread = threading.Thread(
                target=self._perform_generation, args=(self.infinite_mode_prompt,)
            )
            generation_thread.daemon = True
            generation_thread.start()

    def generate_single(
        self, prompt, duration_seconds=213, infer_steps=60, guidance_scale=15.0
    ):
        """Generate and play a single track"""
        self._duration_seconds = duration_seconds
        self._infer_steps = infer_steps
        self._guidance_scale = guidance_scale

        generated_paths = compose_music(
            prompt=prompt,
            duration_seconds=duration_seconds,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            n_gen=1,
        )

        if generated_paths:
            audio_path = generated_paths[0]
            self._play_audio(audio_path)

    def generate_infinite(
        self, prompt, duration_seconds=213, infer_steps=60, guidance_scale=15.0
    ):
        """Start infinite radio mode"""
        self._duration_seconds = duration_seconds
        self._infer_steps = infer_steps
        self._guidance_scale = guidance_scale
        self.infinite_mode_prompt = prompt

        # Start playback manager thread
        self.playback_thread = threading.Thread(target=self._playback_manager)
        self.playback_thread.daemon = True
        self.playback_thread.start()

        # Trigger first generation
        self._trigger_next_generation()

        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._should_terminate = True
        finally:
            if (
                self.current_playing_process
                and self.current_playing_process.poll() is None
            ):
                self.current_playing_process.terminate()
                self.current_playing_process.wait()

    def stop(self):
        """Stop all generation and playback"""
        self._should_terminate = True


def main():
    parser = argparse.ArgumentParser(
        description="Senter Music Generator - Generate music with ACE-Step"
    )
    parser.add_argument("prompt", type=str, help="Description of music to generate")
    parser.add_argument("--lyrics", type=str, default="", help="Lyrics for the song")
    parser.add_argument(
        "--instrumental", action="store_true", help="Force instrumental generation"
    )
    parser.add_argument(
        "--duration", type=int, default=213, help="Duration in seconds (default: 213)"
    )
    parser.add_argument(
        "--infer-steps", type=int, default=60, help="Inference steps (default: 60)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=15.0,
        help="Guidance scale (default: 15.0)",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Enable infinite radio mode (continuous generation)",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=1,
        help="Number of tracks to generate (single mode only)",
    )

    args = parser.parse_args()

    generator = MusicGenerator()

    try:
        if args.infinite:
            print(f"🎵 Starting infinite radio: {args.prompt}")
            generator.generate_infinite(
                prompt=args.prompt,
                duration_seconds=args.duration,
                infer_steps=args.infer_steps,
                guidance_scale=args.guidance_scale,
            )
        else:
            print(f"🎵 Generating {args.n_gen} track(s): {args.prompt}")
            for i in range(args.n_gen):
                generator.generate_single(
                    prompt=args.prompt,
                    duration_seconds=args.duration,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.guidance_scale,
                )
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
        generator.stop()
        print("✅ Stopped")


if __name__ == "__main__":
    main()
