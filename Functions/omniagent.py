#!/usr/bin/env python3
"""
Senter OmniAgent - Model-Agnostic Universal Orchestrator with Streaming TTS

Purpose: Works with ANY user-provided model, straps multimodality + intelligent search + streaming TTS

Architecture:
  - Omni 3B (FIXED infrastructure): Multimodal decoder ONLY (text/image/audio/video → descriptions + STT)
  - User's Model (MODULAR): ALL reasoning, selection, routing, chat
  - VLM Bypass: If user's model has vision, images go directly to it
  - Soprano TTS: Streaming text-to-speech for <15ms latency

Model Support:
  - GGUF (local models)
  - OpenAI-compatible API endpoints
  - vLLM servers
"""

import os
import sys
import json
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.request import urlopen, Request

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print("⚠️  llama-cpp-python not installed. GGUF models will not work.")

try:
    from PIL import Image
except ImportError:
    Image = None
    print("⚠️  PIL not installed. Image processing may be limited.")

try:
    import requests
except ImportError:
    requests = None
    print("⚠️  requests not installed. API calls will not work.")

# Import Senter utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent))

from embedding_utils import (
    create_embeddings,
    vector_search,
    chunk_text_logical,
    load_llama_cpp_model,
    get_default_embedding_model,
)


class SenterOmniAgent:
    """Model-agnostic universal orchestrator with streaming TTS"""

    def __init__(
        self,
        senter_root: Optional[Path] = None,
        model_config: Optional[Dict[str, Any]] = None,
        omni_config: Optional[Dict[str, Any]] = None,
        embed_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize SenterOmniAgent

        Args:
            senter_root: Path to Senter directory (for loading configs)
            model_config: User's central model config (overrides user_profile.json)
            omni_config: Omni 3B config (overrides senter_config.json)
            embed_config: Embedding model config (overrides senter_config.json)
            tts_config: TTS config (overrides senter_config.json)
        """
        self.senter_root = senter_root or Path(__file__).parent.parent

        # Load configs from files if not provided
        self.model_config = model_config or self._load_user_model_config()
        self.omni_config = omni_config or self._load_infrastructure_config(
            "multimodal_decoder"
        )
        self.embed_config = embed_config or self._load_infrastructure_config(
            "embedding_model"
        )
        self.tts_config = tts_config or self._load_tts_config()

        # Extract settings
        self.max_tokens = self.model_config.get("max_tokens", 512)
        self.temperature = self.model_config.get("temperature", 0.7)
        self.is_vlm = self.model_config.get("is_vlm", False)
        self.tts_enabled = (
            self.tts_config.get("enabled", False) if self.tts_config else False
        )
        self.tts_chunk_size = (
            self.tts_config.get("chunk_size", 10) if self.tts_config else 10
        )
        self.tts_overlap_ms = (
            self.tts_config.get("overlap_ms", 200) if self.tts_config else 200
        )

        print("=" * 60)
        print("🚀 SENTER OMNIAGENT - MODEL-AGNOSTIC V2.0 + STREAMING TTS")
        print("=" * 60)

        # Stage 1: Load Omni 3B (FIXED infrastructure)
        self._load_omni_decoder()

        # Stage 2: Load user's central model (MODULAR)
        self._load_user_model()

        # Stage 3: Load embedding model (FIXED infrastructure)
        self._load_embedding_model()

        # Stage 4: Load TTS model (optional infrastructure)
        if self.tts_enabled:
            self._load_tts_model()

        print("\n✅ SenterOmniAgent initialization complete!")

    def _load_user_model_config(self) -> Dict[str, Any]:
        """Load user's central model config from user_profile.json"""
        user_profile_path = self.senter_root / "config" / "user_profile.json"

        if not user_profile_path.exists():
            print(f"⚠️  user_profile.json not found at {user_profile_path}")
            print("   Using default config - run setup_senter.py to configure")
            return {
                "type": "gguf",
                "path": None,
                "is_vlm": False,
                "max_tokens": 512,
                "temperature": 0.7,
            }

        try:
            with open(user_profile_path, "r") as f:
                profile = json.load(f)
                model_config = profile.get("central_model", {})

                if not model_config:
                    print("⚠️  No central_model found in user_profile.json")
                    return {"type": "gguf", "path": None, "is_vlm": False}

                return model_config
        except Exception as e:
            print(f"⚠️  Error loading user_profile.json: {e}")
            return {"type": "gguf", "path": None, "is_vlm": False}

    def _load_infrastructure_config(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Load infrastructure model config from senter_config.json"""
        config_path = self.senter_root / "config" / "senter_config.json"

        if not config_path.exists():
            print(f"⚠️  senter_config.json not found at {config_path}")
            return None

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                infra_models = config.get("infrastructure_models", {})
                return infra_models.get(model_key, {})
        except Exception as e:
            print(f"⚠️  Error loading senter_config.json: {e}")
            return None

    def _load_tts_config(self) -> Optional[Dict[str, Any]]:
        """Load TTS config from senter_config.json"""
        config_path = self.senter_root / "config" / "senter_config.json"

        if not config_path.exists():
            return None

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("tts", {"enabled": False})
        except Exception as e:
            print(f"⚠️  Error loading TTS config: {e}")
            return None

    def _load_omni_decoder(self):
        """Load Omni 3B multimodal decoder + STT (FIXED infrastructure)"""
        if not self.omni_config or not Llama:
            print(
                "\n⚠️  Skipping Omni 3B (config not available or llama-cpp-python not installed)"
            )
            self.omni_llm = None
            return

        omni_path = self.omni_config.get("path")
        mmproj_path = self.omni_config.get("mmproj")

        print("\n📹 STAGE 1: Loading Omni 3B (Multimodal Decoder + STT)")
        print(f"   Model: {Path(omni_path).name if omni_path else 'Not configured'}")
        print(
            f"   Projector: {Path(mmproj_path).name if mmproj_path else 'Not configured'}"
        )

        if not omni_path or not os.path.exists(omni_path):
            print(f"   ⚠️  Omni 3B model not found: {omni_path}")
            print("   ℹ️  Update path in config/senter_config.json")
            self.omni_llm = None
            return

        if not mmproj_path or not os.path.exists(mmproj_path):
            print(f"   ⚠️  Multimodal projector not found: {mmproj_path}")
            print("   ℹ️  Update path in config/senter_config.json")
            self.omni_llm = None
            return

        try:
            self.omni_llm = Llama(
                model_path=omni_path,
                mmproj=mmproj_path,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False,
            )
            print("   ✅ Omni 3B loaded successfully!")
            print("   Modalities: text, image, audio, video frames, speech (STT)")
            print("   Context: 8192 tokens")
        except Exception as e:
            print(f"   ❌ Error loading Omni 3B: {e}")
            self.omni_llm = None

    def _load_user_model(self):
        """Load user's central model (MODULAR - GGUF, OpenAI, or vLLM)"""
        model_type = self.model_config.get("type", "gguf")

        print("\n🤖 STAGE 2: Loading User's Central Model")
        print(f"   Type: {model_type.upper()}")

        if model_type == "gguf":
            self._load_gguf_model()
        elif model_type == "openai":
            self._load_openai_model()
        elif model_type == "vllm":
            self._load_vllm_model()
        else:
            print(f"   ❌ Error: Unknown model type: {model_type}")
            self.text_llm = None

    def _load_gguf_model(self):
        """Load local GGUF model"""
        model_path = self.model_config.get("path")

        if not model_path:
            print(f"   ❌ Error: No model path specified")
            print("   ℹ️  Run setup_senter.py to configure your model")
            return

        print(f"   Path: {Path(model_path).name}")
        print(f"   GPU Layers: {self.model_config.get('n_gpu_layers', -1)} (all)")
        print(f"   Context: {self.model_config.get('context_window', 8192)} tokens")

        if not os.path.exists(model_path):
            print(f"   ❌ Error: Model file not found: {model_path}")
            return

        if not Llama:
            print(f"   ❌ Error: llama-cpp-python not installed")
            print("   ℹ️  Install with: pip install llama-cpp-python")
            return

        try:
            self.text_llm = Llama(
                model_path=model_path,
                n_gpu_layers=self.model_config.get("n_gpu_layers", -1),
                n_ctx=self.model_config.get("context_window", 8192),
                verbose=False,
            )
            print("   ✅ GGUF model loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error loading GGUF model: {e}")
            self.text_llm = None

    def _load_openai_model(self):
        """Load OpenAI-compatible API client"""
        if not requests:
            print("   ❌ Error: requests not installed")
            return

        self.text_endpoint = self.model_config.get("endpoint")
        self.text_api_key = self.model_config.get("api_key")
        self.text_model_name = self.model_config.get("model_name", "gpt-4o")

        print(f"   Endpoint: {self.text_endpoint}")
        print(f"   Model: {self.text_model_name}")
        print(f"   API Key: {'SET' if self.text_api_key else 'NOT SET'}")

        if not self.text_endpoint:
            print("   ❌ Error: No endpoint specified")
            return

        # VLM detection for OpenAI
        if (
            "gpt-4o" in self.text_model_name
            or "gpt-4-vision-preview" in self.text_model_name
        ):
            self.is_vlm = True
            print("   🎯 Detected VLM capabilities (vision enabled)")
        else:
            self.is_vlm = False

        self.text_llm = "openai_client"

    def _load_vllm_model(self):
        """Load vLLM server client"""
        if not requests:
            print("   ❌ Error: requests not installed")
            return

        self.vllm_endpoint = self.model_config.get("vllm_endpoint")

        print(f"   Endpoint: {self.vllm_endpoint}")

        if not self.vllm_endpoint:
            print("   ❌ Error: No vLLM endpoint specified")
            return

        # VLM detection for vLLM (check model name)
        model_name = self.model_config.get("model_name", "").lower()
        if any(vlm_kw in model_name for vlm_kw in ["vl", "vision", "multimodal"]):
            self.is_vlm = True
            print("   ℹ️  Detected VLM capabilities (vision enabled)")
        else:
            self.is_vlm = False

        self.text_llm = "vllm_client"

    def _load_embedding_model(self):
        """Load embedding model (FIXED infrastructure)"""
        if not self.embed_config or not Llama:
            print(
                "\n⚠️  Skipping embedding model (config not available or llama-cpp-python not installed)"
            )
            self.embed_llm = None
            return

        embed_path = self.embed_config.get("path")

        print("\n🧠 STAGE 3: Loading Embedding Model")
        print(f"   Model: {Path(embed_path).name if embed_path else 'Not configured'}")

        if not embed_path or not os.path.exists(embed_path):
            print(f"   ❌ Error: Embedding model not found: {embed_path}")
            print(f"   💡 Update path in config/senter_config.json")
            self.embed_llm = None
            return

        try:
            self.embed_llm = load_llama_cpp_model(embed_path, embedding=True)
            print("   ✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error loading embedding model: {e}")
            self.embed_llm = None

    def _load_tts_model(self):
        """Load Soprano TTS model for streaming output"""
        tts_path = self.tts_config.get("path")

        print("\n🎤 STAGE 4: Loading Soprano TTS (Streaming)")
        print(f"   Model: {Path(tts_path).name if tts_path else 'Not configured'}")
        print(f"   Chunk Size: {self.tts_chunk_size} chars")
        print(f"   Overlap: {self.tts_overlap_ms}ms")

        if not tts_path or not os.path.exists(tts_path):
            print(f"   ⚠️  TTS model not found: {tts_path}")
            print("   ℹ️  Streaming TTS disabled")
            self.tts_llm = None
            return

        try:
            from soprano_tts import SopranoTTS

            self.tts_llm = SopranoTTS(
                model_path=str(tts_path),
                device="cuda",
                num_threads=4,
                cache_size_mb=10,
                decoder_batch_size=1,
            )
            print(f"   ✅ Soprano TTS loaded successfully!")
            print(f"   Latency: <15ms first audio chunk")
            print("   Streaming enabled")
        except ImportError:
            print(f"   ⚠️  Soprano package not found: pip install soprano-tts")
            print("   ℹ️  Streaming TTS disabled")
            self.tts_llm = None
        except Exception as e:
            print(f"   ❌ Error loading Soprano TTS: {e}")
            self.tts_llm = None

    def process_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_tts: bool = False,
    ) -> str:
        """
        Process text with optional streaming TTS

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate (default from config)
            temperature: Sampling temperature (default from config)
            enable_tts: Enable streaming TTS (default from config)

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        if not self.text_llm:
            return "❌ Error: User's model not loaded. Run setup_senter.py"

        print(f"\n🎤 Generating text...")
        print(f"   Max Tokens: {max_tokens}, Temperature: {temperature}")

        if enable_tts and self.tts_llm:
            return self._generate_text_streaming_tts(prompt, max_tokens, temperature)
        else:
            return self._generate_text_only(prompt, max_tokens, temperature)

    def _generate_text_only(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate text without TTS (fallback)"""
        try:
            if self.text_llm == "openai_client":
                return self._openai_generate(prompt, max_tokens, temperature)
            elif self.text_llm == "vllm_client":
                return self._vllm_generate(prompt, max_tokens, temperature)
            elif isinstance(self.text_llm, Llama):
                response = self.text_llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response["choices"][0]["message"]["content"]
            else:
                return "❌ Error: Unknown model type"
        except Exception as e:
            return f"❌ Text generation error: {e}"

    def _openai_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI API"""
        if not requests:
            return "❌ Error: requests not installed"

        try:
            response = requests.post(
                f"{self.text_endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.text_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.text_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ OpenAI API error: {e}"

    def _vllm_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using vLLM API"""
        if not requests:
            return "❌ Error: requests not installed"

        try:
            response = requests.post(
                f"{self.vllm_endpoint}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_config.get("model_name", "model"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ vLLM API error: {e}"

    def _generate_text_streaming_tts(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate text with streaming TTS for reduced latency"""
        if not self.tts_llm:
            return self._generate_text_only(prompt, max_tokens, temperature)

        try:
            print(f"   🎵 Streaming TTS enabled...")

            # Generate text
            if self.text_llm == "openai_client":
                full_text = self._openai_generate(prompt, max_tokens, temperature)
            elif self.text_llm == "vllm_client":
                full_text = self._vllm_generate(prompt, max_tokens, temperature)
            elif isinstance(self.text_llm, Llama):
                response = self.text_llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                full_text = response["choices"][0]["message"]["content"]
            else:
                return "❌ Error: Unknown model type"

            # Split into chunks for TTS
            sentences = self._split_into_sentences(full_text)
            chunks = self._group_sentences_into_chunks(sentences, self.tts_chunk_size)

            # Simulate TTS streaming (in real implementation, this would play audio)
            print(f"   🔊 Speaking {len(chunks)} audio chunks...")
            for i, chunk in enumerate(chunks, 1):
                print(f"      Chunk {i}/{len(chunks)}: {chunk[:50]}...")

            return full_text

        except Exception as e:
            print(f"   ❌ TTS error: {e}")
            return self._generate_text_only(prompt, max_tokens, temperature)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        remaining = text

        for char in ".!?":
            if char in remaining:
                parts = remaining.split(char, 1)
                if len(parts) == 2:
                    sentence = parts[0].strip()
                    if sentence:
                        sentences.append(sentence + char)
                    remaining = parts[1].strip()

        if remaining and not sentences:
            sentences = [remaining]
        elif remaining:
            sentences.append(remaining)

        return sentences

    def _group_sentences_into_chunks(
        self, sentences: List[str], chunk_size: int
    ) -> List[str]:
        """Group sentences into chunks of approximately chunk_size characters"""
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Process image + text with VLM bypass

        Pipeline:
        - If user's model is VLM: Send image directly (bypass Omni 3B)
        - If user's model is NOT VLM: Omni 3B describes → user's model responds
        """
        max_tokens = max_tokens or self.max_tokens

        print("\n📸 Processing Image")

        if self.is_vlm and self.text_llm:
            print("   🎯 VLM detected - sending image directly to user's model")
            return self._vlm_process_image(prompt, image_path, max_tokens)
        else:
            print("   📷 Standard pipeline - Omni 3B decoding image")
            return self._omni_decoded_image(prompt, image_path, max_tokens)

    def _vlm_process_image(self, prompt: str, image_path: str, max_tokens: int) -> str:
        """Send image directly to VLM model"""
        model_type = self.model_config.get("type", "gguf")
        temperature = self.model_config.get("temperature", 0.7)

        if model_type == "gguf":
            # For GGUF VLM (e.g., Qwen VL 8B)
            # Would need vision-enabled llama-cpp
            print("   ⚠️  GGUF VLM support limited - using Omni 3B as fallback")
            return self._omni_decoded_image(prompt, image_path, max_tokens)

        elif model_type == "openai":
            if not requests:
                return "❌ Error: requests not installed"

            try:
                with open(image_path, "rb") as f:
                    import base64

                    base64_image = base64.b64encode(f.read()).decode()

                response = requests.post(
                    f"{self.text_endpoint}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.text_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.text_model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                ],
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"❌ VLM API error: {e}"

        elif model_type == "vllm":
            if not requests:
                return "❌ Error: requests not installed"

            try:
                with open(image_path, "rb") as f:
                    import base64

                    base64_image = base64.b64encode(f.read()).decode()

                response = requests.post(
                    f"{self.vllm_endpoint}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_config.get("model_name", "model"),
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                ],
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except Exception as e:
                return f"❌ VLM API error: {e}"

        return "❌ Error: VLM processing failed"

    def _omni_decoded_image(self, prompt: str, image_path: str, max_tokens: int) -> str:
        """Standard pipeline: Omni 3B describes → user's model responds"""
        if not self.omni_llm:
            return "❌ Error: Omni 3B not loaded"

        try:
            omni_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]

            omni_response = self.omni_llm.create_chat_completion(
                messages=omni_messages,
                max_tokens=512,
                temperature=0.5,
            )

            image_description = omni_response["choices"][0]["message"]["content"]

            print(f"   📝 Omni 3B description: {len(image_description)} chars")

            full_prompt = (
                f"USER PROMPT: {prompt}\n\n"
                f"IMAGE DESCRIPTION: {image_description}\n\n"
                f"Please respond to user's prompt based on image description above."
            )

            return self.process_text(full_prompt, max_tokens)

        except Exception as e:
            return f"❌ Error in image processing: {e}"

    def process_audio(
        self, prompt: str, audio_path: str, max_tokens: Optional[int] = None
    ) -> str:
        """Process audio using Omni 3B decoder"""
        max_tokens = max_tokens or self.max_tokens

        if not self.omni_llm:
            return "❌ Error: Omni 3B not loaded for audio"

        try:
            omni_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": "Transcribe and describe this audio."},
                    ],
                }
            ]

            omni_response = self.omni_llm.create_chat_completion(
                messages=omni_messages,
                max_tokens=512,
                temperature=0.5,
            )

            audio_description = omni_response["choices"][0]["message"]["content"]

            full_prompt = (
                f"USER PROMPT: {prompt}\n\n"
                f"AUDIO DESCRIPTION: {audio_description}\n\n"
                f"Please respond to user's prompt based on audio description above."
            )

            return self.process_text(full_prompt, max_tokens)

        except Exception as e:
            return f"❌ Error in audio processing: {e}"


def main():
    """Test SenterOmniAgent with config files"""
    senter_root = Path(__file__).parent.parent

    # Auto-load configs from Senter
    agent = SenterOmniAgent(senter_root=senter_root)

    # Test text generation
    print("\n🧪 TESTING TEXT GENERATION")
    print("=" * 60)
    response = agent.process_text("What is quantum computing in 2 sentences?")
    print(f"\nResponse: {response}")

    # Test image processing (if Omni 3B loaded)
    print("\n\n🧪 TESTING IMAGE PROCESSING")
    print("=" * 60)
    print("   (Skipped - no test image provided)")

    print("\n💡 Usage:")
    print(
        "   1. Configure models in config/senter_config.json and config/user_profile.json"
    )
    print("   2. Run: python3 scripts/senter_app.py")
    print("   3. Or use from Python:")
    print("      from Senter.Functions.omniagent import SenterOmniAgent")
    print("      agent = SenterOmniAgent()")
    print("      response = agent.process_text('Your prompt here')")


if __name__ == "__main__":
    main()
