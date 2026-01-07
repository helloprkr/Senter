#!/usr/bin/env python3
"""
Async Wrapper for SenterOmniAgent
Enables async/await pattern for parallel omniagent calls
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from pathlib import Path

# Import Senter utilities
import sys

sys.path.insert(0, str(Path(__file__).parent))

from omniagent import SenterOmniAgent
from senter_md_parser import SenterMdParser


class OmniAgentAsync:
    """Async wrapper for SenterOmniAgent"""

    def __init__(
        self,
        senter_root: Optional[Path] = None,
        focus_name: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        omni_config: Optional[Dict[str, Any]] = None,
        embed_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AsyncOmniAgent

        Args:
            senter_root: Path to Senter directory
            focus_name: Name of Focus (loads config from SENTER.md)
            model_config: User's central model config (overrides SENTER.md)
            omni_config: Omni 3B config
            embed_config: Embedding model config
            tts_config: TTS config
        """
        self.senter_root = senter_root or Path(__file__).parent.parent
        self.focus_name = focus_name
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Load config from SENTER.md if focus_name specified
        if focus_name:
            parser = SenterMdParser(self.senter_root)
            focus_config = parser.load_focus_config(focus_name)

            # Use focus-specific model config if not overridden
            if not model_config:
                model_config = focus_config.get("model", {})

        # Initialize SenterOmniAgent
        self.omniagent = SenterOmniAgent(
            model_config=model_config or {},
            omni_config=omni_config or {},
            embed_config=embed_config or {},
            tts_config=tts_config or {},
        )

        # Store system prompt from SENTER.md
        self.system_prompt = ""
        if focus_name:
            parser = SenterMdParser(self.senter_root)
            self.system_prompt = parser.get_system_prompt(focus_name)

        print(f"✅ OmniAgentAsync initialized for: {focus_name or 'custom config'}")

    async def process_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_tts: bool = False,
    ) -> str:
        """Async wrapper for process_text()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.omniagent.process_text,
            prompt,
            max_tokens,
            temperature,
            enable_tts,
        )

    async def process_image(
        self,
        prompt: str,
        image_path: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async wrapper for process_image()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.omniagent.process_image,
            prompt,
            image_path,
            max_tokens,
            temperature,
        )

    async def process_audio(
        self,
        prompt: str,
        audio_path: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async wrapper for process_audio()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.omniagent.process_audio,
            prompt,
            audio_path,
            max_tokens,
            temperature,
        )

    async def process_video_from_url(
        self,
        prompt: str,
        video_url: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async wrapper for process_video_from_url()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.omniagent.process_video_from_url,
            prompt,
            video_url,
            max_tokens,
            temperature,
        )

    async def vector_search(
        self,
        query: str,
        documents: list,
        num_results: int = 4,
    ) -> tuple[list, list]:
        """Async wrapper for vector search via omniagent"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_vector_search,
            query,
            documents,
            num_results,
        )

    def _run_vector_search(
        self,
        query: str,
        documents: list,
        num_results: int,
    ) -> tuple[list, list]:
        """Run vector search (sync, wrapped for async)"""
        from embedding_utils import (
            create_embeddings,
            vector_search,
            get_default_embedding_model,
        )

        embed_model = get_default_embedding_model()
        if not embed_model:
            return documents[:num_results], [1.0] * num_results

        embeddings = create_embeddings(documents, embed_model)
        return vector_search(
            query, embeddings, documents, top_k=num_results, model_path=embed_model
        )

    def get_system_prompt(self) -> str:
        """Get system prompt for this agent"""
        return self.system_prompt

    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        print(f"✅ OmniAgentAsync closed for: {self.focus_name}")


async def main():
    """Test async omniagent"""
    print("Testing OmniAgentAsync...")

    async_agent = OmniAgentAsync(
        senter_root=Path(__file__).parent.parent,
        focus_name="general",
    )

    # Test async text processing
    print("\n📤 Processing text...")
    response = await async_agent.process_text("What is AI?")
    print(f"✅ Response: {response[:100]}...")

    # Test parallel calls
    print("\n🔄 Testing parallel calls...")
    tasks = [async_agent.process_text(f"Question {i}") for i in range(3)]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"  Response {i + 1}: {result[:50]}...")

    await async_agent.close()


if __name__ == "__main__":
    asyncio.run(main())
