#!/usr/bin/env python3
"""
Senter Setup Script
Downloads required models from HuggingFace
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_model(repo_id: str, filename: str = None, local_dir: str = "Models", rename: str = None):
    """Download a model from HuggingFace"""
    print(f"Downloading from {repo_id}...")

    try:
        if filename:
            # Download specific file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            if rename:
                import shutil
                new_path = os.path.join(os.path.dirname(local_path), rename)
                shutil.move(local_path, new_path)
                local_path = new_path
            print(f"✅ Downloaded to {local_path}")
        else:
            # Download entire repo
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded repo to {local_path}")
        return local_path
    except Exception as e:
        print(f"❌ Failed to download from {repo_id}: {e}")
        return None

def main():
    """Main setup function"""
    print("🚀 Setting up Senter AI Assistant...")

    # Create Models directory
    models_dir = Path("Models")
    models_dir.mkdir(exist_ok=True)

    # Download models
    models = [
        {
            "repo": "Qwen/Qwen2.5-Omni-3B",
            "local_dir": "Models/Qwen2.5-Omni-3B"
        },
        {
            "repo": "bartowski/Qwen-Image-0.5B-GGUF",
            "filename": "Qwen-Image-0.5B-Q6_K.gguf",
            "local_dir": "Models",
            "rename": "Qwen_Image-Q6_K.gguf"
        },
        {
            "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
            "filename": "nomic-embed-text-v1.5.Q8_0.gguf",
            "local_dir": "Models",
            "rename": "nomic-embed-text.gguf"
        }
    ]

    for model in models:
        download_model(**model)

    print("🎉 Setup complete! Run 'python scripts/senter_app.py' to start Senter.")

if __name__ == "__main__":
    main()