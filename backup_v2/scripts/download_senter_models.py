#!/usr/bin/env python3
"""
Download Senter Models - Fetches and organizes models for Senter

Downloads:
- Qwen 3B Omni (multimodal decoder) - FIXED INFRASTRUCTURE
- Nomic Embed (embedding model) - FIXED INFRASTRUCTURE
- Soprano 70M TTS (streaming TTS) - OPTIONAL
- Hermes 3B (recommended text model) - OPTIONAL
- Qwen VL 8B (recommended VLM) - OPTIONAL
"""

import sys
import argparse
from pathlib import Path


def download_soprano_tts(model_dir: Path) -> bool:
    """Download Soprano TTS models using huggingface-cli"""
    print("\n🎤 Downloading Soprano TTS Models")
    print("=" * 60)

    models = [
        {
            "name": "Soprano 70M",
            "file": "soprano-70M-Q8_0.gguf",
            "url": "https://huggingface.co/ekwek/Soprano-70M-GGUF/resolve/main/soprano-70M-Q8_0.gguf",
            "description": "70M parameters, ultra-fast TTS",
        },
        {
            "name": "Soprano 80M",
            "file": "soprano-80M-Q8_0.gguf",
            "url": "https://huggingface.co/ekwek/Soprano-80M-GGUF/resolve/main/soprano-80M-Q8_0.gguf",
            "description": "80M parameters, ultra-fast TTS",
        },
    ]

    for model in models:
        model_file = model_dir / model["file"]

        if model_file.exists():
            print(f"   ✅ {model['name']} already exists: {model_file}")
            continue

        print(f"   📥 Downloading {model['name']}...")
        print(
            f"      Size: ~{model['description'].split(',')[0] if 'M' in model['description'] else 'unknown'}"
        )

        try:
            import subprocess

            result = subprocess.run(
                [
                    "huggingface-cli",
                    "download",
                    model["url"],
                    "--local-dir",
                    str(model_dir),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                print(f"   ✅ Downloaded: {model_file.name}")
            else:
                print(f"   ⚠️  Download failed with code {result.returncode}")
                print(f"      {result.stdout}")
                return False

        except FileNotFoundError:
            print(
                "   ⚠️  huggingface-cli not found. Install with: pip install -U 'huggingface_hub[cli]'"
            )
            return False
        except Exception as e:
            print(f"   ❌ Error downloading {model['name']}: {e}")
            return False

    print("\n✅ Soprano TTS models download complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Senter Models")

    parser.add_argument(
        "--soprano-only", action="store_true", help="Download only Soprano TTS models"
    )

    args = parser.parse_args()

    print("📦 SENTER MODEL DOWNLOADER")
    print("=" * 60)

    models_dir = Path(__file__).parent.parent.parent / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Always download infrastructure models (Omni 3B + Nomic Embed)
    # These are required for Senter operation
    print("\n📹 Infrastructure Models (Required for Senter):")
    print("   - Qwen 3B Omni (multimodal decoder)")
    print("   - Nomic Embed (embedding model)")
    print("\n   💡 Note: These models are FIXED infrastructure and cannot be changed.")
    print("      They are part of Senter's utility belt around your model.")
    print("      Omni 3B: decodes images/audio/video into descriptions")
    print("      Nomic Embed: enables intelligent Focus selection")

    # Download infrastructure
    print("\n   Downloading infrastructure models...")
    omni_downloaded = False
    nomic_downloaded = False

    try:
        omni_file = Path(
            "/home/sovthpaw/Models/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q4_K_M.gguf"
        )
        if not omni_file.exists():
            print("   📥 Downloading Qwen 3B Omni...")
            omni_downloaded = (
                subprocess.run(
                    [
                        "huggingface-cli",
                        "download",
                        "https://huggingface.co/Qwen/Qwen2.5-Omni-3B-GGUF/resolve/main/Qwen2.5-Omni-3B-Q4_K_M.gguf",
                        "--local-dir",
                        "/home/sovthpaw/Models",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1200,
                ).returncode
                == 0
            )

        nomic_file = Path(
            "/home/sovthpaw/Models/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        )
        if not nomic_file.exists():
            print("   📥 Downloading Nomic Embed...")
            nomic_downloaded = (
                subprocess.run(
                    [
                        "huggingface-cli",
                        "download",
                        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
                        "--local-dir",
                        "/home/sovthpaw/Models",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1200,
                ).returncode
                == 0
            )
    except Exception as e:
        print(f"   ⚠️  Infrastructure download error: {e}")

    if args.soprano_only:
        print("\n🎤 Soprano TTS Models (Optional - for text-to-speech):")
        print("   - Soprano 70M (70M parameters, ~80MB file size)")
        print("   - Soprano 80M (80M parameters, ~85MB file size)")
        print("\n   💡 Usage:")
        print("      Streaming TTS: Generates speech in <15ms latency")
        print("      Each sentence sent to TTS as it's generated")
        print("      TTS speaks while next sentence generates")
        print("      Significant latency reduction vs non-streaming")
        print("\n   ⚠️  Requires: pip install -U 'soprano-tts'")

        if not download_soprano_tts(models_dir):
            success = False

    print("\n" + "=" * 60)
    print(
        f"\n{'✅ Infrastructure complete' if (omni_downloaded and nomic_downloaded) else '❌ Infrastructure download failed'}"
    )
    print("\n💡 Next Steps:")
    print("   1. Configure your central model in config/user_profile.json")
    print("   2. Run Senter: python3 scripts/senter_app.py")
    print("   3. Test Senter with streaming TTS enabled")
    print("\n📚 Documentation: SENTER_DOCUMENTATION.md")

    return 0 if (omni_downloaded and nomic_downloaded) else 1


if __name__ == "__main__":
    sys.exit(main())
