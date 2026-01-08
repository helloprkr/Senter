#!/usr/bin/env python3
"""
Setup Senter - Initial Configuration and Model Downloads
Guides users through setup with optional model downloads
"""

import json
from pathlib import Path
from typing import Dict, Any


def print_banner():
    """Print setup banner"""
    print("\n" + "=" * 70)
    print("🚀 SENTER SETUP")
    print("=" * 70)
    print("\nUniversal AI Personal Assistant - Model-Agnostic Architecture\n")


def download_infrastructure_models(senter_root: Path) -> bool:
    """
    Download required infrastructure models (Omni 3B + embedding)

    Returns: True if successful
    """
    print("\n📦 Step 1: Download Infrastructure Models (Required)")
    print("-" * 70)

    config_file = senter_root / "config" / "senter_config.json"

    if not config_file.exists():
        print("   ⚠️  senter_config.json not found")
        return False

    with open(config_file, "r") as f:
        config = json.load(f)

    infra_models = config.get("infrastructure_models", {})

    # Create Models directory
    models_dir = senter_root / "Models"
    models_dir.mkdir(exist_ok=True)

    # Download Omni 3B
    omni_config = infra_models.get("multimodal_decoder", {})
    omni_path = omni_config.get("path")
    mmproj_path = omni_config.get("mmproj")

    if omni_path:
        print("\n   📹 Checking Omni 3B model...")
        omni_file = senter_root / omni_path

        if omni_file.exists():
            print(f"   ✅ Omni 3B model exists: {omni_file.name}")
        else:
            print(f"   ⚠️  Omni 3B not found: {omni_path}")
            print(
                "   ℹ️  Please download manually and update path in config/senter_config.json"
            )
            print(f"   Expected path: {omni_path}")

    if mmproj_path:
        mmproj_file = senter_root / mmproj_path
        if mmproj_file.exists():
            print(f"   ✅ Multimodal projector exists: {mmproj_file.name}")
        else:
            print(f"   ⚠️  Multimodal projector not found: {mmproj_path}")

    # Download embedding model
    embed_config = infra_models.get("embedding_model", {})
    embed_path = embed_config.get("path")

    if embed_path:
        print("\n   🧠 Checking embedding model...")
        embed_file = senter_root / embed_path

        if embed_file.exists():
            print(f"   ✅ Embedding model exists: {embed_file.name}")
        else:
            print(f"   ⚠️  Embedding model not found: {embed_path}")
            print(
                "   ℹ️  Please download manually and update path in config/senter_config.json"
            )
            print(f"   Expected path: {embed_path}")

    print("\n   ℹ️  Infrastructure models configured in config/senter_config.json")
    print("   ℹ️  To download models, use huggingface-cli:")
    print("   ℹ️  huggingface-cli download Qwen/Qwen2.5-Omni-3B-GGUF")
    print("   ℹ️  huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF")

    return True


def configure_central_model(senter_root: Path) -> bool:
    """
    Configure user's central model

    Offers: download recommended models, use existing local models, or use API

    Returns: True if successful
    """
    print("\n🤖 Step 2: Configure Your Central Model")
    print("-" * 70)
    print("\n   Senter works with ANY model. Choose an option:")
    print("   1. Download a recommended model")
    print("   2. Use an existing local model")
    print("   3. Use an OpenAI-compatible API endpoint")

    choice = input("\n   Your choice [1/2/3]: ").strip()

    user_profile_file = senter_root / "config" / "user_profile.json"

    # Load existing user profile
    user_profile = {}
    if user_profile_file.exists():
        with open(user_profile_file, "r") as f:
            user_profile = json.load(f)

    if choice == "1":
        # Download recommended model
        return download_recommended_model(senter_root, user_profile_file, user_profile)

    elif choice == "2":
        # Use existing local model
        return use_local_model(senter_root, user_profile_file, user_profile)

    elif choice == "3":
        # Use API endpoint
        return use_api_endpoint(senter_root, user_profile_file, user_profile)
    else:
        print("\n   ⚠️  Invalid choice. Setup incomplete.")
        return False


def download_recommended_model(
    senter_root: Path, user_profile_file: Path, user_profile: Dict[str, Any]
) -> bool:
    """Offer recommended models for download"""
    print("\n   Recommended Models:")
    print("   A. Hermes 3 Llama 3.2 3B (fast, efficient, ~2.1GB)")
    print("   B. Qwen VL 8B (vision + text, ~5.4GB)")
    print("   C. Enter your own model URL")

    choice = input("\n   Choose [A/B/C]: ").strip().upper()

    config_file = senter_root / "config" / "senter_config.json"

    with open(config_file, "r") as f:
        config = json.load(f)

    recommended = config.get("recommended_models", {})

    if choice == "A" and "hermes_3b" in recommended:
        selected = recommended["hermes_3b"]
        print(f"\n   📥 Selected: {selected['name']}")
        print(f"   📝 Description: {selected['description']}")
        print(f"   💾 Size: ~{selected['size_gb']}GB")

        # Update user profile
        user_profile["central_model"] = {
            "type": "gguf",
            "path": selected["url"],
            "is_vlm": selected.get("is_vlm", False),
            "settings": {"max_tokens": 512, "temperature": 0.7, "context_window": 8192},
        }

    elif choice == "B" and "qwen_vl_8b" in recommended:
        selected = recommended["qwen_vl_8b"]
        print(f"\n   📥 Selected: {selected['name']}")
        print(f"   📝 Description: {selected['description']}")
        print(f"   💾 Size: ~{selected['size_gb']}GB")
        print("   👁️  VLM: Yes")

        # Update user profile
        user_profile["central_model"] = {
            "type": "gguf",
            "path": selected["url"],
            "is_vlm": selected.get("is_vlm", True),
            "settings": {"max_tokens": 512, "temperature": 0.7, "context_window": 8192},
        }

    elif choice == "C":
        model_url = input("\n   Enter model URL: ").strip()
        is_vlm = (
            input("   Is this a Vision-Language model? [y/N]: ").strip().lower() == "y"
        )

        print(f"\n   📥 Model URL: {model_url}")
        print(f"   👁️  VLM: {'Yes' if is_vlm else 'No'}")

        # Update user profile
        user_profile["central_model"] = {
            "type": "gguf",
            "path": model_url,
            "is_vlm": is_vlm,
            "settings": {"max_tokens": 512, "temperature": 0.7, "context_window": 8192},
        }

    else:
        print("\n   ⚠️  Invalid choice or model not available in config.")
        return False

    # Mark setup complete
    user_profile["setup_complete"] = True

    # Save user profile
    with open(user_profile_file, "w") as f:
        json.dump(user_profile, f, indent=2)

    print(f"\n   ✅ User profile updated: {user_profile_file}")
    print("   ℹ️  Your model has been configured!")
    print("   ℹ️  Note: Model downloads will need to be done manually")
    print("   ℹ️  Use huggingface-cli or visit the URLs to download")

    return True


def use_local_model(
    senter_root: Path, user_profile_file: Path, user_profile: Dict[str, Any]
) -> bool:
    """Detect and use existing local models"""
    models_dir = senter_root / "Models"

    if not models_dir.exists():
        print("\n   ⚠️  Models directory not found")
        return False

    # Find GGUF models
    gguf_models = list(models_dir.glob("**/*.gguf"))

    if not gguf_models:
        print("\n   ⚠️  No GGUF models found in Models/")
        print("   ℹ️  Please download models and place in Models/ directory")
        return False

    print(f"\n   Found {len(gguf_models)} local GGUF model(s):")

    # List models with numbers
    for i, model_path in enumerate(gguf_models, start=1):
        print(f"   {i}. {model_path.name}")

    model_num = input("\n   Enter model number to use: ").strip()

    try:
        model_num = int(model_num)
        if model_num < 1 or model_num > len(gguf_models):
            print("\n   ⚠️  Invalid model number")
            return False

        selected_model = gguf_models[model_num - 1]

        # Check if VLM (heuristic)
        model_name = selected_model.name.lower()
        is_vlm = any(kw in model_name for kw in ["vl", "vision", "multimodal", "mm"])

        print(f"\n   📥 Selected: {selected_model.name}")
        print(f"   👁️  VLM: {'Yes' if is_vlm else 'No'}")

        # Update user profile
        user_profile["central_model"] = {
            "type": "gguf",
            "path": str(selected_model),
            "is_vlm": is_vlm,
            "settings": {"max_tokens": 512, "temperature": 0.7, "context_window": 8192},
        }

        # Mark setup complete
        user_profile["setup_complete"] = True

        # Save user profile
        with open(user_profile_file, "w") as f:
            json.dump(user_profile, f, indent=2)

        print(f"\n   ✅ User profile updated: {user_profile_file}")
        print("   ℹ️  Your model has been configured!")

        return True

    except ValueError:
        print("\n   ⚠️  Invalid input")
        return False


def use_api_endpoint(
    senter_root: Path, user_profile_file: Path, user_profile: Dict[str, Any]
) -> bool:
    """Configure OpenAI-compatible API endpoint"""
    print("\n   Configure OpenAI-Compatible API Endpoint")

    endpoint = input("   Enter API endpoint URL: ").strip()
    model_name = input("   Model name (e.g., gpt-4o): ").strip()
    api_key = input("   API key (optional, press Enter to skip): ").strip()

    is_vlm = input("   Is this a Vision-Language model? [y/N]: ").strip().lower() == "y"

    print(f"\n   🌐 Endpoint: {endpoint}")
    print(f"   📥 Model: {model_name}")
    print(f"   👁️  VLM: {'Yes' if is_vlm else 'No'}")

    # Update user profile
    user_profile["central_model"] = {
        "type": "openai",
        "endpoint": endpoint,
        "model_name": model_name,
        "api_key": api_key if api_key else "",
        "is_vlm": is_vlm,
        "settings": {"max_tokens": 512, "temperature": 0.7, "context_window": 8192},
    }

    # Mark setup complete
    user_profile["setup_complete"] = True

    # Save user profile
    with open(user_profile_file, "w") as f:
        json.dump(user_profile, f, indent=2)

    print(f"\n   ✅ User profile updated: {user_profile_file}")
    print("   ℹ️  Your API model has been configured!")

    return True


def verify_setup(senter_root: Path) -> bool:
    """Verify setup configuration"""
    print("\n🔍 Step 3: Verify Setup")
    print("-" * 70)

    user_profile_file = senter_root / "config" / "user_profile.json"

    if not user_profile_file.exists():
        print("   ⚠️  user_profile.json not found")
        return False

    with open(user_profile_file, "r") as f:
        profile = json.load(f)

    # Check central model
    central_model = profile.get("central_model")
    if not central_model:
        print("   ⚠️  No central model configured")
        return False

    print(f"   ✅ Central model type: {central_model.get('type', 'N/A')}")
    print(f"   ✅ VLM enabled: {central_model.get('is_vlm', False)}")

    # Check Focuses directory
    focuses_dir = senter_root / "Focuses"
    if not focuses_dir.exists():
        print("   ⚠️  Focuses directory not found")
        return False

    # Check internal Focuses
    internal_dir = focuses_dir / "internal"
    if internal_dir.exists():
        internal_focuses = list([d.name for d in internal_dir.iterdir() if d.is_dir()])
        print(f"   ✅ Internal Focuses: {len(internal_focuses)} created")

    print("\n   ✅ Setup verification complete!")
    return True


def show_next_steps(senter_root: Path):
    """Show next steps after setup"""
    print("\n" + "=" * 70)
    print("💡 NEXT STEPS")
    print("=" * 70)
    print("\n   Senter is now ready to use!")
    print("\n   1. Run Senter app:")
    print("      python3 scripts/senter_app.py")
    print("\n   2. Or use Senter from Python:")
    print("      from senter import Senter")
    print("      from senter_selector import select_focus_and_agent")
    print("\n   3. Configure your models later:")
    print("      Edit config/user_profile.json")
    print("      Edit config/senter_config.json")
    print("\n   4. Learn more:")
    print("      Read the documentation in docs/")
    print("      Explore Focuses in Focuses/")
    print("\n" + "=" * 70)
    print("🎉 Senter setup complete!")
    print("=" * 70)


def main():
    """Main setup flow"""
    print_banner()

    # Get Senter root directory
    senter_root = Path(__file__).parent.parent

    # Step 1: Infrastructure models
    if not download_infrastructure_models(senter_root):
        print("\n   ⚠️  Setup incomplete. Please resolve infrastructure models first.")
        return

    # Step 2: Configure central model
    if not configure_central_model(senter_root):
        return

    # Step 3: Verify setup
    if not verify_setup(senter_root):
        return

    # Show next steps
    show_next_steps(senter_root)


if __name__ == "__main__":
    main()
