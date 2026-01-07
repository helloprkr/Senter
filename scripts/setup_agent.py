#!/usr/bin/env python3
"""
Senter Setup Agent
Automated setup and configuration for Senter AI Personal Assistant
"""

import os
import sys
import json
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

class SenterSetupAgent:
    """Automated setup and configuration agent for Senter"""

    def __init__(self):
        self.senter_root = Path(__file__).parent.parent
        self.config_file = self.senter_root / "config" / "senter_config.json"
        self.models_dir = self.senter_root / "Models"
        self.resources_dir = Path("../../Resources")  # Relative to Senter

        # Default configuration
        self.default_config = {
            "version": "1.0.0",
            "models": {
                "text": {
                    "model": "Qwen/Qwen2.5-Omni-3B",
                    "server_port": 8000,
                    "status": "not_loaded"
                },
                "vision": {
                    "model": "Qwen/Qwen2.5-Omni-3B",
                    "server_port": 8001,
                    "status": "not_loaded"
                },
                "audio": {
                    "model": "Qwen/Qwen2.5-Omni-3B",
                    "server_port": 8002,
                    "status": "not_loaded"
                },
                "music": {
                    "model": "ACE-Step/ACE-Step-v1-3.5B",
                    "server_port": 8003,
                    "status": "not_loaded"
                }
            },
            "system": {
                "auto_start_servers": True,
                "parallel_processing": True,
                "max_concurrent_agents": 3
            }
        }

        # Required packages
        self.required_packages = [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate",
            "numpy",
            "textual",
            "huggingface_hub",
            "psutil",
            "GPUtil",
            "requests",
            "pathlib2",
            "soundfile",
            "scipy",
            "Pillow"
        ]

    def check_system_requirements(self, detailed: bool = False) -> Dict[str, any]:
        """Check if system meets Senter requirements"""
        print("🔍 Checking system requirements...")

        results = {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "requirements_met": True,
            "issues": []
        }

        # Check Python version
        py_version = sys.version_info
        if py_version < (3, 8):
            results["issues"].append(f"Python {py_version.major}.{py_version.minor} detected. Python 3.8+ required.")
            results["requirements_met"] = False

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                results["gpu"] = f"CUDA available: {torch.cuda.get_device_name(0)}"
                results["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB"
            else:
                results["gpu"] = "No CUDA GPU detected"
                results["issues"].append("No CUDA GPU detected. Performance may be limited.")
        except ImportError:
            results["gpu"] = "PyTorch not installed"
            results["issues"].append("PyTorch not installed")

        # Check available RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total // (1024**3)
            results["ram"] = f"{ram_gb}GB"
            if ram_gb < 8:
                results["issues"].append(f"Only {ram_gb}GB RAM detected. 8GB+ recommended.")
        except ImportError:
            results["ram"] = "psutil not available"

        # Check disk space
        try:
            stat = os.statvfs(str(self.senter_root))
            free_gb = (stat.f_bavail * stat.f_frsize) // (1024**3)
            results["disk_free"] = f"{free_gb}GB"
            if free_gb < 50:
                results["issues"].append(f"Only {free_gb}GB free disk space. 50GB+ recommended for models.")
        except:
            results["disk_free"] = "Unable to check"

        if detailed:
            print(json.dumps(results, indent=2))
        else:
            print(f"✅ Python: {results['python_version']}")
            print(f"✅ Platform: {results['platform']} {results['architecture']}")
            print(f"✅ RAM: {results.get('ram', 'Unknown')}")
            print(f"✅ GPU: {results.get('gpu', 'Unknown')}")
            print(f"✅ Disk: {results.get('disk_free', 'Unknown')}")

            if results["issues"]:
                print("\n⚠️  Issues found:")
                for issue in results["issues"]:
                    print(f"   - {issue}")
            else:
                print("\n✅ All basic requirements met!")

        return results

    def install_dependencies(self, packages: List[str] = None, upgrade: bool = False) -> bool:
        """Install required Python packages"""
        if packages is None:
            packages = self.required_packages

        print(f"📦 Installing {len(packages)} packages...")

        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.extend(packages)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Dependencies installed successfully!")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False

    def download_model(self, model_name: str, model_type: str, provider: str = "huggingface") -> bool:
        """Download AI model from provider"""
        print(f"⬇️  Downloading {model_type} model: {model_name}")

        try:
            if provider == "huggingface":
                from huggingface_hub import snapshot_download

                # Create model directory
                model_dir = self.models_dir / model_name.replace("/", "_")
                model_dir.mkdir(exist_ok=True)

                # Download model
                snapshot_download(
                    repo_id=model_name,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )

                print(f"✅ Model downloaded to: {model_dir}")
                return True

            else:
                print(f"❌ Unsupported provider: {provider}")
                return False

        except Exception as e:
            print(f"❌ Failed to download model {model_name}: {e}")
            return False

    def start_model_server(self, model_path: str, port: int = 8000, server_type: str = "vllm") -> bool:
        """Start OpenAI-compatible model server"""
        print(f"🚀 Starting {server_type} server on port {port}...")

        try:
            if server_type == "vllm":
                # Check if vLLM is available
                try:
                    import vllm
                except ImportError:
                    print("❌ vLLM not installed. Install with: pip install vllm")
                    return False

                # Start vLLM server
                cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model_path,
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--trust-remote-code"
                ]

                print(f"Starting server with command: {' '.join(cmd)}")
                # For now, just show the command - in production this would run in background
                print("⚠️  Server start command prepared. Run manually or implement background process.")

            elif server_type == "llama-cpp":
                # Check if llama-cpp-python is available
                try:
                    from llama_cpp import Llama
                except ImportError:
                    print("❌ llama-cpp-python not installed.")
                    return False

                print("⚠️  llama-cpp server implementation pending")
                return False

            return True

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def initialize_senter(self, reset_config: bool = False, create_topics: bool = True) -> bool:
        """Complete Senter system initialization"""
        print("🚀 Initializing Senter system...")

        try:
            # Create config if it doesn't exist or reset requested
            if reset_config or not self.config_file.exists():
                print("📝 Creating default configuration...")
                self.config_file.parent.mkdir(exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(self.default_config, f, indent=2)
                print(f"✅ Config created: {self.config_file}")

            # Create topic structure
            if create_topics:
                print("📁 Creating topic structure...")
                topics_to_create = [
                    "user_personal",
                    "coding",
                    "creative",
                    "research",
                    "agents/analyzer",
                    "agents/summarizer",
                    "agents/router",
                    "agents/creative_writer"
                ]

                for topic in topics_to_create:
                    topic_dir = self.senter_root / "Topics" / topic
                    topic_dir.mkdir(parents=True, exist_ok=True)

                    # Create SENTER.md for each topic
                    senter_file = topic_dir / "SENTER.md"
                    if not senter_file.exists():
                        with open(senter_file, 'w') as f:
                            f.write(f"# {topic.replace('_', ' ').title()} Context\n\n")
                            f.write("## User Preferences\n\n")
                            f.write("## Patterns Observed\n\n")
                            f.write("## Goals & Objectives\n\n")
                            f.write("## Evolution Notes\n\n")
                        print(f"✅ Created: {senter_file}")

            # Create user profile if it doesn't exist
            user_profile_file = self.senter_root / "config" / "user_profile.json"
            if not user_profile_file.exists():
                user_profile = {
                    "preferences": {},
                    "goals": [],
                    "personality": {},
                    "communication_style": {},
                    "last_updated": str(Path(__file__).stat().st_mtime)
                }
                with open(user_profile_file, 'w') as f:
                    json.dump(user_profile, f, indent=2)
                print(f"✅ Created user profile: {user_profile_file}")

            print("✅ Senter initialization complete!")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def run_full_setup(self) -> bool:
        """Run complete Senter setup process"""
        print("🎯 Starting complete Senter setup...")

        # Step 1: Check system requirements
        print("\n1️⃣ Checking system requirements...")
        req_check = self.check_system_requirements()
        if not req_check["requirements_met"]:
            print("❌ System requirements not met. Please resolve issues and try again.")
            return False

        # Step 2: Install dependencies
        print("\n2️⃣ Installing dependencies...")
        if not self.install_dependencies():
            print("❌ Failed to install dependencies.")
            return False

        # Step 3: Initialize Senter
        print("\n3️⃣ Initializing Senter...")
        if not self.initialize_senter():
            print("❌ Failed to initialize Senter.")
            return False

        # Step 4: Download default models (optional - user can do this later)
        print("\n4️⃣ Default model download (optional)...")
        print("⚠️  Skipping automatic model download. Use download_model() when ready.")

        print("\n🎉 Senter setup complete!")
        print("Next steps:")
        print("1. Download models: python setup_agent.py download-models")
        print("2. Start Senter: python scripts/senter.py")
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Senter Setup Agent")
    parser.add_argument("command", choices=[
        "check-requirements", "install-deps", "download-model",
        "start-server", "init", "full-setup"
    ])
    parser.add_argument("--model-name", help="Model name for download")
    parser.add_argument("--model-type", choices=["text", "vision", "audio"], default="text")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-type", choices=["vllm", "llama-cpp"], default="vllm")
    parser.add_argument("--detailed", action="store_true")

    args = parser.parse_args()

    agent = SenterSetupAgent()

    if args.command == "check-requirements":
        agent.check_system_requirements(args.detailed)
    elif args.command == "install-deps":
        agent.install_dependencies()
    elif args.command == "download-model":
        if not args.model_name:
            print("❌ --model-name required")
            return
        agent.download_model(args.model_name, args.model_type)
    elif args.command == "start-server":
        if not args.model_name:
            print("❌ --model-name required")
            return
        agent.start_model_server(args.model_name, args.port, args.server_type)
    elif args.command == "init":
        agent.initialize_senter()
    elif args.command == "full-setup":
        agent.run_full_setup()


if __name__ == "__main__":
    main()