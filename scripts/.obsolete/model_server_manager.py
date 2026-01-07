#!/usr/bin/env python3
"""
Senter Model Server Manager
Manages OpenAI-compatible model servers using vLLM
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

class ModelServerManager:
    """Manages vLLM model servers for Senter"""

    def __init__(self):
        self.senter_root = Path(__file__).parent.parent
        self.models_dir = self.senter_root / "Models"
        self.config_file = self.senter_root / "config" / "senter_config.json"
        self.servers: Dict[str, Dict[str, Any]] = {}
        self._load_config()

    def _load_config(self):
        """Load model configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.model_config = config.get("models", {})
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            self.model_config = {}

    def check_server_health(self, port: int) -> bool:
        """Check if a server is healthy on the given port"""
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_model_server(self, model_name: str, model_type: str = "text", port: int = 8000) -> bool:
        """Start a vLLM server for a specific model"""
        print(f"🚀 Starting {model_type} model server: {model_name} on port {port}")

        # Check if model exists
        model_path = self.models_dir / model_name.replace("/", "_")
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return False

        # Check if server is already running
        if self.check_server_health(port):
            print(f"✅ Server already running on port {port}")
            return True

        try:
            # Start vLLM server
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--trust-remote-code",
                "--max-model-len", "8192",
                "--gpu-memory-utilization", "0.8"
            ]

            print(f"Starting server with command: {' '.join(cmd)}")

            # Start server in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.senter_root)
            )

            # Wait for server to start
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(2)
                if self.check_server_health(port):
                    print(f"✅ Server started successfully on port {port}")

                    # Store server info
                    self.servers[model_type] = {
                        "model": model_name,
                        "port": port,
                        "process": process,
                        "status": "running"
                    }

                    return True

                if process.poll() is not None:
                    # Process died
                    stdout, stderr = process.communicate()
                    print(f"❌ Server failed to start: {stderr.decode()}")
                    return False

            print(f"❌ Server failed to respond within timeout")
            process.terminate()
            return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop_model_server(self, model_type: str) -> bool:
        """Stop a model server"""
        if model_type not in self.servers:
            print(f"❌ No server running for {model_type}")
            return False

        server_info = self.servers[model_type]
        process = server_info.get("process")

        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

            # Check if it's actually stopped
            if not self.check_server_health(server_info["port"]):
                server_info["status"] = "stopped"
                print(f"✅ Server stopped for {model_type}")
                return True
            else:
                print(f"⚠️ Server may still be running")
                return False
        else:
            print(f"✅ Server was not running for {model_type}")
            return True

    def start_all_servers(self) -> bool:
        """Start all configured model servers"""
        print("🎯 Starting all model servers...")

        success_count = 0
        total_count = len(self.model_config)

        for model_type, config in self.model_config.items():
            model_name = config.get("model")
            port = config.get("server_port", 8000)

            if model_name and self.start_model_server(model_name, model_type, port):
                success_count += 1
            else:
                print(f"❌ Failed to start {model_type} server")

        print(f"✅ Started {success_count}/{total_count} model servers")
        return success_count == total_count

    def stop_all_servers(self) -> bool:
        """Stop all running model servers"""
        print("🛑 Stopping all model servers...")

        success_count = 0
        for model_type in list(self.servers.keys()):
            if self.stop_model_server(model_type):
                success_count += 1

        print(f"✅ Stopped {success_count} model servers")
        return True

    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers"""
        status = {}

        for model_type, config in self.model_config.items():
            port = config.get("server_port")
            is_healthy = self.check_server_health(port) if port else False

            status[model_type] = {
                "model": config.get("model"),
                "port": port,
                "healthy": is_healthy,
                "status": "running" if is_healthy else "stopped"
            }

        return status

    def test_server_connection(self, model_type: str) -> bool:
        """Test connection to a model server"""
        if model_type not in self.model_config:
            print(f"❌ Unknown model type: {model_type}")
            return False

        config = self.model_config[model_type]
        port = config.get("server_port")

        if not port:
            print(f"❌ No port configured for {model_type}")
            return False

        try:
            # Test chat completion endpoint
            response = requests.post(
                f"http://127.0.0.1:{port}/chat/completions",
                json={
                    "model": config.get("model", "test"),
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                },
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ {model_type} server connection successful")
                return True
            else:
                print(f"❌ {model_type} server returned status {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Failed to connect to {model_type} server: {e}")
            return False


def main():
    """CLI interface for model server management"""
    import argparse

    parser = argparse.ArgumentParser(description="Senter Model Server Manager")
    parser.add_argument("command", choices=[
        "start", "stop", "status", "test", "start-all", "stop-all"
    ])
    parser.add_argument("--model-type", choices=["text", "vision", "audio", "music"])
    parser.add_argument("--model-name", help="Model name/path")
    parser.add_argument("--port", type=int, help="Server port")

    args = parser.parse_args()

    manager = ModelServerManager()

    if args.command == "start":
        if not args.model_type or not args.model_name:
            print("❌ --model-type and --model-name required")
            return
        port = args.port or 8000
        manager.start_model_server(args.model_name, args.model_type, port)

    elif args.command == "stop":
        if not args.model_type:
            print("❌ --model-type required")
            return
        manager.stop_model_server(args.model_type)

    elif args.command == "start-all":
        manager.start_all_servers()

    elif args.command == "stop-all":
        manager.stop_all_servers()

    elif args.command == "status":
        status = manager.get_server_status()
        print("Model Server Status:")
        for model_type, info in status.items():
            health_icon = "🟢" if info["healthy"] else "🔴"
            print(f"  {health_icon} {model_type}: {info['model']} (port {info['port']}) - {info['status']}")

    elif args.command == "test":
        if not args.model_type:
            print("❌ --model-type required")
            return
        manager.test_server_connection(args.model_type)


if __name__ == "__main__":
    main()