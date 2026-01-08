#!/usr/bin/env python3
"""
Senter Parallel Processing Framework
Manages background tasks for context analysis, user profiling, and agent evolution
"""

import os
import sys
import json
import threading
import time
import queue
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

class BackgroundTaskManager:
    """Manages background processing tasks for Senter"""

    def __init__(self):
        self.senter_root = Path(__file__).parent.parent
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue = queue.Queue()
        self.running = False
        self.workers: List[threading.Thread] = []

        # Task configurations
        self.task_configs = {
            "context_analyzer": {
                "interval": 30,  # seconds
                "function": self._analyze_context,
                "description": "Analyze conversation context and update SENTER.md"
            },
            "user_profiler": {
                "interval": 60,  # seconds
                "function": self._profile_user,
                "description": "Analyze user patterns and detect goals"
            },
            "agent_evolver": {
                "interval": 300,  # 5 minutes
                "function": self._evolve_agents,
                "description": "Update agent capabilities based on usage"
            },
            "model_monitor": {
                "interval": 120,  # 2 minutes
                "function": self._monitor_models,
                "description": "Monitor model server health"
            }
        }

    def start_background_processing(self):
        """Start all background processing tasks"""
        if self.running:
            print("Background processing already running")
            return

        print("🔄 Starting background processing...")
        self.running = True

        # Start worker threads
        for task_name, config in self.task_configs.items():
            worker = threading.Thread(
                target=self._task_worker,
                args=(task_name, config),
                daemon=True,
                name=f"background-{task_name}"
            )
            worker.start()
            self.workers.append(worker)

        print(f"✅ Started {len(self.workers)} background workers")

    def stop_background_processing(self):
        """Stop all background processing"""
        print("🛑 Stopping background processing...")
        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        print("✅ Background processing stopped")

    def _task_worker(self, task_name: str, config: Dict[str, Any]):
        """Worker thread for a specific background task"""
        while self.running:
            try:
                start_time = time.time()
                config["function"]()
                elapsed = time.time() - start_time

                # Log task completion
                print(f"✅ Background task '{task_name}' completed in {elapsed:.1f}s")

                # Wait for next interval
                time.sleep(config["interval"])

            except Exception as e:
                print(f"❌ Background task '{task_name}' failed: {e}")
                time.sleep(60)  # Wait before retrying

    def _analyze_context(self):
        """Analyze conversation context and update topic SENTER.md files"""
        # Load recent conversation data
        conversation_file = self.senter_root / "conversation_history.json"
        if not conversation_file.exists():
            return

        try:
            with open(conversation_file, 'r') as f:
                conversations = json.load(f)
        except:
            return

        # Analyze recent conversations (last 20 messages)
        recent_messages = conversations[-20:] if len(conversations) > 20 else conversations

        # Simple topic detection
        topic_keywords = {
            "coding": ["code", "programming", "python", "function", "debug", "git"],
            "creative": ["music", "image", "art", "design", "create", "generate"],
            "research": ["search", "find", "information", "learn", "study", "analyze"],
            "personal": ["schedule", "reminder", "task", "goal", "plan", "organize"]
        }

        detected_topics = set()
        for msg in recent_messages:
            content = msg.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    detected_topics.add(topic)

        # Update SENTER.md files for detected topics
        for topic in detected_topics:
            self._update_topic_context(topic, recent_messages)

    def _update_topic_context(self, topic: str, messages: List[Dict[str, Any]]):
        """Update SENTER.md for a specific topic with summarization"""
        topic_dir = self.senter_root / "Topics" / topic
        senter_file = topic_dir / "SENTER.md"

        # Read existing content
        existing_content = ""
        if senter_file.exists():
            try:
                with open(senter_file, 'r') as f:
                    existing_content = f.read()
            except:
                existing_content = ""

        # Create new context from messages
        new_context = f"\n## Context Update ({datetime.now().isoformat()})\n"
        new_context += f"- Messages analyzed: {len(messages)}\n"
        new_context += f"- Topic: {topic}\n"
        for msg in messages[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = str(content)
            new_context += f"- {role}: {content[:100]}...\n"
        new_context += "\n"

        # Combine old and new
        combined_content = existing_content + new_context

        # Simple summarization: keep recent parts, truncate if too long
        if len(combined_content) > 5000:
            # Keep header and recent updates
            lines = combined_content.split('\n')
            header_lines = []
            update_blocks = []
            current_block = []

            for line in lines:
                if line.startswith('# ') and not header_lines:
                    header_lines.append(line)
                elif line.startswith('## '):
                    if current_block:
                        update_blocks.append(current_block)
                    current_block = [line]
                else:
                    current_block.append(line)

            if current_block:
                update_blocks.append(current_block)

            # Keep header + last 3 update blocks
            summary_lines = header_lines + [line for block in update_blocks[-3:] for line in block]
            combined_content = '\n'.join(summary_lines)

        # Write back
        try:
            with open(senter_file, 'w') as f:
                f.write(combined_content)
        except Exception as e:
            print(f"Error updating {senter_file}: {e}")

    def _profile_user(self):
        """Analyze user patterns and detect potential goals"""
        # Load user profile
        profile_file = self.senter_root / "config" / "user_profile.json"
        if not profile_file.exists():
            return

        try:
            with open(profile_file, 'r') as f:
                profile = json.load(f)
        except:
            return

        # Load conversation history
        conversation_file = self.senter_root / "conversation_history.json"
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r') as f:
                    conversations = json.load(f)

                # Simple goal detection from conversation patterns
                goal_indicators = [
                    "i want to", "i need to", "i should", "my goal is",
                    "i plan to", "i'm trying to", "i hope to"
                ]

                potential_goals = []
                for msg in conversations[-50:]:  # Last 50 messages
                    content = msg.get("content", "").lower()
                    for indicator in goal_indicators:
                        if indicator in content:
                            # Extract potential goal
                            goal_text = content.split(indicator)[1].strip()[:100]
                            if goal_text and len(goal_text) > 10:
                                potential_goals.append(goal_text)

                # Update profile with potential goals
                if potential_goals:
                    existing_goals = profile.get("goals", [])
                    new_goals = [g for g in potential_goals if g not in existing_goals][:3]  # Max 3 new goals

                    if new_goals:
                        profile.setdefault("goals", []).extend(new_goals)
                        profile["last_updated"] = str(Path(__file__).stat().st_mtime)

                        # Save updated profile
                        with open(profile_file, 'w') as f:
                            json.dump(profile, f, indent=2)

                        print(f"🎯 Detected {len(new_goals)} potential goals")

            except Exception as e:
                print(f"Error in user profiling: {e}")

    def _evolve_agents(self):
        """Update agent capabilities based on usage patterns"""
        # This would analyze agent usage and suggest improvements
        # For now, just log that evolution check ran
        print("🧬 Agent evolution check completed")

    def _monitor_models(self):
        """Monitor model server health"""
        # Check if model servers are running
        config_file = self.senter_root / "config" / "senter_config.json"
        if not config_file.exists():
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            models = config.get("models", {})
            for model_type, model_config in models.items():
                port = model_config.get("server_port")
                if port:
                    # Simple health check
                    try:
                        import requests
                        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                        if response.status_code != 200:
                            print(f"⚠️ {model_type} server health check failed")
                    except:
                        print(f"⚠️ {model_type} server not responding")

        except Exception as e:
            print(f"Error monitoring models: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get status of background processing"""
        return {
            "running": self.running,
            "workers": len(self.workers),
            "tasks": list(self.task_configs.keys()),
            "queue_size": self.task_queue.qsize()
        }

    def force_task_run(self, task_name: str):
        """Force a specific task to run immediately"""
        if task_name in self.task_configs:
            try:
                self.task_configs[task_name]["function"]()
                print(f"✅ Forced {task_name} to run")
            except Exception as e:
                print(f"❌ Failed to force {task_name}: {e}")
        else:
            print(f"❌ Unknown task: {task_name}")


# Global instance for easy access
_background_manager = None

def get_background_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance"""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundTaskManager()
    return _background_manager

def start_background_processing():
    """Start background processing (convenience function)"""
    manager = get_background_manager()
    manager.start_background_processing()

def stop_background_processing():
    """Stop background processing (convenience function)"""
    manager = get_background_manager()
    manager.stop_background_processing()


def main():
    """CLI interface for background task management"""
    import argparse

    parser = argparse.ArgumentParser(description="Senter Background Task Manager")
    parser.add_argument("command", choices=["start", "stop", "status", "run-task"])
    parser.add_argument("--task", help="Task name for run-task command")

    args = parser.parse_args()

    manager = get_background_manager()

    if args.command == "start":
        manager.start_background_processing()
    elif args.command == "stop":
        manager.stop_background_processing()
    elif args.command == "status":
        status = manager.get_status()
        print("Background Processing Status:")
        print(f"  Running: {status['running']}")
        print(f"  Workers: {status['workers']}")
        print(f"  Tasks: {', '.join(status['tasks'])}")
        print(f"  Queue Size: {status['queue_size']}")
    elif args.command == "run-task":
        if not args.task:
            print("❌ --task required")
            return
        manager.force_task_run(args.task)


if __name__ == "__main__":
    main()