#!/usr/bin/env python3
"""
Task Result Storage (US-003)

Stores completed task results persistently for later retrieval.

Features:
- Results stored in data/tasks/results/
- Each result includes: task_id, goal_id, result content, timestamp
- Results queryable by task_id, goal_id, or time range
- Results persist across daemon restarts
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger('senter.task_results')


@dataclass
class TaskResult:
    """A stored task result"""
    task_id: str
    goal_id: str
    result: Any  # The actual result content
    timestamp: float = field(default_factory=time.time)
    status: str = "completed"  # completed, error, timeout
    worker: str = "unknown"
    latency_ms: int = 0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "result": self.result,
            "timestamp": self.timestamp,
            "status": self.status,
            "worker": self.worker,
            "latency_ms": self.latency_ms,
            "description": self.description,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        # Remove derived fields
        d = {k: v for k, v in d.items() if k != "datetime"}
        return cls(**d)


class TaskResultStorage:
    """
    Persistent storage for task results.

    Results are stored as individual JSON files in data/tasks/results/
    organized by date for easy cleanup and querying.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Index for quick lookups
        self.index_file = self.results_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load result index from disk"""
        if self.index_file.exists():
            try:
                self.index = json.loads(self.index_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
                self.index = {"tasks": {}, "goals": {}}
        else:
            self.index = {"tasks": {}, "goals": {}}

    def _save_index(self):
        """Save result index to disk"""
        self.index_file.write_text(json.dumps(self.index, indent=2))

    def _get_result_file(self, task_id: str) -> Path:
        """Get the file path for a task result"""
        # Organize by date
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.results_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{task_id}.json"

    def store(self, result: TaskResult) -> bool:
        """Store a task result"""
        try:
            # Save result file
            result_file = self._get_result_file(result.task_id)
            result_file.write_text(json.dumps(result.to_dict(), indent=2))

            # Update index
            self.index["tasks"][result.task_id] = {
                "file": str(result_file.relative_to(self.results_dir)),
                "goal_id": result.goal_id,
                "timestamp": result.timestamp,
                "status": result.status
            }

            # Update goal index
            if result.goal_id not in self.index["goals"]:
                self.index["goals"][result.goal_id] = []
            if result.task_id not in self.index["goals"][result.goal_id]:
                self.index["goals"][result.goal_id].append(result.task_id)

            self._save_index()

            logger.info(f"Stored result for task {result.task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            return False

    def get_by_task_id(self, task_id: str) -> Optional[TaskResult]:
        """Get a result by task ID"""
        if task_id not in self.index["tasks"]:
            return None

        try:
            rel_path = self.index["tasks"][task_id]["file"]
            result_file = self.results_dir / rel_path
            data = json.loads(result_file.read_text())
            return TaskResult.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load result {task_id}: {e}")
            return None

    def get_by_goal_id(self, goal_id: str) -> list[TaskResult]:
        """Get all results for a goal"""
        results = []
        task_ids = self.index["goals"].get(goal_id, [])

        for task_id in task_ids:
            result = self.get_by_task_id(task_id)
            if result:
                results.append(result)

        return results

    def get_recent(self, limit: int = 20, since: float = None) -> list[TaskResult]:
        """Get recent results"""
        results = []

        # Sort tasks by timestamp
        sorted_tasks = sorted(
            self.index["tasks"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )

        for task_id, info in sorted_tasks:
            if since and info["timestamp"] < since:
                continue

            result = self.get_by_task_id(task_id)
            if result:
                results.append(result)

            if len(results) >= limit:
                break

        return results

    def get_summary(self) -> dict:
        """Get storage summary"""
        total = len(self.index["tasks"])
        goals = len(self.index["goals"])

        # Count by status
        by_status = {}
        for task_info in self.index["tasks"].values():
            status = task_info.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_results": total,
            "total_goals": goals,
            "by_status": by_status,
            "storage_path": str(self.results_dir)
        }

    def clear_old(self, days: int = 30) -> int:
        """Clear results older than N days"""
        cutoff = time.time() - (days * 24 * 3600)
        removed = 0

        for task_id, info in list(self.index["tasks"].items()):
            if info["timestamp"] < cutoff:
                try:
                    rel_path = info["file"]
                    result_file = self.results_dir / rel_path
                    if result_file.exists():
                        result_file.unlink()

                    del self.index["tasks"][task_id]

                    # Update goal index
                    goal_id = info["goal_id"]
                    if goal_id in self.index["goals"]:
                        if task_id in self.index["goals"][goal_id]:
                            self.index["goals"][goal_id].remove(task_id)
                        if not self.index["goals"][goal_id]:
                            del self.index["goals"][goal_id]

                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old result {task_id}: {e}")

        if removed > 0:
            self._save_index()
            logger.info(f"Removed {removed} old results")

        return removed


# Test
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = TaskResultStorage(Path(tmpdir) / "results")

        # Store some results
        result1 = TaskResult(
            task_id="task_001",
            goal_id="goal_001",
            result={"response": "Research completed on AI trends"},
            description="Research AI trends for 2026",
            worker="research",
            latency_ms=1500
        )
        storage.store(result1)

        result2 = TaskResult(
            task_id="task_002",
            goal_id="goal_001",
            result={"response": "Summary of research findings"},
            description="Summarize research findings",
            worker="research",
            latency_ms=800
        )
        storage.store(result2)

        # Query
        print("\n=== By Task ID ===")
        r = storage.get_by_task_id("task_001")
        print(json.dumps(r.to_dict(), indent=2))

        print("\n=== By Goal ID ===")
        results = storage.get_by_goal_id("goal_001")
        print(f"Found {len(results)} results for goal_001")

        print("\n=== Recent ===")
        recent = storage.get_recent(limit=10)
        for r in recent:
            print(f"  {r.task_id}: {r.description[:40]}")

        print("\n=== Summary ===")
        print(json.dumps(storage.get_summary(), indent=2))

        print("\nTest complete")
