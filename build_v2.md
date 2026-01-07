# SENTER 3.0: Build v2 - From Working to Phenomenal

**Document Version**: 2.0  
**Created**: January 7, 2026  
**Purpose**: Transform Senter from "shallow implementation" to "enterprise-grade AI assistant fulfilling the complete vision"

---

## THE VISION (What We're Building Toward)

> "A local AI assistant that works for you 24/7 — not just when you're talking to it."

### Core Capabilities (Must Achieve)

| Capability | Description |
|------------|-------------|
| **24/7 Operation** | Runs as daemon, works while you're away |
| **Autonomous Work** | Executes tasks without prompting |
| **Parallel Processing** | Two inference processes (foreground + background) |
| **Activity Learning** | Learns goals from computer activity |
| **Proactive Research** | Researches your interests autonomously |
| **Voice + Vision** | Gaze + speech activation, no wake word |
| **True Privacy** | Everything local, nothing to cloud |
| **Self-Evolution** | Actually improves from interactions |

---

## CURRENT STATE ASSESSMENT

### What's Working ✅

| Component | Status | Quality |
|-----------|--------|---------|
| Chat loop | ✅ Working | Good |
| Cognitive state detection | ✅ Working | Regex-based (shallow) |
| 4 memory types | ✅ Working | LIKE queries (shallow) |
| Trust tracking | ✅ Working | Persists across sessions |
| 4 coupling modes | ✅ Working | Mode switching works |
| Fitness tracking | ✅ Working | Hardcoded metrics (shallow) |
| Web search | ✅ Working | DuckDuckGo integration |
| 59/59 tests | ✅ Passing | Good coverage |

### What's Missing ❌

| Component | Current State | Required State |
|-----------|---------------|----------------|
| **Daemon mode** | CLI only, exits on quit | Persistent background service |
| **Task queue** | None | Priority queue with autonomous execution |
| **Parallel inference** | Single model | Two simultaneous GPU processes |
| **Background research** | "Parallel mode" adds text only | Actual async task execution |
| **Semantic search** | SQL LIKE queries | Vector embeddings + similarity |
| **Real evolution** | Proposes mutations, doesn't apply | Actual genome modification |
| **Voice input** | None | Whisper STT integration |
| **Vision/gaze** | None | Camera + face detection |
| **Activity monitoring** | None | Screen OCR, app tracking |
| **Proactive suggestions** | None | Goal-based recommendation engine |

---

## PART 1: FOUNDATION UPGRADES (Days 1-3)

### 1.1 Real Semantic Search with Embeddings

**Problem**: Memory retrieval uses SQL LIKE, not semantic similarity.

**Current Code** (memory/semantic.py):
```python
conditions = " OR ".join(["content LIKE ?" for _ in keywords])  # BAD
```

**Required Implementation**:

```python
# memory/semantic.py - REPLACE search method

import numpy as np
from typing import List, Dict, Any, Optional

class SemanticMemory:
    def __init__(self, config: Dict, db_path: Path, embeddings_model):
        self.embeddings = embeddings_model  # Inject embedding model
        self._init_db()
    
    def _init_db(self):
        """Initialize with embedding column."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS semantic (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                domain TEXT,
                embedding BLOB,  -- Store as numpy bytes
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                decay_factor REAL DEFAULT 1.0
            );
            
            CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic(domain);
        """)
    
    def store(self, content: str, domain: str = "general") -> str:
        """Store with embedding."""
        import uuid
        fact_id = str(uuid.uuid4())[:8]
        
        # Generate embedding
        embedding = self.embeddings.embed(content)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        self.conn.execute("""
            INSERT INTO semantic (id, content, domain, embedding)
            VALUES (?, ?, ?, ?)
        """, (fact_id, content, domain, embedding_bytes))
        self.conn.commit()
        
        return fact_id
    
    def search(
        self,
        query: str,
        limit: int = 5,
        domain: str = None,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search."""
        # Generate query embedding
        query_embedding = np.array(self.embeddings.embed(query), dtype=np.float32)
        
        # Get all candidates (or filter by domain)
        if domain:
            cursor = self.conn.execute(
                "SELECT id, content, domain, embedding FROM semantic WHERE domain = ?",
                (domain,)
            )
        else:
            cursor = self.conn.execute(
                "SELECT id, content, domain, embedding FROM semantic"
            )
        
        results = []
        for row in cursor.fetchall():
            if row['embedding']:
                stored_embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity >= threshold:
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'domain': row['domain'],
                        'similarity': float(similarity)
                    })
        
        # Sort by similarity, return top N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
```

**Success Criteria**:
```bash
# Test semantic search
python -c "
from memory.semantic import SemanticMemory
from models.embeddings import EmbeddingModel

embed = EmbeddingModel({'type': 'ollama', 'model': 'nomic-embed-text'})
mem = SemanticMemory({}, 'test.db', embed)

mem.store('My favorite programming language is Python', 'preference')
mem.store('I work at Anthropic on AI safety', 'work')
mem.store('My birthday is March 5th', 'personal')

# Should find the Python one even with different wording
results = mem.search('What coding language do I like?')
print(results[0]['content'])  # Should mention Python
"
```

---

### 1.2 Real Evolution That Modifies Genome

**Problem**: Mutations are proposed but never applied.

**Current Code** (evolution/mutations.py):
```python
def propose(self, fitness, episode):
    # Proposes mutation but doesn't apply it
    return Mutation(...)
```

**Required Implementation**:

```python
# evolution/mutations.py - COMPLETE REWRITE

import yaml
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy


@dataclass
class Mutation:
    """A proposed change to the genome."""
    id: str
    type: str  # prompt_refinement, threshold_change, trigger_addition
    target: str  # Path in genome (e.g., "coupling.trust.initial")
    old_value: Any
    new_value: Any
    reason: str
    fitness_before: float
    proposed_at: datetime


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    mutation: Mutation
    fitness_after: float
    interactions_tested: int
    success: bool
    rolled_back: bool


class MutationEngine:
    """
    Actually applies mutations to genome and tracks results.
    
    The key insight: Evolution only works if mutations are APPLIED,
    TESTED, and SELECTED based on outcomes.
    """
    
    def __init__(self, config: Dict, genome: Dict, genome_path: Path):
        self.config = config
        self.genome = genome
        self.genome_path = genome_path
        self.rate = config.get('rate', 0.05)
        
        # Track active experiment
        self.active_mutation: Optional[Mutation] = None
        self.experiment_start_fitness: float = 0.0
        self.experiment_interactions: int = 0
        self.experiment_duration = config.get('experiment_duration', 10)
        
        # History
        self.history: List[MutationResult] = []
        self._load_history()
    
    def _load_history(self):
        """Load mutation history from disk."""
        history_file = self.genome_path.parent / "data" / "evolution_history.json"
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
                # Reconstruct from JSON
                self.history = [self._dict_to_result(r) for r in data]
    
    def should_mutate(self, fitness: float) -> bool:
        """Decide if we should propose a mutation."""
        # Don't mutate if experiment in progress
        if self.active_mutation:
            return False
        
        # Higher chance of mutation when fitness is low
        import random
        adjusted_rate = self.rate * (2.0 - fitness)  # Higher rate for lower fitness
        return random.random() < adjusted_rate
    
    def propose(self, fitness: float, episode) -> Optional[Mutation]:
        """
        Propose a mutation based on what's not working.
        
        Analyzes the episode to find what to improve.
        """
        import uuid
        
        # Analyze episode for mutation opportunities
        mutation_type, target, old_val, new_val, reason = self._analyze_for_mutation(episode)
        
        if mutation_type is None:
            return None
        
        return Mutation(
            id=str(uuid.uuid4())[:8],
            type=mutation_type,
            target=target,
            old_value=old_val,
            new_value=new_val,
            reason=reason,
            fitness_before=fitness,
            proposed_at=datetime.now()
        )
    
    def _analyze_for_mutation(self, episode) -> tuple:
        """Analyze episode to determine what mutation would help."""
        cognitive_state = episode.get('cognitive_state', {})
        mode = episode.get('mode', 'DIALOGUE')
        
        # If user was frustrated but we didn't detect it well
        if cognitive_state.get('frustration', 0) > 0.3:
            # Maybe adjust frustration detection threshold
            current_threshold = self._get_genome_value(
                'coupling.human_model.frustration_threshold', 0.3
            )
            return (
                'threshold_change',
                'coupling.human_model.frustration_threshold',
                current_threshold,
                max(0.1, current_threshold - 0.05),
                f"User showed frustration ({cognitive_state.get('frustration'):.2f}) - lowering detection threshold"
            )
        
        # If mode seems mismatched
        input_text = episode.get('input', '').lower()
        if 'teach' in input_text and mode != 'TEACHING':
            current_triggers = self._get_genome_value(
                'coupling.protocols.teaching.triggers', []
            )
            if 'teach' not in current_triggers:
                return (
                    'trigger_addition',
                    'coupling.protocols.teaching.triggers',
                    current_triggers,
                    current_triggers + ['teach'],
                    f"User said 'teach' but mode was {mode} - adding trigger"
                )
        
        # If trust is stagnant, adjust increment
        trust_level = episode.get('joint_state', {}).get('trust_level', 0.5)
        if 0.45 < trust_level < 0.55:  # Stuck around default
            current_increment = self._get_genome_value(
                'coupling.trust.increase_on.successful_task_completion', 0.02
            )
            return (
                'threshold_change',
                'coupling.trust.increase_on.successful_task_completion',
                current_increment,
                current_increment + 0.01,
                "Trust seems stagnant - increasing success reward"
            )
        
        return (None, None, None, None, None)
    
    def _get_genome_value(self, path: str, default: Any) -> Any:
        """Get a value from genome by dot-notation path."""
        parts = path.split('.')
        current = self.genome
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def _set_genome_value(self, path: str, value: Any):
        """Set a value in genome by dot-notation path."""
        parts = path.split('.')
        current = self.genome
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    def apply(self, mutation: Mutation):
        """
        Apply a mutation to the genome and start experiment.
        
        THIS IS THE KEY: We actually modify the configuration.
        """
        # Store original for rollback
        self.active_mutation = mutation
        self.experiment_start_fitness = mutation.fitness_before
        self.experiment_interactions = 0
        
        # Apply the mutation to in-memory genome
        self._set_genome_value(mutation.target, mutation.new_value)
        
        print(f"[EVOLUTION] Applied mutation {mutation.id}: {mutation.reason}")
        print(f"[EVOLUTION] Changed {mutation.target}: {mutation.old_value} -> {mutation.new_value}")
    
    def record_interaction(self, fitness: float):
        """Record an interaction during an experiment."""
        if not self.active_mutation:
            return
        
        self.experiment_interactions += 1
        
        # Check if experiment is complete
        if self.experiment_interactions >= self.experiment_duration:
            self._evaluate_experiment(fitness)
    
    def _evaluate_experiment(self, final_fitness: float):
        """Evaluate if the mutation improved things."""
        mutation = self.active_mutation
        avg_fitness_change = final_fitness - mutation.fitness_before
        
        success = avg_fitness_change > 0
        rolled_back = False
        
        if not success:
            # Rollback
            self._set_genome_value(mutation.target, mutation.old_value)
            rolled_back = True
            print(f"[EVOLUTION] Rolled back mutation {mutation.id}: fitness dropped")
        else:
            # Persist successful mutation to genome file
            self._persist_genome()
            print(f"[EVOLUTION] Kept mutation {mutation.id}: fitness improved by {avg_fitness_change:.3f}")
        
        # Record result
        result = MutationResult(
            mutation=mutation,
            fitness_after=final_fitness,
            interactions_tested=self.experiment_interactions,
            success=success,
            rolled_back=rolled_back
        )
        self.history.append(result)
        
        # Clear active experiment
        self.active_mutation = None
        self.experiment_interactions = 0
    
    def _persist_genome(self):
        """Write modified genome back to file."""
        # Create backup first
        backup_path = self.genome_path.parent / "data" / "genome_backups"
        backup_path.mkdir(exist_ok=True)
        backup_file = backup_path / f"genome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        import shutil
        shutil.copy(self.genome_path, backup_file)
        
        # Write updated genome
        with open(self.genome_path, 'w') as f:
            yaml.dump(self.genome, f, default_flow_style=False)
        
        print(f"[EVOLUTION] Persisted genome changes (backup: {backup_file.name})")
    
    def persist(self):
        """Save mutation history."""
        history_file = self.genome_path.parent / "data" / "evolution_history.json"
        history_file.parent.mkdir(exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump([self._result_to_dict(r) for r in self.history], f, indent=2)
    
    def _result_to_dict(self, result: MutationResult) -> Dict:
        return {
            'mutation': {
                'id': result.mutation.id,
                'type': result.mutation.type,
                'target': result.mutation.target,
                'old_value': result.mutation.old_value,
                'new_value': result.mutation.new_value,
                'reason': result.mutation.reason,
                'fitness_before': result.mutation.fitness_before,
                'proposed_at': result.mutation.proposed_at.isoformat()
            },
            'fitness_after': result.fitness_after,
            'interactions_tested': result.interactions_tested,
            'success': result.success,
            'rolled_back': result.rolled_back
        }
    
    def _dict_to_result(self, data: Dict) -> MutationResult:
        m = data['mutation']
        return MutationResult(
            mutation=Mutation(
                id=m['id'],
                type=m['type'],
                target=m['target'],
                old_value=m['old_value'],
                new_value=m['new_value'],
                reason=m['reason'],
                fitness_before=m['fitness_before'],
                proposed_at=datetime.fromisoformat(m['proposed_at'])
            ),
            fitness_after=data['fitness_after'],
            interactions_tested=data['interactions_tested'],
            success=data['success'],
            rolled_back=data['rolled_back']
        )
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of evolution progress."""
        if not self.history:
            return {'total': 0, 'successful': 0, 'rolled_back': 0}
        
        return {
            'total': len(self.history),
            'successful': sum(1 for r in self.history if r.success),
            'rolled_back': sum(1 for r in self.history if r.rolled_back),
            'avg_fitness_improvement': sum(
                r.fitness_after - r.mutation.fitness_before 
                for r in self.history if r.success
            ) / max(1, sum(1 for r in self.history if r.success)),
            'recent_mutations': [
                {
                    'type': r.mutation.type,
                    'target': r.mutation.target,
                    'success': r.success,
                    'reason': r.mutation.reason
                }
                for r in self.history[-5:]
            ]
        }
```

**Success Criteria**:
```bash
# Run 50 interactions and verify:
# 1. genome.yaml has actually been modified
# 2. Evolution history shows successful mutations
# 3. Fitness trend is positive

python -c "
from evolution.mutations import MutationEngine
import yaml

# Check evolution summary
engine = MutationEngine({}, yaml.safe_load(open('genome.yaml')), Path('genome.yaml'))
summary = engine.get_evolution_summary()
print(f'Total mutations: {summary[\"total\"]}')
print(f'Successful: {summary[\"successful\"]}')
print(f'Rolled back: {summary[\"rolled_back\"]}')
"
```

---

## PART 2: DAEMON MODE & BACKGROUND WORK (Days 4-7)

### 2.1 Daemon Architecture

**Problem**: Senter only runs when CLI is active.

**Required Implementation**:

```python
# daemon/senter_daemon.py - NEW FILE

"""
Senter Daemon - Runs 24/7, even when CLI is closed.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    SENTER DAEMON                             │
│                                                             │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │  TASK QUEUE     │────▶│  BACKGROUND WORKER          │   │
│  │  (Priority)     │     │  - Research tasks           │   │
│  └─────────────────┘     │  - File organization        │   │
│          ▲               │  - Goal progress            │   │
│          │               └─────────────────────────────┘   │
│          │                                                  │
│  ┌───────┴───────┐       ┌─────────────────────────────┐   │
│  │  IPC SERVER   │◀─────▶│  CLI / TUI CLIENT           │   │
│  │  (Unix Socket)│       │  (Connects when active)     │   │
│  └───────────────┘       └─────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import os
import signal
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import heapq


class TaskPriority(Enum):
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass(order=True)
class Task:
    priority: int
    created_at: datetime = field(compare=False)
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)  # research, organize, remind, etc.
    description: str = field(compare=False)
    parameters: Dict[str, Any] = field(compare=False, default_factory=dict)
    status: str = field(compare=False, default="pending")
    result: Optional[str] = field(compare=False, default=None)


class TaskQueue:
    """Priority queue for autonomous tasks."""
    
    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.tasks: List[Task] = []
        self._load()
    
    def _load(self):
        """Load tasks from disk."""
        if self.persist_path.exists():
            with open(self.persist_path) as f:
                data = json.load(f)
                for t in data:
                    task = Task(
                        priority=t['priority'],
                        created_at=datetime.fromisoformat(t['created_at']),
                        task_id=t['task_id'],
                        task_type=t['task_type'],
                        description=t['description'],
                        parameters=t['parameters'],
                        status=t['status'],
                        result=t.get('result')
                    )
                    if task.status == 'pending':
                        heapq.heappush(self.tasks, task)
    
    def _save(self):
        """Persist tasks to disk."""
        data = [
            {
                'priority': t.priority,
                'created_at': t.created_at.isoformat(),
                'task_id': t.task_id,
                'task_type': t.task_type,
                'description': t.description,
                'parameters': t.parameters,
                'status': t.status,
                'result': t.result
            }
            for t in self.tasks
        ]
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add(self, task: Task):
        """Add task to queue."""
        heapq.heappush(self.tasks, task)
        self._save()
    
    def pop(self) -> Optional[Task]:
        """Get highest priority task."""
        if self.tasks:
            task = heapq.heappop(self.tasks)
            self._save()
            return task
        return None
    
    def peek(self) -> Optional[Task]:
        """Look at highest priority task without removing."""
        return self.tasks[0] if self.tasks else None
    
    def list_all(self) -> List[Task]:
        """Get all pending tasks."""
        return sorted(self.tasks)


class BackgroundWorker:
    """Executes tasks autonomously."""
    
    def __init__(self, senter_engine, task_queue: TaskQueue):
        self.engine = senter_engine
        self.queue = task_queue
        self.running = False
        self.current_task: Optional[Task] = None
        self.completed_tasks: List[Task] = []
    
    async def start(self):
        """Start the background worker loop."""
        self.running = True
        while self.running:
            task = self.queue.pop()
            if task:
                self.current_task = task
                await self._execute_task(task)
                self.current_task = None
            else:
                # No tasks, wait before checking again
                await asyncio.sleep(30)
    
    async def stop(self):
        """Stop the worker."""
        self.running = False
    
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        print(f"[DAEMON] Executing task: {task.description}")
        
        try:
            if task.task_type == 'research':
                result = await self._do_research(task)
            elif task.task_type == 'organize':
                result = await self._do_organize(task)
            elif task.task_type == 'remind':
                result = await self._do_remind(task)
            elif task.task_type == 'summarize':
                result = await self._do_summarize(task)
            else:
                result = f"Unknown task type: {task.task_type}"
            
            task.status = 'completed'
            task.result = result
            
        except Exception as e:
            task.status = 'failed'
            task.result = str(e)
        
        self.completed_tasks.append(task)
        print(f"[DAEMON] Task completed: {task.task_id} - {task.status}")
    
    async def _do_research(self, task: Task) -> str:
        """Research a topic."""
        query = task.parameters.get('query', task.description)
        
        # Use web search
        from tools.web_search import WebSearch
        searcher = WebSearch()
        results = await searcher.search(query, num_results=5)
        
        # Summarize with LLM
        summary_prompt = f"""Summarize these research findings about "{query}":

{json.dumps(results, indent=2)}

Provide a concise summary with key points."""
        
        summary = await self.engine.model.generate(summary_prompt)
        
        # Store in memory
        self.engine.memory.semantic.store(
            content=f"Research on {query}: {summary}",
            domain="research"
        )
        
        return summary
    
    async def _do_organize(self, task: Task) -> str:
        """Organize files in a directory."""
        path = Path(task.parameters.get('path', '.'))
        # Implementation for file organization
        return f"Organized files in {path}"
    
    async def _do_remind(self, task: Task) -> str:
        """Set up a reminder."""
        message = task.parameters.get('message', task.description)
        # Store as high-priority episodic memory
        return f"Reminder set: {message}"
    
    async def _do_summarize(self, task: Task) -> str:
        """Summarize recent interactions."""
        # Get recent episodes
        episodes = self.engine.memory.episodic.get_recent(limit=20)
        
        # Generate summary
        summary_prompt = f"""Summarize the key topics and outcomes from these {len(episodes)} recent interactions:

{json.dumps([{'input': e.input, 'response': e.response[:200]} for e in episodes], indent=2)}

Focus on: main topics discussed, goals mentioned, preferences expressed."""
        
        summary = await self.engine.model.generate(summary_prompt)
        return summary


class SenterDaemon:
    """
    The main daemon process.
    
    Runs 24/7, manages task queue, handles IPC with CLI.
    """
    
    def __init__(self, genome_path: Path):
        self.genome_path = genome_path
        self.data_dir = genome_path.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.socket_path = self.data_dir / "senter.sock"
        self.pid_file = self.data_dir / "senter.pid"
        
        self.task_queue = TaskQueue(self.data_dir / "task_queue.json")
        self.engine = None  # Lazy init
        self.worker = None
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the daemon."""
        # Write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Initialize Senter engine
        from core.engine import Senter
        self.engine = Senter(self.genome_path)
        
        # Start background worker
        self.worker = BackgroundWorker(self.engine, self.task_queue)
        worker_task = asyncio.create_task(self.worker.start())
        
        # Start IPC server
        self.running = True
        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path)
        )
        
        print(f"[DAEMON] Started. Socket: {self.socket_path}")
        print(f"[DAEMON] PID: {os.getpid()}")
        
        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self):
        """Stop the daemon gracefully."""
        print("[DAEMON] Shutting down...")
        self.running = False
        
        if self.worker:
            await self.worker.stop()
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        if self.engine:
            await self.engine.shutdown()
        
        # Cleanup
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()
        
        print("[DAEMON] Stopped.")
    
    async def _handle_client(self, reader, writer):
        """Handle IPC requests from CLI."""
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                
                request = json.loads(data.decode())
                response = await self._process_request(request)
                
                writer.write(json.dumps(response).encode() + b'\n')
                await writer.drain()
        except Exception as e:
            print(f"[DAEMON] Client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _process_request(self, request: Dict) -> Dict:
        """Process a request from CLI."""
        action = request.get('action')
        
        if action == 'interact':
            # Foreground interaction
            response = await self.engine.interact(request['input'])
            return {
                'status': 'ok',
                'response': response.text,
                'ai_state': {
                    'focus': response.ai_state.focus,
                    'mode': response.ai_state.mode,
                    'trust': response.ai_state.trust_level
                }
            }
        
        elif action == 'add_task':
            # Add background task
            task = Task(
                priority=TaskPriority[request.get('priority', 'NORMAL')].value,
                created_at=datetime.now(),
                task_id=request.get('task_id', str(uuid.uuid4())[:8]),
                task_type=request['task_type'],
                description=request['description'],
                parameters=request.get('parameters', {})
            )
            self.task_queue.add(task)
            return {'status': 'ok', 'task_id': task.task_id}
        
        elif action == 'list_tasks':
            tasks = self.task_queue.list_all()
            return {
                'status': 'ok',
                'tasks': [
                    {'id': t.task_id, 'type': t.task_type, 'desc': t.description}
                    for t in tasks
                ]
            }
        
        elif action == 'completed_tasks':
            return {
                'status': 'ok',
                'tasks': [
                    {
                        'id': t.task_id,
                        'type': t.task_type,
                        'desc': t.description,
                        'result': t.result
                    }
                    for t in self.worker.completed_tasks[-10:]
                ]
            }
        
        elif action == 'status':
            return {
                'status': 'ok',
                'running': self.running,
                'pending_tasks': len(self.task_queue.tasks),
                'current_task': self.worker.current_task.description if self.worker.current_task else None,
                'completed_tasks': len(self.worker.completed_tasks),
                'trust_level': self.engine.trust.level,
                'memory_episodes': len(self.engine.memory.episodic)
            }
        
        elif action == 'shutdown':
            asyncio.create_task(self.stop())
            return {'status': 'ok', 'message': 'Shutting down'}
        
        return {'status': 'error', 'message': f'Unknown action: {action}'}


async def main():
    import sys
    genome_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("genome.yaml")
    daemon = SenterDaemon(genome_path)
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2 CLI Client for Daemon

```python
# senter_client.py - Connects to daemon

"""
CLI client that connects to the Senter daemon.
"""

import asyncio
import json
import sys
from pathlib import Path


class SenterClient:
    """Client that connects to running daemon."""
    
    def __init__(self, socket_path: Path = None):
        self.socket_path = socket_path or Path("data/senter.sock")
        self.reader = None
        self.writer = None
    
    async def connect(self):
        """Connect to daemon."""
        if not self.socket_path.exists():
            raise RuntimeError(
                "Daemon not running. Start with: python -m daemon.senter_daemon"
            )
        
        self.reader, self.writer = await asyncio.open_unix_connection(
            path=str(self.socket_path)
        )
    
    async def disconnect(self):
        """Disconnect from daemon."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
    
    async def send(self, request: dict) -> dict:
        """Send request and get response."""
        self.writer.write(json.dumps(request).encode() + b'\n')
        await self.writer.drain()
        
        data = await self.reader.readline()
        return json.loads(data.decode())
    
    async def interact(self, input_text: str) -> dict:
        """Send interaction to daemon."""
        return await self.send({
            'action': 'interact',
            'input': input_text
        })
    
    async def add_task(self, task_type: str, description: str, 
                       priority: str = 'NORMAL', **params) -> dict:
        """Add background task."""
        return await self.send({
            'action': 'add_task',
            'task_type': task_type,
            'description': description,
            'priority': priority,
            'parameters': params
        })
    
    async def status(self) -> dict:
        """Get daemon status."""
        return await self.send({'action': 'status'})
    
    async def completed_tasks(self) -> dict:
        """Get completed background tasks."""
        return await self.send({'action': 'completed_tasks'})


async def main():
    client = SenterClient()
    
    try:
        await client.connect()
        
        # Show what daemon did while away
        completed = await client.completed_tasks()
        if completed['tasks']:
            print("\n📋 While you were away, I completed:")
            for task in completed['tasks']:
                print(f"  • {task['desc']}")
                if task['result']:
                    print(f"    → {task['result'][:100]}...")
            print()
        
        # Interactive loop
        print("Senter 3.0 (Connected to daemon)")
        print("Type 'quit' to exit, '/task <type> <description>' to add background task\n")
        
        while True:
            try:
                user_input = input("[daemon] You: ").strip()
                
                if user_input.lower() in ('quit', 'exit'):
                    break
                
                if user_input.startswith('/task '):
                    # Add background task
                    parts = user_input[6:].split(' ', 1)
                    if len(parts) == 2:
                        result = await client.add_task(parts[0], parts[1])
                        print(f"  Task added: {result.get('task_id')}")
                    continue
                
                if user_input == '/status':
                    status = await client.status()
                    print(f"  Pending tasks: {status['pending_tasks']}")
                    print(f"  Current task: {status['current_task'] or 'None'}")
                    print(f"  Completed: {status['completed_tasks']}")
                    continue
                
                # Regular interaction
                response = await client.interact(user_input)
                
                if response['status'] == 'ok':
                    ai_state = response['ai_state']
                    print(f"\n[AI State: Mode={ai_state['mode']}, Trust={ai_state['trust']:.2f}]")
                    print(f"Senter: {response['response']}\n")
                else:
                    print(f"Error: {response.get('message')}")
                    
            except KeyboardInterrupt:
                break
                
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### 2.3 Systemd Service File

```ini
# /etc/systemd/user/senter.service

[Unit]
Description=Senter AI Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/senter/daemon/senter_daemon.py /path/to/genome.yaml
ExecStop=/usr/bin/python3 /path/to/senter/senter_client.py --shutdown
Restart=on-failure
RestartSec=5
WorkingDirectory=/path/to/senter

[Install]
WantedBy=default.target
```

**Success Criteria**:
```bash
# Start daemon
python -m daemon.senter_daemon &

# In another terminal, connect with client
python senter_client.py

# Add background task
/task research "Latest developments in AI safety"

# Check status
/status

# Wait for task completion, reconnect
python senter_client.py
# Should show "While you were away..."
```

---

## PART 3: PARALLEL PROCESSING (Days 8-10)

### 3.1 Dual Model Architecture

**Problem**: Single inference thread, can't do background work during interaction.

**Required Implementation**:

```python
# models/parallel.py - NEW FILE

"""
Parallel Model Manager - Runs two inference processes.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                  PARALLEL MODEL MANAGER                      │
│                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │   FOREGROUND MODEL      │  │   BACKGROUND MODEL      │  │
│  │   (High priority)       │  │   (Low priority)        │  │
│  │                         │  │                         │  │
│  │   - User interactions   │  │   - Research tasks      │  │
│  │   - Real-time responses │  │   - Summarization       │  │
│  │   - Preempts background │  │   - Analysis            │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                COORDINATION LAYER                     │   │
│  │   - Preemption signals                                │   │
│  │   - GPU memory management                             │   │
│  │   - Result queuing                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from queue import Empty
import time


class ModelPriority(Enum):
    FOREGROUND = 1  # User interaction - never wait
    BACKGROUND = 2  # Autonomous tasks - can be preempted


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    priority: ModelPriority
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class InferenceResult:
    request_id: str
    text: str
    tokens_generated: int
    time_taken: float
    preempted: bool = False


def _model_worker(
    model_config: Dict,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    preempt_event: mp.Event,
    worker_id: str
):
    """
    Worker process that runs model inference.
    
    Runs in separate process to enable true parallelism.
    """
    # Initialize model in this process
    from models.gguf import GGUFModel
    from models.ollama import OllamaModel
    
    model_type = model_config.get('type', 'ollama')
    if model_type == 'gguf':
        model = GGUFModel(model_config)
    else:
        model = OllamaModel(model_config)
    
    print(f"[{worker_id}] Model loaded")
    
    while True:
        try:
            # Get request (block with timeout)
            request: InferenceRequest = request_queue.get(timeout=1.0)
            
            if request is None:  # Shutdown signal
                break
            
            start_time = time.time()
            preempted = False
            
            # For background tasks, check preemption periodically
            if request.priority == ModelPriority.BACKGROUND:
                # Generate with preemption checking
                text = ""
                for chunk in model.generate_stream(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    if preempt_event.is_set():
                        preempted = True
                        preempt_event.clear()
                        break
                    text += chunk
            else:
                # Foreground: generate without interruption
                text = model.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
            
            result = InferenceResult(
                request_id=request.request_id,
                text=text,
                tokens_generated=len(text.split()),
                time_taken=time.time() - start_time,
                preempted=preempted
            )
            
            result_queue.put(result)
            
        except Empty:
            continue
        except Exception as e:
            print(f"[{worker_id}] Error: {e}")


class ParallelModelManager:
    """
    Manages two model inference processes.
    
    Foreground: User interactions, high priority
    Background: Autonomous tasks, can be preempted
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Queues for each worker
        self.fg_request_queue = mp.Queue()
        self.fg_result_queue = mp.Queue()
        self.bg_request_queue = mp.Queue()
        self.bg_result_queue = mp.Queue()
        
        # Preemption event for background
        self.bg_preempt = mp.Event()
        
        # Pending results
        self._pending: Dict[str, asyncio.Future] = {}
        
        # Workers
        self.fg_worker = None
        self.bg_worker = None
    
    def start(self):
        """Start both worker processes."""
        fg_config = self.config.get('foreground', self.config.get('primary', {}))
        bg_config = self.config.get('background', fg_config)  # Default to same model
        
        # Foreground worker
        self.fg_worker = mp.Process(
            target=_model_worker,
            args=(
                fg_config,
                self.fg_request_queue,
                self.fg_result_queue,
                mp.Event(),  # Foreground never preempted
                "FOREGROUND"
            )
        )
        self.fg_worker.start()
        
        # Background worker
        self.bg_worker = mp.Process(
            target=_model_worker,
            args=(
                bg_config,
                self.bg_request_queue,
                self.bg_result_queue,
                self.bg_preempt,
                "BACKGROUND"
            )
        )
        self.bg_worker.start()
        
        # Start result collector
        asyncio.create_task(self._collect_results())
    
    def stop(self):
        """Stop worker processes."""
        self.fg_request_queue.put(None)
        self.bg_request_queue.put(None)
        
        self.fg_worker.join(timeout=5)
        self.bg_worker.join(timeout=5)
    
    async def generate(
        self,
        prompt: str,
        priority: ModelPriority = ModelPriority.FOREGROUND,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text with specified priority.
        
        Foreground requests preempt background work.
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Create future for result
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        
        # Send to appropriate queue
        if priority == ModelPriority.FOREGROUND:
            # Preempt any background work
            self.bg_preempt.set()
            self.fg_request_queue.put(request)
        else:
            self.bg_request_queue.put(request)
        
        # Wait for result
        result = await future
        return result.text
    
    async def _collect_results(self):
        """Collect results from both workers."""
        while True:
            # Check foreground results
            try:
                while True:
                    result = self.fg_result_queue.get_nowait()
                    if result.request_id in self._pending:
                        self._pending[result.request_id].set_result(result)
                        del self._pending[result.request_id]
            except Empty:
                pass
            
            # Check background results
            try:
                while True:
                    result = self.bg_result_queue.get_nowait()
                    if result.request_id in self._pending:
                        self._pending[result.request_id].set_result(result)
                        del self._pending[result.request_id]
            except Empty:
                pass
            
            await asyncio.sleep(0.01)
```

**Usage in Engine**:
```python
# core/engine.py - Update to use parallel models

class Senter:
    def _init_model_layer(self) -> None:
        # Use parallel model manager instead of single model
        from models.parallel import ParallelModelManager, ModelPriority
        
        self.models = ParallelModelManager(self.genome.get('models', {}))
        self.models.start()
        
        # Convenience methods
        self.generate_foreground = lambda p: self.models.generate(p, ModelPriority.FOREGROUND)
        self.generate_background = lambda p: self.models.generate(p, ModelPriority.BACKGROUND)
```

**Success Criteria**:
```bash
# Test parallel execution
python -c "
import asyncio
from models.parallel import ParallelModelManager, ModelPriority

async def test():
    mgr = ParallelModelManager({'type': 'ollama', 'model': 'llama3.2'})
    mgr.start()
    
    # Start background task
    bg_task = asyncio.create_task(
        mgr.generate('Write a long essay about AI', ModelPriority.BACKGROUND)
    )
    
    # Immediately do foreground task (should preempt)
    fg_result = await mgr.generate('Hello', ModelPriority.FOREGROUND)
    print(f'Foreground done: {fg_result[:50]}')
    
    # Background may be preempted
    bg_result = await bg_task
    print(f'Background done (may be partial): {len(bg_result)} chars')
    
    mgr.stop()

asyncio.run(test())
"
```

---

## PART 4: VOICE & VISION INTERFACE (Days 11-15)

### 4.1 Speech-to-Text with Whisper

```python
# interface/voice.py - NEW FILE

"""
Voice interface using local Whisper model.
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import wave
import tempfile


class VoiceInterface:
    """
    Local speech-to-text using Whisper.
    
    No wake word needed - activated by gaze detection.
    """
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.is_listening = False
        self.audio_buffer = []
        
        # Voice activity detection settings
        self.vad_threshold = 0.02  # Energy threshold
        self.silence_duration = 1.5  # Seconds of silence to end
    
    def load(self):
        """Load Whisper model."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            print(f"[VOICE] Whisper {self.model_size} loaded")
        except ImportError:
            print("[VOICE] Install: pip install openai-whisper")
            raise
    
    async def start_listening(self, on_transcript: Callable[[str], None]):
        """
        Start listening for speech.
        
        Uses voice activity detection to determine when user is speaking.
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("[VOICE] Install: pip install sounddevice")
            raise
        
        self.is_listening = True
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(sample_rate * chunk_duration)
        
        silence_chunks = 0
        max_silence_chunks = int(self.silence_duration / chunk_duration)
        recording = False
        audio_data = []
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_chunks, recording, audio_data
            
            # Calculate energy
            energy = np.sqrt(np.mean(indata ** 2))
            
            if energy > self.vad_threshold:
                # Voice detected
                if not recording:
                    print("[VOICE] Recording started...")
                recording = True
                silence_chunks = 0
                audio_data.append(indata.copy())
            elif recording:
                # Silence during recording
                silence_chunks += 1
                audio_data.append(indata.copy())
                
                if silence_chunks >= max_silence_chunks:
                    # End of utterance
                    recording = False
                    if audio_data:
                        # Process in background
                        asyncio.create_task(
                            self._process_audio(
                                np.concatenate(audio_data),
                                sample_rate,
                                on_transcript
                            )
                        )
                    audio_data = []
                    silence_chunks = 0
        
        # Start audio stream
        stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            callback=audio_callback
        )
        
        with stream:
            while self.is_listening:
                await asyncio.sleep(0.1)
    
    def stop_listening(self):
        """Stop listening."""
        self.is_listening = False
    
    async def _process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        callback: Callable[[str], None]
    ):
        """Process recorded audio through Whisper."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            
            # Write WAV
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        try:
            # Transcribe
            result = self.model.transcribe(
                temp_path,
                language="en",
                fp16=False
            )
            
            text = result["text"].strip()
            if text:
                print(f"[VOICE] Heard: {text}")
                callback(text)
                
        finally:
            Path(temp_path).unlink()
    
    async def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file."""
        result = self.model.transcribe(str(audio_path), language="en")
        return result["text"].strip()
```

### 4.2 Gaze Detection for Activation

```python
# interface/gaze.py - NEW FILE

"""
Gaze detection for wake-word-free activation.

Look at camera → Senter starts listening.
"""

import asyncio
import cv2
import numpy as np
from typing import Callable, Optional


class GazeDetector:
    """
    Detects when user is looking at camera.
    
    Uses MediaPipe for face mesh and gaze estimation.
    """
    
    def __init__(self):
        self.running = False
        self.cap = None
        self.face_mesh = None
        
        # Gaze tracking state
        self.looking_at_camera = False
        self.gaze_start_time = None
        self.activation_threshold = 0.5  # Seconds of sustained gaze
    
    def load(self):
        """Load face detection model."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[GAZE] MediaPipe loaded")
        except ImportError:
            print("[GAZE] Install: pip install mediapipe")
            raise
    
    async def start(
        self,
        on_gaze_start: Callable[[], None],
        on_gaze_end: Callable[[], None],
        camera_id: int = 0
    ):
        """
        Start gaze detection.
        
        Calls on_gaze_start when user looks at camera,
        on_gaze_end when they look away.
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.running = True
        
        import time
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            is_looking = False
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get eye landmarks for gaze estimation
                    is_looking = self._estimate_gaze(face_landmarks, frame.shape)
            
            # State machine
            current_time = time.time()
            
            if is_looking and not self.looking_at_camera:
                # Started looking
                self.gaze_start_time = current_time
                
            elif is_looking and self.looking_at_camera:
                # Continue looking - check if past threshold
                if current_time - self.gaze_start_time >= self.activation_threshold:
                    if not self._activated:
                        self._activated = True
                        on_gaze_start()
                        
            elif not is_looking and self.looking_at_camera:
                # Stopped looking
                if self._activated:
                    self._activated = False
                    on_gaze_end()
                self.gaze_start_time = None
            
            self.looking_at_camera = is_looking
            
            # Small delay to not hog CPU
            await asyncio.sleep(0.033)  # ~30 FPS
    
    def _estimate_gaze(self, landmarks, frame_shape) -> bool:
        """
        Estimate if user is looking at camera.
        
        Uses iris position relative to eye corners.
        """
        h, w = frame_shape[:2]
        
        # Left eye landmarks
        left_eye_inner = landmarks.landmark[133]
        left_eye_outer = landmarks.landmark[33]
        left_iris = landmarks.landmark[468]  # Iris center
        
        # Right eye landmarks
        right_eye_inner = landmarks.landmark[362]
        right_eye_outer = landmarks.landmark[263]
        right_iris = landmarks.landmark[473]  # Iris center
        
        # Calculate iris position relative to eye corners (0 = outer, 1 = inner)
        def iris_ratio(iris, inner, outer):
            eye_width = abs(inner.x - outer.x)
            if eye_width < 0.01:
                return 0.5
            return (iris.x - outer.x) / (inner.x - outer.x)
        
        left_ratio = iris_ratio(left_iris, left_eye_inner, left_eye_outer)
        right_ratio = iris_ratio(right_iris, right_eye_inner, right_eye_outer)
        
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Looking at camera when iris is roughly centered (0.35-0.65)
        return 0.35 <= avg_ratio <= 0.65
    
    def stop(self):
        """Stop detection."""
        self.running = False
        if self.cap:
            self.cap.release()
```

### 4.3 Integrated Multimodal Interface

```python
# interface/multimodal.py - NEW FILE

"""
Integrated multimodal interface.

Gaze + Voice = Seamless interaction without wake words.
"""

import asyncio
from typing import Callable


class MultimodalInterface:
    """
    Combines gaze detection + voice input.
    
    Workflow:
    1. User looks at camera
    2. After 0.5s of sustained gaze → start listening
    3. User speaks
    4. User looks away → stop listening, process
    """
    
    def __init__(self):
        from interface.gaze import GazeDetector
        from interface.voice import VoiceInterface
        
        self.gaze = GazeDetector()
        self.voice = VoiceInterface()
        
        self.on_input: Callable[[str], None] = None
        self.is_active = False
    
    def load(self):
        """Load all models."""
        self.gaze.load()
        self.voice.load()
    
    async def start(self, on_input: Callable[[str], None]):
        """
        Start multimodal interface.
        
        on_input is called with transcribed text when user finishes speaking.
        """
        self.on_input = on_input
        
        # Start gaze detection
        gaze_task = asyncio.create_task(
            self.gaze.start(
                on_gaze_start=self._on_gaze_start,
                on_gaze_end=self._on_gaze_end
            )
        )
        
        await gaze_task
    
    def _on_gaze_start(self):
        """Called when user starts looking at camera."""
        print("[MULTIMODAL] 👀 Gaze detected - listening...")
        self.is_active = True
        
        # Start voice input
        asyncio.create_task(
            self.voice.start_listening(self._on_transcript)
        )
    
    def _on_gaze_end(self):
        """Called when user looks away."""
        print("[MULTIMODAL] 👀 Gaze ended")
        self.is_active = False
        self.voice.stop_listening()
    
    def _on_transcript(self, text: str):
        """Called when voice is transcribed."""
        if self.on_input:
            self.on_input(text)
    
    def stop(self):
        """Stop all interfaces."""
        self.gaze.stop()
        self.voice.stop_listening()
```

**Success Criteria**:
```bash
# Test voice
python -c "
import asyncio
from interface.voice import VoiceInterface

voice = VoiceInterface('base')
voice.load()

async def test():
    def on_text(t):
        print(f'Got: {t}')
    
    await voice.start_listening(on_text)

asyncio.run(test())
"

# Test gaze
python -c "
import asyncio
from interface.gaze import GazeDetector

gaze = GazeDetector()
gaze.load()

async def test():
    def on_start():
        print('Looking at camera!')
    def on_end():
        print('Looked away')
    
    await gaze.start(on_start, on_end)

asyncio.run(test())
"
```

---

## PART 5: PROACTIVE INTELLIGENCE (Days 16-20)

### 5.1 Goal Detection from Conversations

```python
# intelligence/goals.py - NEW FILE

"""
Automatic goal detection from conversations.

Learns user goals without explicit statements.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import re


@dataclass
class Goal:
    id: str
    description: str
    category: str  # career, health, learning, project, personal
    confidence: float  # How confident we are this is a real goal
    evidence: List[str]  # Conversation snippets that suggest this goal
    created_at: datetime
    last_mentioned: datetime
    progress: float  # 0-1 estimated progress
    status: str  # active, completed, abandoned


class GoalDetector:
    """
    Detects goals from conversation patterns.
    
    Looks for:
    - Explicit goals: "I want to...", "My goal is..."
    - Implicit goals: Repeated topics, frustrations, questions
    - Project references: "Working on...", "Building..."
    """
    
    EXPLICIT_PATTERNS = [
        r"(?:i want to|i'd like to|i need to|my goal is|i'm trying to|i hope to) (.+?)(?:\.|,|$)",
        r"(?:i'm working on|i'm building|i'm learning|i'm studying) (.+?)(?:\.|,|$)",
        r"(?:i have to|i must|i should) (.+?)(?:\.|,|$)",
    ]
    
    CATEGORY_KEYWORDS = {
        'career': ['job', 'work', 'career', 'promotion', 'salary', 'interview', 'resume'],
        'health': ['exercise', 'diet', 'weight', 'sleep', 'health', 'gym', 'run'],
        'learning': ['learn', 'study', 'course', 'book', 'understand', 'skill'],
        'project': ['build', 'create', 'develop', 'launch', 'ship', 'code', 'app'],
        'personal': ['relationship', 'family', 'friend', 'hobby', 'travel'],
    }
    
    def __init__(self, memory):
        self.memory = memory
        self.goals: Dict[str, Goal] = {}
        self._load_goals()
    
    def _load_goals(self):
        """Load persisted goals."""
        stored = self.memory.semantic.get_by_domain('goals')
        for item in stored:
            goal_data = item.get('content', {})
            if isinstance(goal_data, dict):
                self.goals[goal_data.get('id')] = Goal(**goal_data)
    
    def analyze_interaction(self, input_text: str, response_text: str) -> List[Goal]:
        """
        Analyze an interaction for goal signals.
        
        Returns new or updated goals.
        """
        input_lower = input_text.lower()
        new_goals = []
        
        # Check explicit patterns
        for pattern in self.EXPLICIT_PATTERNS:
            matches = re.findall(pattern, input_lower)
            for match in matches:
                goal = self._create_or_update_goal(
                    description=match.strip(),
                    evidence=input_text,
                    confidence=0.8  # Explicit statement = high confidence
                )
                if goal:
                    new_goals.append(goal)
        
        # Check for topic repetition (implicit goals)
        recent_topics = self._get_recent_topics()
        current_topics = self._extract_topics(input_text)
        
        for topic in current_topics:
            if topic in recent_topics and recent_topics[topic] >= 3:
                # Mentioned 3+ times = likely a goal
                goal = self._create_or_update_goal(
                    description=f"Focus on {topic}",
                    evidence=input_text,
                    confidence=0.5 + (0.1 * min(recent_topics[topic], 5))
                )
                if goal:
                    new_goals.append(goal)
        
        # Check for frustrations (implicit blocked goals)
        frustration_patterns = [
            r"(?:frustrated|annoyed|stuck|can't|won't work|failing) (?:with|at|on) (.+?)(?:\.|,|$)"
        ]
        for pattern in frustration_patterns:
            matches = re.findall(pattern, input_lower)
            for match in matches:
                goal = self._create_or_update_goal(
                    description=f"Resolve issues with {match.strip()}",
                    evidence=input_text,
                    confidence=0.6
                )
                if goal:
                    new_goals.append(goal)
        
        return new_goals
    
    def _create_or_update_goal(
        self,
        description: str,
        evidence: str,
        confidence: float
    ) -> Optional[Goal]:
        """Create new goal or update existing one."""
        import uuid
        
        # Check for similar existing goal
        for goal in self.goals.values():
            if self._similarity(goal.description, description) > 0.7:
                # Update existing
                goal.confidence = min(1.0, goal.confidence + 0.1)
                goal.evidence.append(evidence)
                goal.last_mentioned = datetime.now()
                self._persist_goal(goal)
                return goal
        
        # Create new
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            description=description,
            category=self._categorize(description),
            confidence=confidence,
            evidence=[evidence],
            created_at=datetime.now(),
            last_mentioned=datetime.now(),
            progress=0.0,
            status='active'
        )
        
        self.goals[goal.id] = goal
        self._persist_goal(goal)
        return goal
    
    def _categorize(self, description: str) -> str:
        """Categorize a goal."""
        desc_lower = description.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return 'personal'
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Simple similarity measure."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0
    
    def _get_recent_topics(self) -> Dict[str, int]:
        """Get topic frequency from recent conversations."""
        episodes = self.memory.episodic.get_recent(limit=50)
        topics = {}
        for ep in episodes:
            for topic in self._extract_topics(ep.input):
                topics[topic] = topics.get(topic, 0) + 1
        return topics
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple: extract nouns/noun phrases
        import re
        words = text.lower().split()
        # Filter to meaningful words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'be', 'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'this', 'from', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'when', 'can', 'no', 'just', 'into', 'your', 'some', 'could', 'them', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'also', 'back', 'after', 'use', 'how', 'our', 'well', 'way', 'want', 'she', 'him', 'his', 'her', 'he', 'we', 'me', 'you', 'i'}
        return [w.strip('.,!?') for w in words if w not in stopwords and len(w) > 3]
    
    def _persist_goal(self, goal: Goal):
        """Save goal to memory."""
        self.memory.semantic.store(
            content=goal.__dict__,
            domain='goals'
        )
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by confidence."""
        active = [g for g in self.goals.values() if g.status == 'active']
        return sorted(active, key=lambda g: g.confidence, reverse=True)
    
    def suggest_actions(self) -> List[Dict[str, Any]]:
        """Suggest actions based on goals."""
        suggestions = []
        
        for goal in self.get_active_goals()[:5]:  # Top 5 goals
            if goal.category == 'learning':
                suggestions.append({
                    'goal': goal,
                    'action': 'research',
                    'description': f"Research resources for: {goal.description}"
                })
            elif goal.category == 'project':
                suggestions.append({
                    'goal': goal,
                    'action': 'plan',
                    'description': f"Create action plan for: {goal.description}"
                })
            elif goal.progress < 0.2:
                suggestions.append({
                    'goal': goal,
                    'action': 'start',
                    'description': f"Get started on: {goal.description}"
                })
        
        return suggestions
```

### 5.2 Proactive Suggestion Engine

```python
# intelligence/proactive.py - NEW FILE

"""
Proactive suggestion engine.

Senter doesn't just respond - it anticipates and suggests.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class ProactiveSuggestionEngine:
    """
    Generates proactive suggestions based on:
    - User goals
    - Conversation patterns
    - Time-based triggers
    - Context awareness
    """
    
    def __init__(self, senter_engine):
        self.engine = senter_engine
        self.last_suggestions: Dict[str, datetime] = {}
        self.suggestion_cooldown = timedelta(hours=4)
    
    async def generate_suggestions(self) -> List[Dict[str, Any]]:
        """Generate current suggestions."""
        suggestions = []
        
        # Goal-based suggestions
        goal_suggestions = await self._goal_based_suggestions()
        suggestions.extend(goal_suggestions)
        
        # Time-based suggestions
        time_suggestions = await self._time_based_suggestions()
        suggestions.extend(time_suggestions)
        
        # Pattern-based suggestions
        pattern_suggestions = await self._pattern_based_suggestions()
        suggestions.extend(pattern_suggestions)
        
        # Filter by cooldown and trust level
        filtered = self._filter_suggestions(suggestions)
        
        return filtered[:3]  # Return top 3
    
    async def _goal_based_suggestions(self) -> List[Dict]:
        """Suggestions based on user goals."""
        suggestions = []
        
        # Get active goals
        goals = self.engine.goal_detector.get_active_goals()
        
        for goal in goals[:3]:  # Top 3 goals
            # Check if we haven't suggested this recently
            if self._should_suggest(f"goal_{goal.id}"):
                # Generate specific suggestion
                if goal.progress < 0.1:
                    suggestions.append({
                        'type': 'goal_start',
                        'id': f"goal_{goal.id}",
                        'priority': goal.confidence,
                        'title': f"Get started on: {goal.description}",
                        'action': f"Would you like me to help break down '{goal.description}' into actionable steps?",
                        'task_type': 'plan',
                        'goal': goal
                    })
                elif goal.category == 'learning':
                    suggestions.append({
                        'type': 'goal_research',
                        'id': f"goal_{goal.id}",
                        'priority': goal.confidence * 0.9,
                        'title': f"Research opportunity: {goal.description}",
                        'action': f"I can research the latest resources about {goal.description}. Want me to do that in the background?",
                        'task_type': 'research',
                        'goal': goal
                    })
        
        return suggestions
    
    async def _time_based_suggestions(self) -> List[Dict]:
        """Time-sensitive suggestions."""
        suggestions = []
        now = datetime.now()
        
        # Morning reflection (8-10 AM)
        if 8 <= now.hour <= 10 and self._should_suggest('morning_reflection'):
            suggestions.append({
                'type': 'time_morning',
                'id': 'morning_reflection',
                'priority': 0.7,
                'title': "Morning planning",
                'action': "Good morning! Would you like me to summarize what we worked on yesterday and suggest priorities for today?",
                'task_type': 'summarize'
            })
        
        # Evening review (6-8 PM)
        if 18 <= now.hour <= 20 and self._should_suggest('evening_review'):
            suggestions.append({
                'type': 'time_evening',
                'id': 'evening_review',
                'priority': 0.6,
                'title': "Daily review",
                'action': "End of day! Want me to compile what we accomplished today?",
                'task_type': 'summarize'
            })
        
        # Weekly review (Sunday)
        if now.weekday() == 6 and self._should_suggest('weekly_review'):
            suggestions.append({
                'type': 'time_weekly',
                'id': 'weekly_review',
                'priority': 0.8,
                'title': "Weekly review",
                'action': "It's Sunday - good time for a weekly review. Shall I analyze this week's progress on your goals?",
                'task_type': 'summarize'
            })
        
        return suggestions
    
    async def _pattern_based_suggestions(self) -> List[Dict]:
        """Suggestions based on conversation patterns."""
        suggestions = []
        
        # Check for repeated questions
        recent_questions = self._get_repeated_questions()
        for question, count in recent_questions.items():
            if count >= 3 and self._should_suggest(f"faq_{hash(question)}"):
                suggestions.append({
                    'type': 'pattern_faq',
                    'id': f"faq_{hash(question)}",
                    'priority': 0.5,
                    'title': "Frequently asked",
                    'action': f"I notice you've asked about '{question[:50]}...' several times. Want me to create a comprehensive reference?",
                    'task_type': 'research'
                })
        
        # Check for stalled topics
        stalled = self._get_stalled_topics()
        for topic in stalled:
            if self._should_suggest(f"stalled_{topic}"):
                suggestions.append({
                    'type': 'pattern_stalled',
                    'id': f"stalled_{topic}",
                    'priority': 0.4,
                    'title': "Revisit topic",
                    'action': f"We haven't discussed '{topic}' in a while. Want to pick up where we left off?",
                    'task_type': 'recall'
                })
        
        return suggestions
    
    def _should_suggest(self, suggestion_id: str) -> bool:
        """Check if suggestion is past cooldown."""
        if suggestion_id not in self.last_suggestions:
            return True
        
        elapsed = datetime.now() - self.last_suggestions[suggestion_id]
        return elapsed >= self.suggestion_cooldown
    
    def mark_suggested(self, suggestion_id: str):
        """Mark suggestion as shown."""
        self.last_suggestions[suggestion_id] = datetime.now()
    
    def _filter_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Filter by trust level and cooldown."""
        trust = self.engine.trust.level
        
        # Only show proactive suggestions at sufficient trust
        if trust < 0.6:
            return []  # Not enough trust for proactive behavior
        
        # Higher trust = more suggestions
        max_suggestions = 1 if trust < 0.7 else (2 if trust < 0.8 else 3)
        
        # Sort by priority
        sorted_suggestions = sorted(suggestions, key=lambda s: s['priority'], reverse=True)
        
        return sorted_suggestions[:max_suggestions]
    
    def _get_repeated_questions(self) -> Dict[str, int]:
        """Find questions asked multiple times."""
        episodes = self.engine.memory.episodic.get_recent(limit=100)
        questions = {}
        
        for ep in episodes:
            if '?' in ep.input:
                # Normalize question
                q = ep.input.lower().strip()
                questions[q] = questions.get(q, 0) + 1
        
        return {q: c for q, c in questions.items() if c >= 3}
    
    def _get_stalled_topics(self) -> List[str]:
        """Find topics that were discussed but dropped."""
        # Get topics from 2+ weeks ago that haven't been mentioned since
        old_episodes = self.engine.memory.episodic.get_by_date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now() - timedelta(days=14)
        )
        
        recent_episodes = self.engine.memory.episodic.get_recent(limit=50)
        recent_topics = set()
        for ep in recent_episodes:
            recent_topics.update(ep.input.lower().split())
        
        stalled = []
        for ep in old_episodes:
            topics = set(ep.input.lower().split()) - recent_topics
            for topic in topics:
                if len(topic) > 5 and topic not in ['about', 'would', 'could', 'should']:
                    stalled.append(topic)
        
        return list(set(stalled))[:5]
```

---

## PART 6: INTEGRATION & POLISH (Days 21-25)

### 6.1 Updated Main Entry Point

```python
# senter.py - COMPLETE REWRITE

"""
Senter 3.0 - Your 24/7 AI Assistant

Usage:
  python senter.py              # Interactive CLI (connects to daemon)
  python senter.py --daemon     # Start daemon
  python senter.py --voice      # Voice + Gaze mode
  python senter.py --status     # Show daemon status
"""

import asyncio
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Senter 3.0")
    parser.add_argument('--daemon', action='store_true', help='Start as daemon')
    parser.add_argument('--voice', action='store_true', help='Enable voice + gaze')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--version', action='store_true', help='Show version')
    parser.add_argument('--genome', type=Path, default=Path('genome.yaml'), help='Genome file')
    
    args = parser.parse_args()
    
    if args.version:
        print("Senter 3.0.0")
        return
    
    if args.daemon:
        # Start daemon mode
        from daemon.senter_daemon import SenterDaemon
        daemon = SenterDaemon(args.genome)
        asyncio.run(daemon.start())
        
    elif args.voice:
        # Voice + Gaze mode
        asyncio.run(run_multimodal(args.genome))
        
    elif args.status:
        # Show status
        asyncio.run(show_status())
        
    else:
        # Interactive CLI (connects to daemon or runs standalone)
        asyncio.run(run_cli(args.genome))


async def run_cli(genome_path: Path):
    """Run interactive CLI."""
    from senter_client import SenterClient
    
    # Try to connect to daemon
    client = SenterClient()
    try:
        await client.connect()
        print("Connected to Senter daemon")
        use_daemon = True
    except RuntimeError:
        # No daemon, run standalone
        print("No daemon running, starting standalone...")
        from core.engine import Senter
        engine = Senter(genome_path)
        use_daemon = False
    
    # Show what happened while away
    if use_daemon:
        completed = await client.completed_tasks()
        if completed.get('tasks'):
            print("\n📋 While you were away:")
            for task in completed['tasks']:
                print(f"  ✓ {task['desc']}")
            print()
    
    # Show proactive suggestions
    if not use_daemon:
        suggestions = await engine.proactive.generate_suggestions()
        if suggestions:
            print("\n💡 Suggestions:")
            for s in suggestions:
                print(f"  • {s['title']}")
            print()
    
    print("Senter 3.0 ready. Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            
            if not user_input:
                continue
            
            # Handle slash commands
            if user_input.startswith('/'):
                await handle_command(user_input, client if use_daemon else engine)
                continue
            
            # Regular interaction
            if use_daemon:
                response = await client.interact(user_input)
                ai_state = response.get('ai_state', {})
                print(f"\n[Mode: {ai_state.get('mode', '?')}, Trust: {ai_state.get('trust', 0):.2f}]")
                print(f"Senter: {response['response']}\n")
            else:
                response = await engine.interact(user_input)
                print(f"\n[Mode: {response.ai_state.mode}, Trust: {response.ai_state.trust_level:.2f}]")
                print(f"Senter: {response.text}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    if use_daemon:
        await client.disconnect()
    else:
        await engine.shutdown()


async def run_multimodal(genome_path: Path):
    """Run with voice + gaze activation."""
    from interface.multimodal import MultimodalInterface
    from senter_client import SenterClient
    
    print("Loading voice + gaze interface...")
    
    interface = MultimodalInterface()
    interface.load()
    
    client = SenterClient()
    await client.connect()
    
    async def on_voice_input(text: str):
        print(f"\n🎤 You said: {text}")
        response = await client.interact(text)
        print(f"🤖 Senter: {response['response']}\n")
        # TODO: Text-to-speech response
    
    print("\n👀 Look at your camera and speak to interact")
    print("Press Ctrl+C to exit\n")
    
    try:
        await interface.start(on_voice_input)
    except KeyboardInterrupt:
        pass
    finally:
        interface.stop()
        await client.disconnect()


async def show_status():
    """Show daemon status."""
    from senter_client import SenterClient
    
    client = SenterClient()
    try:
        await client.connect()
        status = await client.status()
        
        print("\n📊 Senter Status")
        print("─" * 40)
        print(f"  Running: {'✓' if status['running'] else '✗'}")
        print(f"  Trust Level: {status.get('trust_level', 0):.2f}")
        print(f"  Memory Episodes: {status.get('memory_episodes', 0)}")
        print(f"  Pending Tasks: {status['pending_tasks']}")
        print(f"  Current Task: {status['current_task'] or 'None'}")
        print(f"  Completed Tasks: {status['completed_tasks']}")
        print()
        
        await client.disconnect()
        
    except RuntimeError:
        print("Daemon not running")


async def handle_command(cmd: str, client_or_engine):
    """Handle slash commands."""
    parts = cmd.split()
    command = parts[0].lower()
    
    if command == '/help':
        print("""
Commands:
  /status       - System status
  /memory       - Memory statistics
  /trust        - Trust level details
  /goals        - Active goals
  /tasks        - Background tasks
  /task <type> <desc> - Add background task
  /suggest      - Get proactive suggestions
  /evolution    - Evolution status
""")
    
    elif command == '/status':
        await show_status()
    
    elif command == '/task' and len(parts) >= 3:
        task_type = parts[1]
        description = ' '.join(parts[2:])
        if hasattr(client_or_engine, 'add_task'):
            result = await client_or_engine.add_task(task_type, description)
            print(f"  Task added: {result.get('task_id')}")
        else:
            print("  Background tasks require daemon mode")
    
    elif command == '/goals':
        if hasattr(client_or_engine, 'goal_detector'):
            goals = client_or_engine.goal_detector.get_active_goals()
            print("\n🎯 Active Goals:")
            for g in goals[:5]:
                print(f"  • [{g.confidence:.0%}] {g.description}")
            print()
    
    elif command == '/evolution':
        if hasattr(client_or_engine, 'mutations'):
            summary = client_or_engine.mutations.get_evolution_summary()
            print(f"\n🧬 Evolution Status:")
            print(f"  Total mutations: {summary['total']}")
            print(f"  Successful: {summary['successful']}")
            print(f"  Rolled back: {summary['rolled_back']}")
            if summary.get('recent_mutations'):
                print("  Recent:")
                for m in summary['recent_mutations'][-3:]:
                    status = '✓' if m['success'] else '✗'
                    print(f"    {status} {m['type']}: {m['reason'][:40]}...")
            print()
    
    else:
        print(f"  Unknown command: {command}")


if __name__ == "__main__":
    main()
```

---

## PART 7: SUCCESS CRITERIA

### Validation Checklist

#### Foundation (Part 1)
- [ ] Semantic search uses embeddings, not LIKE
- [ ] Memory retrieval returns semantically similar results
- [ ] Evolution actually modifies genome.yaml
- [ ] Mutations are applied, tested, and selected

#### Daemon Mode (Part 2)
- [ ] `python senter.py --daemon` starts persistent service
- [ ] CLI connects to running daemon via Unix socket
- [ ] Tasks execute in background while CLI is closed
- [ ] "While you were away" shows on reconnect

#### Parallel Processing (Part 3)
- [ ] Two model processes run simultaneously
- [ ] Foreground requests preempt background work
- [ ] Background tasks complete without blocking chat

#### Voice & Vision (Part 4)
- [ ] Whisper transcribes speech accurately
- [ ] Gaze detection activates on sustained look
- [ ] Combined mode: look → speak → response

#### Proactive Intelligence (Part 5)
- [ ] Goals detected from conversation patterns
- [ ] Proactive suggestions generated based on trust
- [ ] Time-based suggestions (morning/evening)

#### Integration (Part 6)
- [ ] All modes work: CLI, daemon, voice
- [ ] Slash commands work
- [ ] Error handling is robust

### The Ultimate Test

Run Senter for one week:

```
Day 1:
- Start daemon: python senter.py --daemon
- Have 10 conversations about various projects

Day 2:
- Reconnect - should show background task completions
- Ask "what did we talk about yesterday?"
- Add research task: /task research "AI safety papers"

Day 3:
- Check evolution: /evolution
- Verify genome.yaml has been modified
- Trust should be > 0.55

Day 4:
- Try voice mode: python senter.py --voice
- Look at camera, speak, get response

Day 5-7:
- Continue using
- By Day 7:
  - Trust > 0.7
  - Goals detected automatically
  - Proactive suggestions appearing
  - Multiple successful mutations
  - Memory recall accurate
```

---

## SUMMARY

### What This Build Document Delivers

| Phase | Capability | Vision Requirement |
|-------|------------|-------------------|
| **Part 1** | Real semantic search + real evolution | "Learns from interactions" |
| **Part 2** | Daemon + task queue | "Works 24/7" |
| **Part 3** | Parallel inference | "Background research" |
| **Part 4** | Voice + gaze | "No wake word" |
| **Part 5** | Goal detection + proactive | "Anticipates needs" |
| **Part 6** | Integration | "Seamless experience" |

### The End State

When all parts are implemented, Senter will:

1. **Run 24/7** as a daemon, not just when CLI is open
2. **Execute tasks autonomously** while you're away
3. **Learn your goals** from conversation patterns
4. **Proactively suggest** actions based on trust level
5. **Respond to voice** activated by looking at camera
6. **Actually evolve** its configuration based on what works
7. **Remember semantically** not just by keywords

This is the difference between "implemented" and "phenomenal."

---

**Priority Order for Implementation:**

1. **Part 1** (Foundation) - Without this, nothing else matters
2. **Part 2** (Daemon) - The "24/7" differentiator
3. **Part 5** (Proactive) - Makes it feel intelligent
4. **Part 3** (Parallel) - Enables background work
5. **Part 4** (Voice) - Convenience, not critical
6. **Part 6** (Integration) - Polish

Start with Part 1. Each part builds on the previous.
