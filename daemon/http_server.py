"""
Senter HTTP API Server - Exposes Senter engine via HTTP for UI integration.

Runs alongside the daemon, providing a REST API for the Electron frontend.
"""

from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from aiohttp import web
from aiohttp.web import Request, Response, StreamResponse
import aiohttp_cors


class SenterHTTPServer:
    """HTTP API server for Senter UI integration."""

    def __init__(self, genome_path: Path, host: str = "127.0.0.1", port: int = 8420):
        self.genome_path = genome_path
        self.host = host
        self.port = port
        self.engine = None
        self.app = web.Application()
        self.runner = None
        self.start_time = datetime.now()

        # Setup CORS for Electron app
        self.cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "OPTIONS"]
            )
        })

        # Setup routes (after CORS setup)
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes with CORS."""
        # Add routes and enable CORS for each
        routes = [
            ("POST", "/api/chat", self.handle_chat),
            ("GET", "/api/goals", self.handle_goals),
            ("GET", "/api/activity", self.handle_activity),
            ("GET", "/api/suggestions", self.handle_suggestions),
            ("GET", "/api/away", self.handle_away),
            ("GET", "/api/status", self.handle_status),
            ("GET", "/api/memories", self.handle_memories),
        ]

        for method, path, handler in routes:
            if method == "POST":
                route = self.app.router.add_post(path, handler)
            else:
                route = self.app.router.add_get(path, handler)
            self.cors.add(route)

    async def handle_chat(self, request: Request) -> Response:
        """
        POST /api/chat
        Main interaction endpoint.

        Request: { "message": string }
        Response: {
            "response": string,
            "suggestions": [],
            "goals": [],
            "fitness": float,
            "mode": string,
            "trust_level": float
        }
        """
        try:
            data = await request.json()
            message = data.get("message", "")

            if not message:
                return web.json_response(
                    {"error": "No message provided"},
                    status=400
                )

            if not self.engine:
                return web.json_response(
                    {"error": "Engine not initialized"},
                    status=503
                )

            # Get response from Senter engine
            response = await self.engine.interact(message)

            # Get current goals
            goals = []
            if self.engine.goal_detector:
                try:
                    active_goals = self.engine.goal_detector.get_active_goals()
                    goals = [g.to_dict() for g in active_goals[:5]]
                except Exception:
                    pass

            return web.json_response({
                "response": response.text,
                "suggestions": response.suggestions or [],
                "goals": goals,
                "fitness": response.fitness,
                "mode": response.ai_state.mode if response.ai_state else "unknown",
                "trust_level": response.ai_state.trust_level if response.ai_state else 0.5,
                "episode_id": response.episode_id,
            })

        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    async def handle_goals(self, request: Request) -> Response:
        """
        GET /api/goals
        Returns active goals.

        Response: { "goals": [{ id, description, progress, category, status, source }] }
        """
        try:
            goals = []

            if self.engine and self.engine.goal_detector:
                active_goals = self.engine.goal_detector.get_active_goals()
                goals = [
                    {
                        "id": g.id,
                        "description": g.description,
                        "progress": g.progress,
                        "category": g.category,
                        "status": g.status,
                        "source": getattr(g, "source", "conversation"),
                        "confidence": g.confidence,
                    }
                    for g in active_goals
                ]

            return web.json_response({"goals": goals})

        except Exception as e:
            return web.json_response({"goals": [], "error": str(e)})

    async def handle_activity(self, request: Request) -> Response:
        """
        GET /api/activity
        Returns current activity context.

        Response: {
            "context": string,
            "project": string,
            "duration_minutes": int,
            "snapshot_count": int
        }
        """
        try:
            activity = {
                "context": "unknown",
                "project": None,
                "duration_minutes": 0,
                "snapshot_count": 0,
            }

            # Try to get from daemon's activity monitor
            if hasattr(self, 'activity_monitor') and self.activity_monitor:
                context = self.activity_monitor.get_current_context()
                summary = self.activity_monitor.get_activity_summary()

                activity = {
                    "context": context or "unknown",
                    "project": self.activity_monitor.project_detector.get_current_project() if hasattr(self.activity_monitor, 'project_detector') else None,
                    "duration_minutes": summary.get("total_minutes", 0),
                    "snapshot_count": len(self.activity_monitor.history),
                }

            return web.json_response(activity)

        except Exception as e:
            return web.json_response({
                "context": "unknown",
                "project": None,
                "duration_minutes": 0,
                "snapshot_count": 0,
                "error": str(e)
            })

    async def handle_suggestions(self, request: Request) -> Response:
        """
        GET /api/suggestions
        Returns proactive suggestions.

        Response: { "suggestions": [{ type, content, priority, reason }] }
        """
        try:
            suggestions = []

            if self.engine and self.engine.proactive:
                try:
                    raw_suggestions = await self.engine.proactive.generate_suggestions()
                    suggestions = raw_suggestions[:5]  # Limit to 5
                except Exception:
                    pass

            return web.json_response({"suggestions": suggestions})

        except Exception as e:
            return web.json_response({"suggestions": [], "error": str(e)})

    async def handle_away(self, request: Request) -> Response:
        """
        GET /api/away
        Returns "while you were away" summary.

        Response: {
            "summary": string,
            "research": [{ goal, summary, sources }],
            "insights": [],
            "duration_hours": float
        }
        """
        try:
            # Calculate how long server has been running
            duration = datetime.now() - self.start_time
            duration_hours = duration.total_seconds() / 3600

            research = []
            insights = []

            # Get research results if available
            if hasattr(self, 'research_results'):
                research = [
                    {
                        "goal": r.goal_id,
                        "summary": r.summary,
                        "sources": r.sources[:3] if r.sources else [],
                    }
                    for r in self.research_results[-5:]  # Last 5
                ]

            # Generate summary
            if research:
                summary = f"While you were away ({duration_hours:.1f}h), I researched {len(research)} topics related to your goals."
            else:
                summary = f"Senter has been running for {duration_hours:.1f} hours, ready to help."

            return web.json_response({
                "summary": summary,
                "research": research,
                "insights": insights,
                "duration_hours": duration_hours,
            })

        except Exception as e:
            return web.json_response({
                "summary": "Senter is ready.",
                "research": [],
                "insights": [],
                "duration_hours": 0,
                "error": str(e)
            })

    async def handle_status(self, request: Request) -> Response:
        """
        GET /api/status
        Returns system status.

        Response: {
            "status": string,
            "trust_level": float,
            "memory_count": int,
            "goal_count": int,
            "uptime_hours": float
        }
        """
        try:
            status = "ok" if self.engine else "initializing"
            trust_level = 0.5
            memory_count = 0
            goal_count = 0

            if self.engine:
                trust_level = self.engine.trust.level if self.engine.trust else 0.5

                if self.engine.memory:
                    try:
                        stats = self.engine.memory.get_stats()
                        memory_count = stats.get("total_memories", 0)
                    except Exception:
                        pass

                if self.engine.goal_detector:
                    try:
                        goals = self.engine.goal_detector.get_active_goals()
                        goal_count = len(goals)
                    except Exception:
                        pass

            uptime = (datetime.now() - self.start_time).total_seconds() / 3600

            return web.json_response({
                "status": status,
                "trust_level": trust_level,
                "memory_count": memory_count,
                "goal_count": goal_count,
                "uptime_hours": round(uptime, 2),
            })

        except Exception as e:
            return web.json_response({
                "status": "error",
                "error": str(e)
            })

    async def handle_memories(self, request: Request) -> Response:
        """
        GET /api/memories
        Returns recent memories/episodes.

        Response: { "memories": [{ input, response, timestamp }] }
        """
        try:
            memories = []

            if self.engine and self.engine.memory:
                try:
                    # Get recent episodes
                    episodes = self.engine.memory._episodic.get_recent(limit=10)
                    memories = [
                        {
                            "input": e.data.get("input", ""),
                            "response": e.data.get("response", ""),
                            "timestamp": e.timestamp.isoformat() if hasattr(e, 'timestamp') else None,
                            "fitness": e.data.get("fitness"),
                        }
                        for e in episodes
                    ]
                except Exception:
                    pass

            return web.json_response({"memories": memories})

        except Exception as e:
            return web.json_response({"memories": [], "error": str(e)})

    async def start(self):
        """Start the HTTP server."""
        print(f"[HTTP] Starting Senter HTTP API server...")

        # Initialize Senter engine
        from core.engine import Senter

        print(f"[HTTP] Initializing Senter engine from {self.genome_path}...")
        self.engine = Senter(self.genome_path)
        await self.engine.initialize()
        print(f"[HTTP] Senter engine initialized")

        # Initialize activity monitor
        try:
            from intelligence.activity import ActivityMonitor
            self.activity_monitor = ActivityMonitor()
            print(f"[HTTP] ActivityMonitor initialized")
        except Exception as e:
            print(f"[HTTP] ActivityMonitor not available: {e}")
            self.activity_monitor = None

        # Research results storage
        self.research_results = []

        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        print(f"[HTTP] Server running at http://{self.host}:{self.port}")
        print(f"[HTTP] API endpoints:")
        print(f"       POST /api/chat      - Send message")
        print(f"       GET  /api/goals     - Get goals")
        print(f"       GET  /api/activity  - Get activity")
        print(f"       GET  /api/suggestions - Get suggestions")
        print(f"       GET  /api/away      - Get away summary")
        print(f"       GET  /api/status    - Get status")
        print(f"       GET  /api/memories  - Get memories")

    async def stop(self):
        """Stop the HTTP server."""
        print(f"[HTTP] Shutting down...")

        if self.engine:
            await self.engine.shutdown()

        if self.runner:
            await self.runner.cleanup()

        print(f"[HTTP] Server stopped")


async def main():
    """Main entry point."""
    import sys

    # Find genome.yaml
    genome_path = Path(__file__).parent.parent / "genome.yaml"

    if not genome_path.exists():
        print(f"Error: genome.yaml not found at {genome_path}")
        sys.exit(1)

    server = SenterHTTPServer(genome_path)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    await server.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
