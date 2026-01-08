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
            ("POST", "/api/chat/stream", self.handle_chat_stream),
            ("GET", "/api/goals", self.handle_goals),
            ("GET", "/api/activity", self.handle_activity),
            ("GET", "/api/suggestions", self.handle_suggestions),
            ("GET", "/api/away", self.handle_away),
            ("GET", "/api/status", self.handle_status),
            ("GET", "/api/memories", self.handle_memories),
            ("GET", "/api/briefing", self.handle_briefing),
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

    async def handle_chat_stream(self, request: Request) -> StreamResponse:
        """
        US-V4-003: POST /api/chat/stream
        Streaming chat endpoint using Server-Sent Events.

        Request: { "message": string }
        Response: SSE stream with tokens
        """
        response = StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
        )
        await response.prepare(request)

        try:
            data = await request.json()
            message = data.get("message", "")

            if not message:
                await response.write(b'data: {"error": "No message provided"}\n\n')
                return response

            if not self.engine:
                await response.write(b'data: {"error": "Engine not initialized"}\n\n')
                return response

            # Send start event
            await response.write(b'data: {"type": "start"}\n\n')

            # Stream the response
            full_text = ""
            try:
                # Check if model supports streaming
                if hasattr(self.engine, 'model') and hasattr(self.engine.model, 'stream'):
                    # Build prompt (simplified for streaming)
                    prompt = f"User: {message}\nAssistant:"

                    async for chunk in self.engine.model.stream(prompt):
                        full_text += chunk
                        chunk_data = json.dumps({"type": "token", "content": chunk})
                        await response.write(f'data: {chunk_data}\n\n'.encode())
                else:
                    # Fallback: Get full response and simulate streaming
                    result = await self.engine.interact(message)
                    full_text = result.text

                    # Send in chunks for streaming effect
                    words = full_text.split()
                    for i in range(0, len(words), 3):
                        chunk = ' '.join(words[i:i+3]) + ' '
                        chunk_data = json.dumps({"type": "token", "content": chunk})
                        await response.write(f'data: {chunk_data}\n\n'.encode())
                        await asyncio.sleep(0.05)  # Small delay for streaming effect

            except Exception as e:
                error_data = json.dumps({"type": "error", "content": str(e)})
                await response.write(f'data: {error_data}\n\n'.encode())

            # Send completion event with metadata
            goals = []
            if self.engine.goal_detector:
                try:
                    active_goals = self.engine.goal_detector.get_active_goals()
                    goals = [g.to_dict() for g in active_goals[:5]]
                except Exception:
                    pass

            complete_data = json.dumps({
                "type": "complete",
                "full_text": full_text,
                "goals": goals,
                "trust_level": self.engine.trust.level if self.engine.trust else 0.5,
            })
            await response.write(f'data: {complete_data}\n\n'.encode())

        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            await response.write(f'data: {error_data}\n\n'.encode())

        return response

    async def handle_briefing(self, request: Request) -> Response:
        """
        GET /api/briefing
        Returns the latest briefing or generates one on-demand.
        """
        try:
            briefing_dir = Path("data/briefings")

            # Find latest briefing
            if briefing_dir.exists():
                briefings = sorted(briefing_dir.glob("briefing_*.md"), reverse=True)
                if briefings:
                    latest = briefings[0]
                    content = latest.read_text()
                    return web.json_response({
                        "filename": latest.name,
                        "content": content,
                        "generated": latest.stat().st_mtime,
                    })

            return web.json_response({
                "filename": None,
                "content": "No briefing available yet. Senter is working on it!",
                "generated": None,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_goals(self, request: Request) -> Response:
        """
        US-V4-005: GET /api/goals
        Returns active goals with action plans.

        Response: { "goals": [{ id, description, progress, category, status, source, action_plan }] }
        """
        try:
            goals = []

            if self.engine and self.engine.goal_detector:
                active_goals = self.engine.goal_detector.get_active_goals()

                for g in active_goals:
                    # Generate action plan based on goal category
                    action_plan = self._generate_action_plan(g)

                    goals.append({
                        "id": g.id,
                        "description": g.description,
                        "progress": g.progress,
                        "category": g.category,
                        "status": g.status,
                        "source": getattr(g, "source", "conversation"),
                        "confidence": g.confidence,
                        "action_plan": action_plan,
                    })

            return web.json_response({"goals": goals})

        except Exception as e:
            return web.json_response({"goals": [], "error": str(e)})

    def _generate_action_plan(self, goal) -> List[Dict[str, Any]]:
        """
        US-V4-005: Generate action plan steps for a goal.
        """
        # Category-specific plan templates
        plan_templates = {
            "learning": [
                {"step": 1, "description": f"Research foundational resources for {goal.description}", "status": "pending", "effort": "30 min"},
                {"step": 2, "description": "Create a structured learning schedule", "status": "pending", "effort": "15 min"},
                {"step": 3, "description": "Complete first practical exercise or tutorial", "status": "pending", "effort": "1 hour"},
                {"step": 4, "description": "Build a small project to apply knowledge", "status": "pending", "effort": "2 hours"},
                {"step": 5, "description": "Review progress and identify next learning areas", "status": "pending", "effort": "15 min"},
            ],
            "project": [
                {"step": 1, "description": f"Define clear objectives for {goal.description}", "status": "pending", "effort": "20 min"},
                {"step": 2, "description": "Break down into milestones", "status": "pending", "effort": "30 min"},
                {"step": 3, "description": "Complete first milestone", "status": "pending", "effort": "2 hours"},
                {"step": 4, "description": "Review and iterate", "status": "pending", "effort": "1 hour"},
            ],
            "career": [
                {"step": 1, "description": f"Clarify specific outcomes for {goal.description}", "status": "pending", "effort": "20 min"},
                {"step": 2, "description": "Identify key skills or experiences needed", "status": "pending", "effort": "30 min"},
                {"step": 3, "description": "Create action items for skill development", "status": "pending", "effort": "1 hour"},
                {"step": 4, "description": "Network or seek mentorship opportunities", "status": "pending", "effort": "ongoing"},
            ],
            "health": [
                {"step": 1, "description": f"Set specific metrics for {goal.description}", "status": "pending", "effort": "15 min"},
                {"step": 2, "description": "Create daily routine or habit", "status": "pending", "effort": "15 min"},
                {"step": 3, "description": "Track progress for one week", "status": "pending", "effort": "ongoing"},
                {"step": 4, "description": "Adjust approach based on results", "status": "pending", "effort": "20 min"},
            ],
        }

        # Get template for category or use default
        template = plan_templates.get(goal.category, [
            {"step": 1, "description": f"Define what success looks like for {goal.description}", "status": "pending", "effort": "15 min"},
            {"step": 2, "description": "Identify first actionable step", "status": "pending", "effort": "10 min"},
            {"step": 3, "description": "Complete first step and evaluate", "status": "pending", "effort": "1 hour"},
        ])

        # Mark steps as complete based on progress
        steps_to_complete = int(len(template) * goal.progress)
        for i, step in enumerate(template):
            if i < steps_to_complete:
                step["status"] = "completed"

        return template

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
                    "project": self.activity_monitor.get_current_project() if hasattr(self.activity_monitor, 'get_current_project') else None,
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
        US-V4-006: GET /api/suggestions
        Returns proactive suggestions including cross-system synthesis.

        Response: { "suggestions": [{ type, content, priority, reason }] }
        """
        try:
            suggestions = []

            # Get regular proactive suggestions
            if self.engine and self.engine.proactive:
                try:
                    raw_suggestions = await self.engine.proactive.generate_suggestions()
                    suggestions = raw_suggestions[:3]  # Limit regular suggestions
                except Exception:
                    pass

            # US-V4-006: Add cross-system synthesis insights
            synthesis_insights = await self._generate_synthesis_insights()
            suggestions.extend(synthesis_insights)

            return web.json_response({"suggestions": suggestions[:5]})

        except Exception as e:
            return web.json_response({"suggestions": [], "error": str(e)})

    async def _generate_synthesis_insights(self) -> List[Dict[str, Any]]:
        """
        US-V4-006: Generate insights by correlating activity, goals, and memory.
        """
        insights = []

        try:
            # Get activity data
            activity_context = "unknown"
            activity_project = None
            activity_summary = {}

            if self.activity_monitor and self.activity_monitor.history:
                activity_context = self.activity_monitor.get_current_context()
                activity_project = self.activity_monitor.get_current_project()
                activity_summary = self.activity_monitor.get_activity_summary()

            # Get goals
            goals = []
            goal_descriptions = []
            if self.engine and self.engine.goal_detector:
                goals = self.engine.goal_detector.get_active_goals()
                goal_descriptions = [g.description.lower() for g in goals]

            # Synthesis 1: Activity doesn't match any goal
            if activity_context != "unknown" and activity_project:
                project_matches_goal = any(
                    activity_project.lower() in desc or desc in activity_project.lower()
                    for desc in goal_descriptions
                )
                if not project_matches_goal and len(self.activity_monitor.history) > 10:
                    insights.append({
                        "type": "synthesis",
                        "title": f"New focus detected: {activity_project}",
                        "content": f"You've been working on '{activity_project}' but it's not in your goals. Should Senter add it?",
                        "priority": "medium",
                        "reason": "Cross-system analysis: activity vs goals",
                        "action": "add_goal",
                        "data": {"project": activity_project, "context": activity_context}
                    })

            # Synthesis 2: Context type without matching goal
            context_goal_map = {
                "coding": ["project", "learning"],
                "writing": ["project", "career"],
                "research": ["learning", "career"],
            }
            if activity_context in context_goal_map:
                expected_categories = context_goal_map[activity_context]
                has_matching_goal = any(g.category in expected_categories for g in goals)
                if not has_matching_goal and len(self.activity_monitor.history) > 5:
                    insights.append({
                        "type": "synthesis",
                        "title": f"Activity pattern: {activity_context}",
                        "content": f"You've been {activity_context} but have no related goals. What are you working toward?",
                        "priority": "low",
                        "reason": "Cross-system analysis: activity patterns",
                    })

            # Synthesis 3: Research results related to activity
            if self.research_results and activity_project:
                related_research = [
                    r for r in self.research_results
                    if activity_project.lower() in r.get("goal_description", "").lower()
                ]
                if related_research:
                    insights.append({
                        "type": "synthesis",
                        "title": "Relevant research available",
                        "content": f"While you work on {activity_project}, check out the research Senter gathered.",
                        "priority": "high",
                        "reason": "Cross-system: activity + research",
                        "data": {"research_count": len(related_research)}
                    })

        except Exception as e:
            print(f"[SYNTHESIS] Error generating insights: {e}")

        return insights

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
            self.activity_monitor = ActivityMonitor(senter_engine=self.engine)
            print(f"[HTTP] ActivityMonitor initialized")
        except Exception as e:
            print(f"[HTTP] ActivityMonitor not available: {e}")
            self.activity_monitor = None

        # Research results storage
        self.research_results = []

        # Background task tracking
        self.background_tasks = []
        self.running = True

        # US-V4-001: Start activity capture loop
        if self.activity_monitor:
            activity_task = asyncio.create_task(self._run_activity_capture())
            self.background_tasks.append(activity_task)
            print(f"[HTTP] Activity capture loop started (60s interval)")

        # US-V4-002: Start auto-research loop
        research_task = asyncio.create_task(self._run_auto_research())
        self.background_tasks.append(research_task)
        print(f"[HTTP] Auto-research loop started (5 min interval for demo)")

        # US-V4-004: Start briefing generator (runs once after delay)
        briefing_task = asyncio.create_task(self._generate_daily_briefing())
        self.background_tasks.append(briefing_task)
        print(f"[HTTP] Daily briefing generator scheduled")

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
        print(f"[HTTP] Background intelligence: ACTIVE")

    async def _run_activity_capture(self):
        """
        US-V4-001: Run activity capture loop.
        Captures user activity every 60 seconds.
        """
        capture_interval = 60  # seconds
        print(f"[ACTIVITY] Starting capture loop...")

        while self.running:
            try:
                if self.activity_monitor:
                    # Capture current activity
                    await self.activity_monitor._capture_cycle()
                    snapshot_count = len(self.activity_monitor.history)
                    context = self.activity_monitor.get_current_context()
                    project = self.activity_monitor.get_current_project()

                    if snapshot_count % 5 == 0:  # Log every 5 captures
                        print(f"[ACTIVITY] Captured #{snapshot_count}: context={context}, project={project}")

            except Exception as e:
                print(f"[ACTIVITY] Capture error: {e}")

            await asyncio.sleep(capture_interval)

    async def _run_auto_research(self):
        """
        US-V4-002: Run auto-research loop.
        Researches user's learning goals every 5 minutes (shortened for demo).
        """
        research_interval = 5 * 60  # 5 minutes for demo (normally 6 hours)
        print(f"[RESEARCH] Starting auto-research loop...")

        # Wait initial delay before first research
        await asyncio.sleep(60)  # Wait 1 minute before first research

        while self.running:
            try:
                if self.engine and self.engine.goal_detector:
                    goals = self.engine.goal_detector.get_active_goals()
                    learning_goals = [g for g in goals if g.category == "learning"]

                    for goal in learning_goals[:2]:  # Limit to 2 goals per cycle
                        print(f"[RESEARCH] Researching: {goal.description}")

                        # Use web search to research the goal
                        try:
                            from tools.web_search import WebSearch
                            ws = WebSearch()
                            query = f"{goal.description} tutorial guide tips"
                            results = await ws.search(query, max_results=3)

                            if results:
                                # Create research result
                                research_entry = {
                                    "goal_id": goal.id,
                                    "goal_description": goal.description,
                                    "query": query,
                                    "timestamp": datetime.now().isoformat(),
                                    "summary": f"Found {len(results)} resources for '{goal.description}'",
                                    "sources": [
                                        {"title": r.get("title", ""), "url": r.get("url", "")}
                                        for r in results[:3]
                                    ]
                                }
                                self.research_results.append(research_entry)
                                print(f"[RESEARCH] Completed: {len(results)} results for {goal.description}")

                                # Keep only last 20 results
                                if len(self.research_results) > 20:
                                    self.research_results = self.research_results[-20:]

                        except Exception as e:
                            print(f"[RESEARCH] Search error for {goal.description}: {e}")

            except Exception as e:
                print(f"[RESEARCH] Loop error: {e}")

            await asyncio.sleep(research_interval)

    async def _generate_daily_briefing(self):
        """
        US-V4-004: Generate daily briefing - the autonomous artifact.
        Creates a markdown file with activity summary, goals, and suggestions.
        """
        # Wait for some data to accumulate
        await asyncio.sleep(180)  # Wait 3 minutes before first briefing

        briefing_dir = Path("data/briefings")
        briefing_dir.mkdir(parents=True, exist_ok=True)

        while self.running:
            try:
                print(f"[BRIEFING] Generating daily briefing...")

                # Gather data
                activity_summary = {}
                if self.activity_monitor:
                    activity_summary = self.activity_monitor.get_activity_summary()

                goals = []
                if self.engine and self.engine.goal_detector:
                    goals = self.engine.goal_detector.get_active_goals()[:5]

                suggestions = []
                if self.engine and self.engine.proactive:
                    try:
                        suggestions = await self.engine.proactive.generate_suggestions()
                    except Exception:
                        pass

                # Create briefing content
                now = datetime.now()
                briefing_content = f"""# Senter Daily Briefing
**Generated:** {now.strftime("%Y-%m-%d %H:%M")}

---

## Activity Summary

"""
                if activity_summary:
                    context = activity_summary.get("current_context", "unknown")
                    snapshots = activity_summary.get("total_snapshots", 0)
                    top_apps = activity_summary.get("top_apps", {})
                    top_projects = activity_summary.get("top_projects", {})

                    briefing_content += f"- **Current Context:** {context}\n"
                    briefing_content += f"- **Activity Snapshots:** {snapshots}\n"

                    if top_apps:
                        briefing_content += f"- **Top Applications:** {', '.join(list(top_apps.keys())[:3])}\n"

                    if top_projects:
                        briefing_content += f"- **Detected Projects:** {', '.join(list(top_projects.keys())[:3])}\n"
                else:
                    briefing_content += "No activity data collected yet.\n"

                briefing_content += "\n## Your Goals\n\n"

                if goals:
                    for g in goals:
                        progress_pct = int(g.progress * 100)
                        briefing_content += f"- **{g.description}** ({g.category}) - {progress_pct}% progress\n"
                else:
                    briefing_content += "No goals detected yet. Keep chatting with Senter!\n"

                briefing_content += "\n## Suggestions\n\n"

                if suggestions:
                    for s in suggestions[:3]:
                        title = s.get("title", s.get("content", "Suggestion"))
                        briefing_content += f"- {title}\n"
                else:
                    briefing_content += "No suggestions at this time.\n"

                briefing_content += "\n## Research Findings\n\n"

                if self.research_results:
                    for r in self.research_results[-3:]:
                        briefing_content += f"### {r.get('goal_description', 'Research')}\n"
                        briefing_content += f"{r.get('summary', '')}\n"
                        for src in r.get("sources", [])[:2]:
                            briefing_content += f"- [{src.get('title', 'Link')}]({src.get('url', '')})\n"
                        briefing_content += "\n"
                else:
                    briefing_content += "No research completed yet. Senter is working on it!\n"

                briefing_content += f"""
---

*This briefing was created autonomously by Senter while you worked.*
*Senter is always learning and improving to help you better.*
"""

                # Write briefing file
                filename = f"briefing_{now.strftime('%Y%m%d_%H%M')}.md"
                filepath = briefing_dir / filename
                filepath.write_text(briefing_content)

                print(f"[BRIEFING] Created: {filepath}")

            except Exception as e:
                print(f"[BRIEFING] Generation error: {e}")

            # Generate new briefing every 30 minutes
            await asyncio.sleep(30 * 60)

    async def stop(self):
        """Stop the HTTP server."""
        print(f"[HTTP] Shutting down...")

        # Stop background loops
        self.running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self.engine:
            await self.engine.shutdown()

        if self.runner:
            await self.runner.cleanup()

        print(f"[HTTP] Server stopped")


async def main():
    """Main entry point."""
    import sys

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Find genome.yaml
    genome_path = project_root / "genome.yaml"

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
