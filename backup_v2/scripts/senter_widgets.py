#!/usr/bin/env python3
"""
Senter Custom Widgets
Interactive widgets for the modular sidebar system
"""

import os
import sys
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Grid
from textual.widgets import (
    Static,
    Button,
    Input,
    TextArea,
)
from textual.widget import Widget


@dataclass
class SidebarItem:
    """Data class for sidebar list items"""

    id: str
    title: str
    description: str = ""
    status: str = "active"
    proposed: bool = False
    approved: bool = False
    priority: str = "medium"
    due_date: Optional[date] = None
    created: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class EditableListItem(Widget):
    """An editable list item with approve/deny/edit controls"""

    def __init__(
        self,
        item: SidebarItem,
        on_approve: Callable = None,
        on_deny: Callable = None,
        on_edit: Callable = None,
    ):
        super().__init__()
        self.item = item
        self.on_approve = on_approve
        self.on_deny = on_deny
        self.on_edit = on_edit
        self.is_editing = False

    def compose(self) -> ComposeResult:
        classes = ["sidebar-item"]
        if self.item.proposed:
            classes.append("proposed")
        if not self.item.approved and self.item.proposed:
            classes.append("new-item")

        with Vertical(classes=classes):
            # Title and priority (editable mode)
            if self.is_editing:
                with Horizontal():
                    yield Input(
                        placeholder="Title",
                        id="edit-title",
                        classes="edit-input",
                        value=self.item.title,
                    )
                    priority_options = ["low", "medium", "high"]
                    yield Input(
                        placeholder=f"Priority ({', '.join(priority_options)})",
                        id="edit-priority",
                        classes="edit-input",
                        value=self.item.priority,
                    )
            else:
                with Horizontal():
                    title = f"[{self.item.priority.upper()}] {self.item.title}"
                    if self.item.due_date:
                        days_until = (self.item.due_date - date.today()).days
                        if days_until < 0:
                            title += f" (OVERDUE: {abs(days_until)} days)"
                        elif days_until == 0:
                            title += " (DUE TODAY)"
                        elif days_until <= 3:
                            title += f" (Due in {days_until} days)"

                    yield Static(title, classes="item-title")

            # Description (editable)
            if self.is_editing:
                yield Input(
                    placeholder="Description",
                    id="edit-description",
                    classes="edit-input",
                    value=self.item.description or "",
                )
            elif self.item.description:
                yield Static(self.item.description, classes="item-description")

            # Status and metadata
            if not self.is_editing:
                status_info = []
                if self.item.status != "active":
                    status_info.append(f"Status: {self.item.status}")
                if self.item.created:
                    status_info.append(
                        f"Created: {self.item.created.strftime('%m/%d')}"
                    )
                if status_info:
                    yield Static(" | ".join(status_info), classes="item-meta")

            # Controls
            if self.is_editing:
                with Horizontal(classes="item-controls"):
                    yield Button(
                        "💾", id="save-edit", classes="small-button save-button"
                    )
                    yield Button(
                        "❌", id="cancel-edit", classes="small-button cancel-button"
                    )
            else:
                with Horizontal(classes="item-controls"):
                    if self.item.proposed and not self.item.approved:
                        yield Button(
                            "✓", id="approve", classes="small-button approve-button"
                        )
                        yield Button("✗", id="deny", classes="small-button deny-button")

                    yield Button("✏️", id="edit", classes="small-button edit-button")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "approve" and self.on_approve:
            await self.on_approve(self.item.id)
        elif event.button.id == "deny" and self.on_deny:
            await self.on_deny(self.item.id)
        elif event.button.id == "edit" and self.on_edit:
            self.is_editing = True
            await self.on_edit(self.item.id)
            self.refresh()
        elif event.button.id == "save-edit":
            await self._save_edit()
        elif event.button.id == "cancel-edit":
            self.is_editing = False
            self.refresh()

    async def _save_edit(self):
        """Save edited item"""
        try:
            title_input = self.query_one("#edit-title", Input)
            priority_input = self.query_one("#edit-priority", Input)
            desc_input = self.query_one("#edit-description", Input)

            if title_input:
                self.item.title = title_input.value
            if priority_input:
                new_priority = priority_input.value.lower()
                if new_priority in ["low", "medium", "high"]:
                    self.item.priority = new_priority
            if desc_input:
                self.item.description = desc_input.value

            self.is_editing = False
            self.refresh()
        except Exception as e:
            print(f"Error saving edit: {e}")
            self.is_editing = False
            self.refresh()


class FocusList(Widget):
    """Focuses sidebar module"""

    def __init__(
        self, focuses: List[SidebarItem] = None, current_focus: str = "general"
    ):
        super().__init__()
        self.focuses = focuses or []
        self.current_focus = current_focus

    def compose(self) -> ComposeResult:
        yield Static("🎯 Focuses", classes="sidebar-module-title")

        for focus in self.focuses:
            is_current = focus.id == self.current_focus
            classes = ["sidebar-item"]
            if is_current:
                classes.append("current-focus")

            with Vertical(classes=classes):
                title = f"{'▶ ' if is_current else ''}{focus.title}"
                yield Static(title, classes="item-title")
                if focus.description:
                    yield Static(focus.description, classes="item-description")


class GoalsList(Widget):
    """Goals sidebar module with approve/deny functionality"""

    def __init__(self, goals: List[SidebarItem] = None):
        super().__init__()
        self.goals = goals or []

    def compose(self) -> ComposeResult:
        yield Static("🎯 Goals", classes="sidebar-module-title")

        for goal in self.goals:
            yield EditableListItem(
                goal,
                on_approve=self.approve_goal,
                on_deny=self.deny_goal,
                on_edit=self.edit_goal,
            )

    async def approve_goal(self, goal_id: str):
        """Approve a proposed goal"""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.approved = True
                goal.proposed = False
                break
        # Trigger UI refresh
        self.refresh()

    async def deny_goal(self, goal_id: str):
        """Deny a proposed goal"""
        self.goals = [g for g in self.goals if g.id != goal_id]
        # Trigger UI refresh
        self.refresh()

    async def edit_goal(self, goal_id: str):
        """Edit a goal (placeholder for now)"""
        # TODO: Implement inline editing
        pass


class TasksList(Widget):
    """AI-proposed tasks sidebar module"""

    def __init__(self, tasks: List[SidebarItem] = None):
        super().__init__()
        self.tasks = tasks or []

    def compose(self) -> ComposeResult:
        yield Static("✅ AI Tasks", classes="sidebar-module-title")

        for task in self.tasks:
            yield EditableListItem(
                task,
                on_approve=self.approve_task,
                on_deny=self.deny_task,
                on_edit=self.edit_task,
            )

    async def approve_task(self, task_id: str):
        """Approve a proposed task"""
        for task in self.tasks:
            if task.id == task_id:
                task.approved = True
                task.proposed = False
                break
        self.refresh()

    async def deny_task(self, task_id: str):
        """Deny a proposed task"""
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.refresh()

    async def edit_task(self, task_id: str):
        """Edit a task"""
        pass


class TodoList(Widget):
    """User's personal to-do list"""

    def __init__(self, todos: List[SidebarItem] = None):
        super().__init__()
        self.todos = todos or []

    def compose(self) -> ComposeResult:
        yield Static("📝 To-Do", classes="sidebar-module-title")

        for todo in self.todos:
            yield EditableListItem(todo, on_edit=self.edit_todo)

        # Add new todo button
        yield Button("+ Add Task", id="add-todo", classes="add-button")

    async def edit_todo(self, todo_id: str):
        """Edit a todo item"""
        pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle add button"""
        if event.button.id == "add-todo":
            await self._add_todo_dialog()

    async def _add_todo_dialog(self):
        """Show dialog to add new todo"""
        from textual.screen import ModalScreen
        from textual.containers import Horizontal, Vertical

        class AddTodoModal(ModalScreen):
            def __init__(self, parent_widget):
                super().__init__()
                self.parent_widget = parent_widget

            def compose(self) -> ComposeResult:
                yield Vertical(
                    Static("Add New Task", classes="modal-title"),
                    Input(
                        placeholder="Task title",
                        id="new-todo-title",
                        classes="modal-input",
                    ),
                    Input(
                        placeholder="Description (optional)",
                        id="new-todo-desc",
                        classes="modal-input",
                    ),
                    Horizontal(
                        Button("Cancel", id="cancel-add", classes="modal-button"),
                        Button("Add", id="confirm-add", classes="modal-button primary"),
                        classes="modal-buttons",
                    ),
                    classes="modal-container",
                )

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "cancel-add":
                    self.app.pop_screen()
                elif event.button.id == "confirm-add":
                    title_input = self.query_one("#new-todo-title", Input)
                    desc_input = self.query_one("#new-todo-desc", Input)

                    if title_input.value.strip():
                        new_todo = SidebarItem(
                            id=f"todo_{len(self.parent_widget.todos)}_{datetime.now().timestamp()}",
                            title=title_input.value.strip(),
                            description=desc_input.value.strip(),
                            priority="medium",
                            status="active",
                        )
                        self.parent_widget.todos.append(new_todo)
                        self.parent_widget.refresh()
                    self.app.pop_screen()

        await self.app.push_screen(AddTodoModal(self))


class CalendarWidget(Widget):
    """30-day calendar view"""

    def __init__(self, events: Dict[date, List[SidebarItem]] = None):
        super().__init__()
        self.current_month = date.today().replace(day=1)
        self.events = events or {}
        self.selected_date = None

    def compose(self) -> ComposeResult:
        # Month header with navigation
        with Horizontal(classes="calendar-header"):
            yield Button("◀", id="prev-month", classes="nav-button")
            month_year = self.current_month.strftime("%B %Y")
            yield Static(month_year, classes="month-title")
            yield Button("▶", id="next-month", classes="nav-button")

        # Day headers
        with Grid(classes="day-headers", grid_size=(7, 1)):
            for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
                yield Static(day, classes="day-header")

        # Calendar grid
        with Grid(classes="calendar-grid", grid_size=(7, 6)):
            for week in range(6):
                for day_of_week in range(7):
                    day_date = self._get_date_for_cell(week, day_of_week)
                    if day_date:
                        has_events = day_date in self.events
                        is_today = day_date == date.today()
                        is_current_month = day_date.month == self.current_month.month

                        classes = ["calendar-day"]
                        if is_today:
                            classes.append("today")
                        if has_events:
                            classes.append("has-events")
                        if self.selected_date and day_date == self.selected_date:
                            classes.append("selected")
                        if not is_current_month:
                            classes.append("other-month")

                        day_num = str(day_date.day)
                        yield Button(day_num, classes=classes, id=f"day-{day_date}")

        # Show events for selected date
        if self.selected_date:
            yield Static(
                f"Selected: {self.selected_date.strftime('%B %d, %Y')}",
                classes="selected-date-header",
            )
            if self.selected_date in self.events and self.events[self.selected_date]:
                with Vertical(classes="selected-date-events"):
                    for event in self.events[self.selected_date]:
                        yield Static(f"• {event.title}", classes="selected-date-event")
                        if event.description:
                            yield Static(
                                f"  {event.description}", classes="event-description"
                            )
            else:
                yield Static("No events for this date", classes="no-events-message")

    def _get_date_for_cell(self, week: int, day_of_week: int) -> Optional[date]:
        """Get the date for a specific calendar cell"""
        # Find first day of the month and its weekday
        first_day = self.current_month
        first_weekday = first_day.weekday()  # 0=Monday, 6=Sunday

        # Calculate the date for this cell
        cell_day = week * 7 + day_of_week - first_weekday + 1

        try:
            cell_date = first_day.replace(day=cell_day)
            return cell_date
        except ValueError:
            return None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle calendar navigation and day selection"""
        if event.button.id == "prev-month":
            # Go to previous month
            if self.current_month.month == 1:
                self.current_month = self.current_month.replace(
                    year=self.current_month.year - 1, month=12
                )
            else:
                self.current_month = self.current_month.replace(
                    month=self.current_month.month - 1
                )
            self.refresh()

        elif event.button.id == "next-month":
            # Go to next month
            if self.current_month.month == 12:
                self.current_month = self.current_month.replace(
                    year=self.current_month.year + 1, month=1
                )
            else:
                self.current_month = self.current_month.replace(
                    month=self.current_month.month + 1
                )
            self.refresh()

        elif event.button.id.startswith("day-"):
            # Day selected
            date_str = event.button.id[4:]  # Remove "day-" prefix
            selected_date = date.fromisoformat(date_str)
            self.selected_date = selected_date
            # TODO: Show events for selected date
            self.refresh()


class UpcomingEvents(Widget):
    """Upcoming events list below calendar"""

    def __init__(self, events: Dict[date, List[SidebarItem]] = None):
        super().__init__()
        self.events = events or {}

    def compose(self) -> ComposeResult:
        yield Static("📋 Upcoming Events", classes="events-header")

        # Get next 5 upcoming events
        upcoming = self._get_upcoming_events()[:5]

        if not upcoming:
            yield Static("No upcoming events", classes="no-events")
        else:
            for event in upcoming:
                days_until = (event.due_date - date.today()).days
                if days_until == 0:
                    time_str = "Today"
                elif days_until == 1:
                    time_str = "Tomorrow"
                else:
                    time_str = f"In {days_until} days"

                yield Static(f"• {event.title} ({time_str})", classes="event-item")

    def _get_upcoming_events(self) -> List[SidebarItem]:
        """Get upcoming events sorted by date"""
        upcoming = []
        today = date.today()

        for event_date, events in self.events.items():
            if event_date >= today:
                for event in events:
                    event.due_date = event_date  # Ensure due_date is set
                    upcoming.append(event)

        return sorted(upcoming, key=lambda x: x.due_date)


class CalendarView(Widget):
    """Complete calendar module (takes 2 module heights)"""

    def __init__(self, events: Dict[date, List[SidebarItem]] = None):
        super().__init__()
        self.events = events or {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield CalendarWidget(self.events)
            yield UpcomingEvents(self.events)


class NotificationBadge(Widget):
    """Visual notification indicator"""

    def __init__(self, count: int = 0):
        super().__init__()
        self.count = count

    def compose(self) -> ComposeResult:
        if self.count > 0:
            yield Static(f"{self.count}", classes="notification-badge")
        else:
            yield Static("", classes="notification-badge hidden")

    def update_count(self, count: int):
        """Update notification count"""
        self.count = count
        self.refresh()


class SidebarContainer(Widget):
    """Dynamic sidebar container with searchable items"""

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref  # Reference to main app
        self.all_items = self._load_all_items()
        self.filtered_items = self.all_items.copy()
        self.selected_item = None
        self.search_query = ""

    def _load_all_items(self) -> List[SidebarItem]:
        """Load all focuses, agents, functions, settings"""
        items = []

        # Load Focuses
        focuses_dir = Path("Focuses")
        if focuses_dir.exists():
            for focus_dir in focuses_dir.iterdir():
                if focus_dir.is_dir():
                    senter_md = focus_dir / "SENTER.md"
                    description = ""
                    if senter_md.exists():
                        try:
                            with open(senter_md, "r") as f:
                                content = f.read()
                                # Get first line or summary
                                lines = content.split("\n")
                                description = lines[0] if lines else ""
                        except:
                            pass
                    items.append(
                        SidebarItem(
                            id=f"focus:{focus_dir.name}",
                            title=focus_dir.name,
                            description=description,
                            status="focus",
                        )
                    )

        # Load agents
        agents_dir = Path("Agents")
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.json"):
                try:
                    with open(agent_file, "r") as f:
                        agent_data = json.load(f)
                        name = agent_data.get("agent", {}).get("name", agent_file.stem)
                        description = agent_data.get("agent", {}).get("description", "")
                        items.append(
                            SidebarItem(
                                id=f"agent:{agent_file.stem}",
                                title=name,
                                description=description,
                                status="agent",
                            )
                        )
                except:
                    pass

        # Load functions
        functions_dir = Path("Functions")
        if functions_dir.exists():
            for func_file in functions_dir.glob("*.py"):
                items.append(
                    SidebarItem(
                        id=f"function:{func_file.stem}",
                        title=func_file.stem,
                        description=f"Python function: {func_file.stem}",
                        status="function",
                    )
                )

        # Load settings
        config_dir = Path("config")
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                items.append(
                    SidebarItem(
                        id=f"setting:{config_file.stem}",
                        title=config_file.stem,
                        description=f"Configuration: {config_file.stem}",
                        status="setting",
                    )
                )

        return items

    def _render_item_list(self) -> Widget:
        """Render the filtered list of items"""
        container = Vertical(id="item-list")
        for item in self.filtered_items:
            safe_id = item.id.replace(":", "_").replace("/", "_").replace(".", "_")
            button = Button(f"{item.title} ({item.status})", id=f"item-{safe_id}")
            container.mount(button)
        return container

    def _render_selected_item(self) -> Widget:
        """Render the selected item details"""
        with Vertical() as container:
            # Back button
            yield Button("← Back", id="back-button")

            # Item details
            yield Static(
                f"Type: {self.selected_item.status.title()}", classes="item-type"
            )
            yield Static(f"Name: {self.selected_item.title}", classes="item-title")

            if self.selected_item.status == "focus":
                # Load and show SENTER.md
                focus_name = self.selected_item.id.split(":", 1)[1]
                senter_path = Path("Focuses") / focus_name / "SENTER.md"
                content = ""
                if senter_path.exists():
                    try:
                        with open(senter_path, "r") as f:
                            content = f.read()
                    except:
                        content = "Error loading SENTER.md"
                yield TextArea(content, id="senter-editor", classes="senter-editor")

                # Show mapped agent
                agent_name = self.app_ref.focus_agent_map.get(focus_name, "senter")
                yield Static(f"Mapped Agent: {agent_name}", classes="mapped-agent")
                yield Button(
                    "✏️ Edit Agent", id="edit-agent", classes="edit-agent-button"
                )

            elif self.selected_item.status == "agent":
                agent_name = self.selected_item.id.split(":", 1)[1]
                agent_path = Path("Agents") / f"{agent_name}.json"
                if agent_path.exists():
                    try:
                        with open(agent_path, "r") as f:
                            agent_data = json.load(f)
                            # Pretty print JSON
                            content = json.dumps(agent_data, indent=2)
                    except:
                        content = "Error loading agent"
                else:
                    content = "Agent file not found"
                yield TextArea(content, readonly=True, classes="agent-viewer")

            else:
                yield Static(
                    self.selected_item.description or "No description available",
                    classes="item-description",
                )

        return container

    def compose(self) -> ComposeResult:
        # Search input
        yield Input(
            placeholder="Search topics, agents, functions, settings...",
            id="sidebar-search",
        )

        # Content area
        with Vertical(id="sidebar-content"):
            if self.selected_item:
                yield self._render_selected_item()
            else:
                yield self._render_item_list()

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "sidebar-search":
            self.search_query = event.value.lower()
            self._filter_items()
            if not self.selected_item:
                self._refresh_list()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "back-button":
            self.selected_item = None
            self._refresh_content()
        elif event.button.id.startswith("item-"):
            safe_id = event.button.id[5:]  # Remove "item-" prefix
            # Convert back to original id
            item_id = safe_id.replace("_", ":").replace(
                "_", "/"
            )  # Reverse the sanitization
            for (
                item
            ) in self.all_items:  # Check all_items since filtered may not have it
                if (
                    item.id.replace(":", "_").replace("/", "_").replace(".", "_")
                    == safe_id
                ):
                    self.selected_item = item
                    self._refresh_content()
                    break

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "back-button":
            self.selected_item = None
            self._refresh_content()
        elif (
            event.button.id == "edit-agent"
            and self.selected_item
            and self.selected_item.status == "focus"
        ):
            await self._edit_agent_mapping()

    async def _edit_agent_mapping(self):
        """Open dialog to edit agent mapping for focus"""
        from textual.screen import ModalScreen
        from textual.containers import Horizontal, Vertical

        if self.selected_item.status != "focus":
            return

        focus_name = self.selected_item.id.split(":", 1)[1]
        current_agent = self.app_ref.focus_agent_map.get(focus_name, "senter")

        class EditAgentModal(ModalScreen):
            def __init__(self, parent_widget, focus_name, current_agent):
                super().__init__()
                self.parent_widget = parent_widget
                self.focus_name = focus_name
                self.current_agent = current_agent

            def compose(self) -> ComposeResult:
                yield Vertical(
                    Static(f"Edit Agent for: {self.focus_name}", classes="modal-title"),
                    Input(
                        placeholder="Agent name",
                        id="agent-name-input",
                        classes="modal-input",
                        value=self.current_agent,
                    ),
                    Horizontal(
                        Button(
                            "Cancel", id="cancel-edit-agent", classes="modal-button"
                        ),
                        Button(
                            "Save", id="save-edit-agent", classes="modal-button primary"
                        ),
                        classes="modal-buttons",
                    ),
                    classes="modal-container",
                )

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "cancel-edit-agent":
                    self.app.pop_screen()
                elif event.button.id == "save-edit-agent":
                    agent_input = self.query_one("#agent-name-input", Input)
                    if agent_input.value.strip():
                        self.parent_widget.app_ref.focus_agent_map[self.focus_name] = (
                            agent_input.value.strip()
                        )
                        self.parent_widget._refresh_content()
                    self.app.pop_screen()

        await self.app.push_screen(EditAgentModal(self, focus_name, current_agent))

    def _filter_items(self):
        """Filter items based on search query"""
        if not self.search_query:
            self.filtered_items = self.all_items.copy()
        else:
            self.filtered_items = [
                item
                for item in self.all_items
                if self.search_query in item.title.lower()
                or self.search_query in item.description.lower()
                or self.search_query in item.status.lower()
            ]

    def _refresh_list(self):
        """Refresh the item list"""
        content_area = self.query_one("#sidebar-content", Vertical)
        content_area.remove_children()
        content_area.mount(self._render_item_list())

    def _refresh_content(self):
        """Refresh the content area"""
        content_area = self.query_one("#sidebar-content", Vertical)
        content_area.remove_children()
        if self.selected_item:
            content_area.mount(self._render_selected_item())
        else:
            content_area.mount(self._render_item_list())
