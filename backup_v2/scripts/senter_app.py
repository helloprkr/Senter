#!/usr/bin/env python3
"""
Senter TUI - Advanced Terminal User Interface
Uses OmniAgentChain for async omniagent orchestration with Focus system
"""

import sys
from pathlib import Path
from typing import List

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, Container
    from textual.widgets import (
        Static,
        Input,
        Button,
        Header,
        Footer,
    )
    from textual.containers import ScrollableContainer
    from textual import events
    from textual.binding import Binding

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

# Setup path - works from any directory
senter_root = Path(__file__).parent.parent
sys.path.insert(0, str(senter_root))
sys.path.insert(1, str(senter_root / "Functions"))
sys.path.insert(2, str(senter_root / "Focuses"))

try:
    from Functions.omniagent_chain import OmniAgentChain
except ImportError:
    print("Error importing OmniAgentChain")
    sys.exit(1)


def create_gradient_ascii():
    """Create ASCII art with diagonal gradient from dark green to mint"""
    lines = [
        "███████╗███████╗███████╗███████╗███████╗",
        "██╔════╝██╔════╝███████╗  ██████╔╝",
        "███████║███████║██║   ██║   ██████║",
        "╚════██║╚════██║██║   ██║   ╚════██║",
        "███████║███████║██║   ██║   ██████║",
        "███████║╚════██║██║   ██║   ╚════██║",
        "███████║███████║██║   ██║   ██████║",
    ]

    start_color = (0x00, 0x80, 0x80)
    end_color = (0x00, 0xFF, 0xAA)

    max_row = len(lines) - 1
    max_col = max(len(line) for line in lines) - 1

    result = []
    for row, line in enumerate(lines):
        colored_line = []
        for col, char in enumerate(line):
            t = row / max_row + col / max_col
            r = int(start_color[0] + t * (end_color[0] - start_color[0]))
            g = int(start_color[1] + t * (end_color[1] - start_color[1]))
            b = int(start_color[2] + t * (end_color[2] - start_color[2]))
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colored_line.append(f"[{hex_color}]{char}[/#{hex_color}]")
        result.append("".join(colored_line))

    return "\n".join(result)


class ChatArea(ScrollableContainer):
    """Main chat display area"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[str] = []
        self.show_ascii = True
        self.chain = None
        self.current_focus = "general"

    def compose(self) -> ComposeResult:
        yield Static(create_gradient_ascii(), id="chat-content", markup=True)

    def add_message(self, sender: str, message: str, focus: str = ""):
        """Add a message to chat"""
        if sender == "user":
            formatted = f"You: {message}"
        else:
            formatted = f"Senter: {message}"

        self.messages.append(formatted)
        self.show_ascii = False
        self._update_display()

    def clear_chat(self):
        """Clear all messages"""
        self.messages = []
        self.show_ascii = True
        self._update_display()

    def set_chain(self, chain):
        """Set omniagent chain instance"""
        self.chain = chain

    def update_focus(self, focus: str):
        """Update current focus"""
        self.current_focus = focus
        self.add_message("Senter", f"🎯 Switched to Focus: {focus}", focus)

    def _update_display(self):
        """Update chat display with current messages"""
        display_messages = "\n".join(self.messages[-50:])

        if self.current_focus and not self.show_ascii:
            focus_display = f"🎯 Focus: {self.current_focus}\n\n"
            display_messages = focus_display + display_messages

        content = self.query_one("#chat-content", Static)
        content.update(display_messages)
        self.scroll_end(animate=False)

    def scroll_end(self, animate: bool = True):
        """Scroll to end of chat area"""
        super().scroll_end(animate=animate)


class FocusList(Vertical):
    """Focus selection list widget"""

    def __init__(self, available_focuses: List[str], current_focus: str = "general"):
        super().__init__()
        self.available_focuses = available_focuses
        self.current_focus = current_focus

    def compose(self) -> ComposeResult:
        yield Static("🎯 Focuses", classes="section-title")

        for focus in self.available_focuses:
            is_current = focus == self.current_focus
            classes = ["focus-item"]
            if is_current:
                classes.append("current-focus")

            button = Button(
                f"{'▶ ' if is_current else ''}{focus}",
                classes=classes,
                id=f"focus-{focus}",
            )
            yield button

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle focus selection"""
        if event.button.id.startswith("focus-"):
            new_focus = event.button.id[6:]

            if hasattr(self, "app") and hasattr(self.app, "chat_area"):
                self.app.chat_area.update_focus(new_focus)

            self.current_focus = new_focus
            self.refresh()


class InputBar(Static):
    """Input bar that matches chat area width"""

    def compose(self) -> ComposeResult:
        with Container(id="input-container"):
            yield Input(
                placeholder="Type your message or /command...", id="message-input"
            )
            yield Button("Send", id="send-button", variant="primary")


class SenterApp(App):
    """Advanced Senter TUI with OmniAgentChain backend"""

    CSS = """
    Screen {
        background: ansi_black;
    }

    #chat-content {
        height: 1fr;
        background: ansi_black;
    }

    #input-container {
        height: 3;
    }

    #snack-bar {
        background: ansi_bright_green;
        color: ansi_black;
    }

    .section-title {
        text-align: center;
        text-style: bold;
        color: ansi_bright_green;
    }

    .focus-item {
        margin: 1;
        text-align: left;
        width: 100%;
    }

    .current-focus {
        background: ansi_green;
        color: ansi_black;
        text-style: bold;
    }

    .snack-bar-left {
        color: ansi_cyan;
    }

    .snack-bar-right {
        color: ansi_bright_green;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_area = ChatArea()
        self.input_bar = InputBar()
        self.focus_list = FocusList([], current_focus="general")
        self.chain = None

    async def on_mount(self) -> None:
        """Initialize on startup"""
        print("🚀 Initializing Senter TUI with OmniAgentChain...")

        try:
            self.chain = OmniAgentChain(senter_root)
            print("🔄 Loading all Focus agents...")
            await self.chain.initialize()
        except Exception as e:
            print(f"❌ Failed to initialize chain: {e}")
            self.exit()

        print("📝 Available user Focuses:")
        try:
            user_focuses = self.chain.list_user_focuses() or []
            for focus in user_focuses:
                print(f"   - {focus}")
        except Exception as e:
            print(f"⚠️  Could not list Focuses: {e}")
            user_focuses = []

        self.focus_list = FocusList(user_focuses, current_focus="general")
        self.chat_area.set_chain(self.chain)

        print("✅ Senter TUI initialized!")
        print("\nAvailable commands:")
        print("  /list       - List all Focuses")
        print("  /focus <name> - Set Focus")
        print("  /exit       - Exit")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button presses"""
        if event.button.id == "send-button":
            input_widget = self.query_one("#message-input", Input)

            if input_widget:
                message = input_widget.value.strip()

                if message:
                    print(f"\n📤 Processing: {message[:100]}...")

                    if message.startswith("/"):
                        await self._handle_command(message)
                    else:
                        try:
                            response = await self.chain.process_query(
                                message,
                                context="",
                                focus_hint=self.chat_area.current_focus,
                            )
                            self.chat_area.add_message(
                                "Senter", response, self.chat_area.current_focus
                            )
                            print(f"✅ Response: {response[:100]}...")
                        except Exception as e:
                            self.chat_area.add_message(
                                "Senter",
                                f"Error: {str(e)}",
                                self.chat_area.current_focus,
                            )
                            print(f"❌ Error: {e}")

                    input_widget.value = ""

    async def _handle_command(self, command: str):
        """Handle slash commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()

        if cmd == "/list":
            user_focuses = self.chain.list_user_focuses() or []
            print("\n📁 Available Focuses:")
            for focus in user_focuses:
                print(f"   - {focus}")
        elif cmd == "/focus" and len(cmd_parts) > 1:
            new_focus = " ".join(cmd_parts[1:])
            user_focuses = self.chain.list_user_focuses() or []
            if new_focus in user_focuses:
                self.chat_area.update_focus(new_focus)
                self.focus_list.current_focus = new_focus
                self.focus_list.refresh()
                print(f"\n🎯 Focus set to: {new_focus}")
            else:
                print(f"\n⚠️  Unknown focus: {new_focus}")
        elif cmd == "/exit":
            self.exit()
        else:
            print(f"\n⚠️  Unknown command: {command}")

    def action_clear_chat(self) -> None:
        """Clear chat history"""
        self.chat_area.clear_chat()

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="left-panel"):
                yield self.focus_list
                yield self.chat_area
                yield self.input_bar
            with Horizontal(id="snack-bar"):
                yield Static("", id="model-info", classes="snack-bar-left")
                yield Static(
                    "Type '/' for commands",
                    id="command-hint",
                    classes="snack-bar-right",
                )


def main():
    """Main entry point"""
    if not TEXTUAL_AVAILABLE:
        print("⚠️  Textual is not installed. Run: pip install textual")
        print("📝 Falling back to CLI mode...")

        from Functions.omniagent_chain import OmniAgentChain

        async def cli_mode():
            chain = OmniAgentChain()
            await chain.initialize()
            print("\n✅ Senter CLI mode ready!")
            print("Press Ctrl+C to exit")
            await chain.close()

        import asyncio

        asyncio.run(cli_mode())
        return

    app = SenterApp()
    app.run()


if __name__ == "__main__":
    main()
