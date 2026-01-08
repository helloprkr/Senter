# SENTER AI

<div align="center">

![SENTER AI](public/placeholder-logo.png)

**A Beautiful, Transparent macOS AI Assistant with Floating Glass Panels**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Next.js](https://img.shields.io/badge/Next.js-15.2.4-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-blue)](https://react.dev/)
[![Electron](https://img.shields.io/badge/Electron-Latest-47848F)](https://www.electronjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6)](https://www.typescriptlang.org/)

[Features](#features) • [Installation](#installation) • [Architecture](#architecture) • [Development](#development) • [Contributing](#contributing)

</div>

---

## 🌟 Overview

SENTER is a next-generation macOS AI assistant that combines the power of modern AI models with a stunning glassmorphism UI. Inspired by macOS Spotlight, SENTER floats above your desktop with transparent panels that blur the background, creating a seamless integration with your workflow.

### ✨ Key Highlights

- 🎨 **Glassmorphism Design**: Beautiful floating panels with backdrop blur and transparency
- 🤖 **Multi-Provider AI**: Support for Ollama (local), OpenAI, and Anthropic
- 🎤 **Voice Input**: Real-time speech-to-text transcription
- 📸 **Visual Context**: Capture screenshots and include them in conversations
- 🖥️ **Desktop Integration**: Click-through transparent areas to interact with your desktop
- ⚡ **Zero Jumping**: Stable "Big Canvas" architecture prevents UI repositioning
- 🎯 **Context-Aware**: Access clipboard content and recent files for enhanced responses

---

## 🚀 Features

### AI Capabilities
- **Multiple AI Providers**: 
  - 🦙 **Ollama** - Run models locally (Llama 3.2, Mistral, etc.)
  - 🤖 **OpenAI** - GPT-4, GPT-3.5-turbo
  - 🧠 **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- **Streaming Responses**: Real-time token streaming for instant feedback
- **Context Management**: Include clipboard and file context in conversations
- **Chat History**: Full conversation tracking and history

### Interface & Interaction
- **Transparent Glass Panels**: Floating UI with custom brand color (`#123524`)
- **Click-Through Windows**: Interact with desktop through transparent areas
- **Smooth Animations**: Framer Motion powered transitions
- **Responsive Layout**: Adapts to content without clipping or jumping
- **Context Menu**: Quick access to clipboard and recent files
- **Settings Panel**: Configure AI providers, models, and API keys

### Input Methods
- **Text Input**: Rich textarea with multi-line support
- **Voice Recording**: Browser-based speech recognition
- **Camera Capture**: Take screenshots to include visual context
- **Keyboard Shortcuts**: 
  - `Enter` to send
  - `Shift + Enter` for new line

### macOS Integration
- **Frameless Window**: Custom title bar with window controls
- **Desktop Click-Through**: Transparent areas allow desktop interaction
- **Auto-Resize**: Window grows to fit content dynamically
- **Always on Top**: Overlay interface stays visible (optional)
- **Dark Mode**: Native macOS dark theme integration

---

## 🛠️ Tech Stack

### Frontend
- **[Next.js 15.2.4](https://nextjs.org/)** - React framework with App Router
- **[React 19](https://react.dev/)** - UI library
- **[TypeScript](https://www.typescriptlang.org/)** - Type safety
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first styling
- **[Shadcn UI](https://ui.shadcn.com/)** - Component library
- **[Framer Motion](https://www.framer.com/motion/)** - Animation library
- **[Zustand](https://github.com/pmndrs/zustand)** - State management

### Desktop
- **[Electron](https://www.electronjs.org/)** - Cross-platform desktop framework
- **IPC Communication** - Main/Renderer process bridge
- **Native APIs** - Clipboard, file system, window management

### AI Integration
- **Ollama API** - Local model inference
- **OpenAI API** - GPT models
- **Anthropic API** - Claude models
- **Streaming Support** - Real-time response streaming

### Development Tools
- **[pnpm](https://pnpm.io/)** - Fast, efficient package manager
- **[Concurrently](https://github.com/open-cli-tools/concurrently)** - Run multiple commands
- **[wait-on](https://github.com/jeffbski/wait-on)** - Wait for dev server
- **[Electron Builder](https://www.electron.build/)** - Package and distribute

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **pnpm** (v8 or higher) - Install with `npm install -g pnpm`
- **macOS** (12.0 Monterey or higher) - Required for Electron transparency features
- **Ollama** (Optional) - For local AI models - [Download](https://ollama.ai/)

### Optional Requirements
- **OpenAI API Key** - For GPT models
- **Anthropic API Key** - For Claude models
- **Microphone Access** - For voice input (browser will prompt)
- **Camera Access** - For screenshot capture (browser will prompt)

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/helloprkr/SENTER-ai.git
cd SENTER-ai
git checkout jp-v2
```

### 2. Install Dependencies

```bash
pnpm install
```

### 3. Configure Environment (Optional)

Create a `.env.local` file for default API keys:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-...

# Ollama Configuration (if running on custom port)
OLLAMA_ENDPOINT=http://localhost:11434
```

> **Note**: API keys can also be configured in the Settings panel within the app.

### 4. Start Ollama (Optional)

If using local models:

```bash
# Install Ollama from https://ollama.ai/
# Then pull a model
ollama pull llama3.2

# Start the Ollama server
ollama serve
```

---

## 🎮 Usage

### Development Mode

Run the app in development mode with hot-reload:

```bash
pnpm run electron:dev
```

This command:
1. Starts Next.js dev server on `http://localhost:3000` (or next available port)
2. Waits for the server to be ready
3. Launches Electron with the dev build

### Production Build

Build the app for production:

```bash
# Build Next.js static export
pnpm run build

# Build Electron app (macOS)
pnpm run electron:build
```

The packaged app will be available in the `dist` folder.

### Running Production Build Locally

```bash
# Start production server
pnpm start

# Or run Electron in production mode
pnpm run electron:start
```

---

## 🏗️ Architecture

### Big Canvas Design

SENTER uses a unique "Big Canvas" architecture to achieve stable, flicker-free transparency:

```
┌─────────────────────────────────────┐
│  Electron Window (85% screen height)│
│  ┌───────────────────────────────┐  │
│  │     Transparent Area          │  │ <- Click-through enabled
│  │  (Desktop visible)            │  │
│  │                               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │   Context Menu Panel    │  │  │ <- Glass panel (solid)
│  │  └─────────────────────────┘  │  │
│  │                               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │    Main Input Panel     │  │  │ <- Glass panel (solid)
│  │  └─────────────────────────┘  │  │
│  │                               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │    Chat View Panel      │  │  │ <- Glass panel (solid)
│  │  └─────────────────────────┘  │  │
│  │                               │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Key Principles:**
- Window size is **fixed** at 85% of screen height
- Content is positioned with **fixed CSS** (no viewport units)
- Window **never moves** or resizes (eliminates jumping)
- Click-through detects elements with `document.elementFromPoint()`

### Project Structure

```
senter-ai/
├── app/                      # Next.js App Router
│   ├── api/                  # API routes (chat streaming)
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Entry point
│   └── globals.css          # Global styles
├── components/              # React components
│   ├── camera-modal.tsx    # Screenshot capture
│   ├── chat-view.tsx       # Chat history panel
│   ├── context-menu.tsx    # Context selection menu
│   ├── settings-modal.tsx  # Settings configuration
│   └── ui/                 # Shadcn UI components
├── electron/               # Electron main process
│   ├── main.js            # Main process entry
│   └── preload.js         # Preload script (IPC bridge)
├── hooks/                 # Custom React hooks
│   ├── use-audio-recording.ts
│   ├── use-camera.ts
│   ├── use-electron-chat.ts
│   └── use-toast.ts
├── lib/                   # Utilities
│   ├── store.ts          # Zustand state management
│   ├── utils.ts          # Helper functions
│   └── whisper.ts        # Speech recognition
├── public/               # Static assets
├── styles/              # Additional styles
├── types/               # TypeScript definitions
├── liquid-glass-app.tsx # Main app component
└── package.json
```

### IPC Communication

```typescript
// Renderer Process (React)
window.electronAPI.resizeWindow(width, height)
window.electronAPI.chatStream({ messages, provider, model })

// Main Process (Electron)
ipcMain.on('resize-window', (event, width, height) => { ... })
ipcMain.handle('chat-stream', async (event, options) => { ... })
```

### State Management

```typescript
// Zustand store (lib/store.ts)
interface Store {
  settings: {
    provider: 'ollama' | 'openai' | 'anthropic'
    model: string
    apiKey?: string
  }
  isSettingsOpen: boolean
  isChatViewOpen: boolean
  isContextMenuOpen: boolean
}
```

---

## 🎨 Customization

### Brand Colors

Update the glassmorphism colors in `app/globals.css`:

```css
.glass-panel {
  background: rgba(18, 53, 36, 0.75) !important; /* Your brand color */
  backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid rgba(250, 250, 250, 0.12) !important;
  border-radius: 24px;
}
```

### Window Configuration

Modify Electron window settings in `electron/main.js`:

```javascript
const initialWidth = 800;
const initialHeight = Math.round(screenHeight * 0.85); // Adjust percentage
const initialY = Math.round(screenHeight * 0.05); // Adjust top margin
```

### AI Models

Add or modify models in `components/settings-modal.tsx`:

```typescript
const ollamaModels = [
  'llama3.2',
  'mistral',
  'codellama',
  'your-custom-model' // Add your model
]
```

---

## 🧪 Development

### Running Tests

```bash
# Run unit tests
pnpm test

# Run E2E tests
pnpm test:e2e

# Type checking
pnpm type-check
```

### Debugging

**Renderer Process (React):**
- Open DevTools: `Cmd+Option+I` in the Electron window
- Or enable in `electron/main.js`: `mainWindow.webContents.openDevTools()`

**Main Process (Electron):**
```bash
# Run with debugger
electron --inspect=5858 electron/main.js
```

Then attach Chrome DevTools to `chrome://inspect`

### Hot Reload

The development server includes:
- ✅ Next.js Fast Refresh (React components)
- ✅ CSS Hot Reload (Tailwind)
- ⚠️ Electron main process requires restart

To restart Electron without restarting Next.js:
1. Close Electron window
2. Run `pnpm run electron:start` in a new terminal

---

## 📦 Building & Distribution

### macOS App Bundle

```bash
# Build and package for macOS
pnpm run electron:build
```

Output: `dist/SENTER-1.0.0.dmg`

### Configuration

Edit `electron-builder.yml` to customize:

```yaml
appId: com.yourcompany.senter
productName: SENTER
directories:
  output: dist
files:
  - out/**/*
  - electron/**/*
  - node_modules/**/*
mac:
  category: public.app-category.productivity
  icon: build/icon.icns
  target:
    - dmg
    - zip
```

### Code Signing (Optional)

For distribution outside the Mac App Store:

```bash
# Install certificate
# Then build with signing
CSC_LINK=/path/to/certificate.p12 CSC_KEY_PASSWORD=password pnpm run electron:build
```

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Issue: "Port 3000 already in use"

**Solution:** The app automatically tries ports 3001, 3002, 3003. If all are in use:
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Issue: "Window is clipping content"

**Solution:** This should be resolved with the Big Canvas architecture. If still occurring:
1. Check that `electron/main.js` has `initialHeight = Math.round(screenHeight * 0.85)`
2. Verify `liquid-glass-app.tsx` uses fixed padding (e.g., `pt-32`) not viewport units
3. Ensure window `resizable: false` and `movable: false` in `main.js`

### Issue: "Click-through not working"

**Solution:**
1. Verify `setIgnoreMouseEvents` is called in `liquid-glass-app.tsx`
2. Check that transparent areas have `data-transparent-gap` attribute
3. Ensure panels have `data-ui-panel` attribute

### Issue: "Microphone not working"

**Solution:**
1. Grant microphone permissions: System Settings → Privacy & Security → Microphone
2. Check browser compatibility (Chrome/Edge work best)
3. Verify HTTPS or localhost (required for `getUserMedia`)

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pnpm test`
5. Commit with conventional commits: `git commit -m "feat: add amazing feature"`
6. Push to your fork: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### Code Style

- Use TypeScript for all new code
- Follow the existing code style (Prettier configured)
- Use functional components with hooks
- Prefer `const` over `let`
- Use descriptive variable names

### Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Request review from maintainers
5. Address review feedback
6. Squash commits before merging

---

## 📝 Roadmap

### v1.1 (Planned)
- [ ] Cross-platform support (Windows, Linux)
- [ ] Custom keyboard shortcuts
- [ ] Multiple chat sessions
- [ ] Export conversations
- [ ] Plugin system

### v1.2 (Planned)
- [ ] System tray integration
- [ ] Global hotkey activation
- [ ] File attachment support
- [ ] Code syntax highlighting
- [ ] Markdown rendering improvements

### v2.0 (Future)
- [ ] Team collaboration features
- [ ] Custom AI model fine-tuning
- [ ] Workflow automation
- [ ] Integration with productivity tools
- [ ] Mobile companion app

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 SENTER AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **[Next.js](https://nextjs.org/)** - Amazing React framework
- **[Electron](https://www.electronjs.org/)** - Cross-platform desktop apps
- **[Shadcn UI](https://ui.shadcn.com/)** - Beautiful component library
- **[Ollama](https://ollama.ai/)** - Local AI inference
- **[Vercel](https://vercel.com/)** - Inspiration from their design system
- **[Raycast](https://www.raycast.com/)** - Inspiration for launcher UX
- **macOS Spotlight** - Design inspiration

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/helloprkr/SENTER-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/helloprkr/SENTER-ai/discussions)
- **Email**: support@senter.ai
- **Twitter**: [@SenterAI](https://twitter.com/SenterAI)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=helloprkr/SENTER-ai&type=Date)](https://star-history.com/#helloprkr/SENTER-ai&Date)

---

<div align="center">

**Made with ❤️ by the SENTER Team**

[Website](https://senter.ai) • [Documentation](https://docs.senter.ai) • [Blog](https://blog.senter.ai)

</div>
