const { app, BrowserWindow, ipcMain, clipboard, nativeTheme, screen } = require('electron');
const path = require('path');
const fs = require('fs');

let mainWindow;

async function createWindow() {
  // Set dark mode
  nativeTheme.themeSource = 'dark';

  // Get screen dimensions
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  // FIXED-SIZE MOVABLE WINDOW approach:
  // Window is tall enough to contain all panels (Context Menu + Main + Chat View)
  // without needing runtime resize. User can drag to reposition.
  const windowWidth = 600;
  const windowHeight = 850; // Tall enough for Context Menu (~300px) + Main (~200px) + Chat (~350px)

  // Center horizontally, position in upper third of screen
  const initialX = Math.round((screenWidth - windowWidth) / 2);
  const initialY = Math.round(screenHeight * 0.15);

  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: initialX,
    y: initialY,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    hasShadow: false,
    resizable: false,  // Fixed size - no dynamic resizing needed
    movable: true,     // User can drag to reposition
    // macOS: Don't use vibrancy at window level - apply it per-panel in CSS instead
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true
    }
  });

  // Determine if we're in development mode
  const isDev = !app.isPackaged && process.env.NODE_ENV !== 'production';
  
  if (isDev) {
    // Development: load from Next.js dev server
    const tryPorts = [3002, 3001, 3000, 3003];
    let loaded = false;
    
    for (const port of tryPorts) {
      try {
        await mainWindow.loadURL(`http://localhost:${port}`);
        console.log(`Successfully loaded from port ${port}`);
        loaded = true;
        break;
      } catch (err) {
        console.log(`Port ${port} not available, trying next...`);
      }
    }
    
    if (!loaded) {
      console.error('Could not connect to Next.js dev server');
      mainWindow.loadURL('http://localhost:3000');
    }
  } else {
    // Production: load from the static export
    // When packaged, files are in resources/out/ or adjacent to the app
    let staticPath;
    if (app.isPackaged) {
      // In packaged app, look in resources folder
      staticPath = path.join(process.resourcesPath, 'out', 'index.html');
    } else {
      // Running with NODE_ENV=production but not packaged
      staticPath = path.join(__dirname, '../out/index.html');
    }
    
    console.log('Loading static file from:', staticPath);
    await mainWindow.loadFile(staticPath);
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ============================================
// IPC Handlers for Native Features
// ============================================

// Read current clipboard text
ipcMain.handle('read-clipboard', async () => {
  try {
    return clipboard.readText();
  } catch (error) {
    console.error('Error reading clipboard:', error);
    return '';
  }
});

// Read clipboard with format info
ipcMain.handle('read-clipboard-formats', async () => {
  try {
    const formats = clipboard.availableFormats();
    const text = clipboard.readText();
    const html = clipboard.readHTML();
    
    return {
      formats,
      text,
      html: html || null
    };
  } catch (error) {
    console.error('Error reading clipboard formats:', error);
    return { formats: [], text: '', html: null };
  }
});

// Read recent files from Documents folder
ipcMain.handle('read-recent-files', async () => {
  try {
    const documentsPath = app.getPath('documents');
    const files = fs.readdirSync(documentsPath);
    
    const recentFiles = files
      .map(file => {
        try {
          const filePath = path.join(documentsPath, file);
          const stats = fs.statSync(filePath);
          
          // Only include files, not directories
          if (!stats.isFile()) return null;
          
          return {
            name: file,
            path: filePath,
            modified: stats.mtime,
            size: stats.size,
            extension: path.extname(file).toLowerCase()
          };
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .sort((a, b) => b.modified - a.modified)
      .slice(0, 10);
    
    return recentFiles;
  } catch (error) {
    console.error('Error reading recent files:', error);
    return [];
  }
});

// Read file content (for context)
ipcMain.handle('read-file-content', async (event, filePath) => {
  try {
    // Only allow reading from safe locations
    const documentsPath = app.getPath('documents');
    const desktopPath = app.getPath('desktop');
    
    if (!filePath.startsWith(documentsPath) && !filePath.startsWith(desktopPath)) {
      throw new Error('Access denied: File outside allowed directories');
    }
    
    const content = fs.readFileSync(filePath, 'utf-8');
    return { success: true, content };
  } catch (error) {
    console.error('Error reading file:', error);
    return { success: false, error: error.message };
  }
});

// Get app paths
ipcMain.handle('get-app-paths', async () => {
  return {
    userData: app.getPath('userData'),
    documents: app.getPath('documents'),
    desktop: app.getPath('desktop'),
    home: app.getPath('home')
  };
});

// Check if Ollama is running
ipcMain.handle('check-ollama', async () => {
  try {
    const response = await fetch('http://localhost:11434/api/tags');
    if (response.ok) {
      const data = await response.json();
      return { running: true, models: data.models || [] };
    }
    return { running: false, models: [] };
  } catch {
    return { running: false, models: [] };
  }
});

// Window control handlers
ipcMain.handle('window-minimize', () => {
  mainWindow?.minimize();
});

ipcMain.handle('window-maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow?.maximize();
  }
});

ipcMain.handle('window-close', () => {
  mainWindow?.close();
});

// Click-through handler for transparent window gaps
// When hovering over empty space, allow clicks to pass through to desktop
ipcMain.on('set-ignore-mouse-events', (event, ignore, options = {}) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (win) {
    win.setIgnoreMouseEvents(ignore, options);
  }
});

// Dynamic window resize handler (Static Full-Screen Canvas architecture)
// With the full-screen canvas approach, no resizing is ever needed.
// The window covers the entire screen; UI panels are positioned with CSS.
// This handler is kept for API compatibility but does nothing.
ipcMain.on('resize-window', () => {
  // No-op: Window is full-screen and static. No resizing needed.
});

// ============================================
// AI Chat Streaming Handler
// ============================================

const SYSTEM_PROMPT = `You are Senter, a thoughtful, intelligent, and helpful AI assistant. You are running locally on the user's Mac and have access to their context (clipboard, recent files, etc.).

Your personality:
- Be concise but thorough when needed
- Be friendly and conversational
- Help with coding, writing, analysis, and creative tasks
- When you don't know something, say so honestly
- Format code with proper syntax highlighting using markdown code blocks`;

// Active abort controllers for cancellation
const activeStreams = new Map();

// Chat streaming handler - streams responses back to renderer
ipcMain.handle('chat-stream', async (event, { messages, provider, model, apiKey, ollamaEndpoint, requestId }) => {
  const abortController = new AbortController();
  activeStreams.set(requestId, abortController);
  
  try {
    const effectiveModel = model || 'llama3.2';
    const effectiveEndpoint = ollamaEndpoint || 'http://localhost:11434';
    
    let response;
    
    switch (provider) {
      case 'ollama':
        response = await streamOllama(event, messages, effectiveModel, effectiveEndpoint, abortController.signal);
        break;
      case 'openai':
        if (!apiKey) {
          throw new Error('API key required for OpenAI');
        }
        response = await streamOpenAI(event, messages, effectiveModel || 'gpt-4', apiKey, abortController.signal);
        break;
      case 'anthropic':
        if (!apiKey) {
          throw new Error('API key required for Anthropic');
        }
        response = await streamAnthropic(event, messages, effectiveModel || 'claude-3-sonnet-20240229', apiKey, abortController.signal);
        break;
      default:
        response = await streamOllama(event, messages, effectiveModel, effectiveEndpoint, abortController.signal);
    }
    
    return { success: true };
  } catch (error) {
    if (error.name === 'AbortError') {
      return { success: false, error: 'Request cancelled' };
    }
    
    console.error('Chat stream error:', error);
    
    // Check for common Ollama errors
    if (error.message?.includes('ECONNREFUSED') || error.message?.includes('fetch failed')) {
      return { 
        success: false, 
        error: 'Cannot connect to Ollama. Make sure Ollama is running (ollama serve).' 
      };
    }
    
    return { success: false, error: error.message || 'Unknown error occurred' };
  } finally {
    activeStreams.delete(requestId);
  }
});

// Cancel an active stream
ipcMain.handle('chat-cancel', async (event, { requestId }) => {
  const controller = activeStreams.get(requestId);
  if (controller) {
    controller.abort();
    activeStreams.delete(requestId);
    return { success: true };
  }
  return { success: false, error: 'No active stream found' };
});

// Stream from Ollama API
async function streamOllama(event, messages, model, endpoint, signal) {
  const formattedMessages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...messages.map(m => ({ role: m.role, content: m.content }))
  ];
  
  const response = await fetch(`${endpoint}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: formattedMessages,
      stream: true
    }),
    signal
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Ollama error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n').filter(line => line.trim());
    
    for (const line of lines) {
      try {
        const data = JSON.parse(line);
        if (data.message?.content) {
          fullContent += data.message.content;
          event.sender.send('chat-chunk', { 
            content: data.message.content, 
            done: false 
          });
        }
        if (data.done) {
          event.sender.send('chat-chunk', { content: '', done: true, fullContent });
        }
      } catch (e) {
        // Skip invalid JSON lines
      }
    }
  }
  
  return fullContent;
}

// Stream from OpenAI API
async function streamOpenAI(event, messages, model, apiKey, signal) {
  const formattedMessages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...messages.map(m => ({ role: m.role, content: m.content }))
  ];
  
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: formattedMessages,
      stream: true
    }),
    signal
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n').filter(line => line.trim() && line.startsWith('data: '));
    
    for (const line of lines) {
      const data = line.slice(6); // Remove 'data: ' prefix
      if (data === '[DONE]') {
        event.sender.send('chat-chunk', { content: '', done: true, fullContent });
        continue;
      }
      
      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) {
          fullContent += content;
          event.sender.send('chat-chunk', { content, done: false });
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
  
  return fullContent;
}

// Stream from Anthropic API
async function streamAnthropic(event, messages, model, apiKey, signal) {
  const formattedMessages = messages.map(m => ({ role: m.role, content: m.content }));
  
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: formattedMessages,
      stream: true
    }),
    signal
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic error: ${response.status} - ${errorText}`);
  }
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = '';
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n').filter(line => line.trim() && line.startsWith('data: '));
    
    for (const line of lines) {
      const data = line.slice(6);
      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
          fullContent += parsed.delta.text;
          event.sender.send('chat-chunk', { content: parsed.delta.text, done: false });
        }
        if (parsed.type === 'message_stop') {
          event.sender.send('chat-chunk', { content: '', done: true, fullContent });
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }
  
  return fullContent;
}
