const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Clipboard operations
  readClipboard: () => ipcRenderer.invoke('read-clipboard'),
  readClipboardFormats: () => ipcRenderer.invoke('read-clipboard-formats'),
  
  // File system operations
  readRecentFiles: () => ipcRenderer.invoke('read-recent-files'),
  readFileContent: (filePath) => ipcRenderer.invoke('read-file-content', filePath),
  
  // App info
  getAppPaths: () => ipcRenderer.invoke('get-app-paths'),
  
  // Ollama status
  checkOllama: () => ipcRenderer.invoke('check-ollama'),
  
  // Window controls
  windowMinimize: () => ipcRenderer.invoke('window-minimize'),
  windowMaximize: () => ipcRenderer.invoke('window-maximize'),
  windowClose: () => ipcRenderer.invoke('window-close'),

  // Click-through for transparent window gaps
  // When true with forward option, mouse events pass through to desktop but still track position
  setIgnoreMouseEvents: (ignore, options = {}) => {
    ipcRenderer.send('set-ignore-mouse-events', ignore, options);
  },

  // Dynamic window resize (Big Canvas architecture)
  // With the big canvas approach, window position never changes
  // Only grows if content exceeds bounds
  resizeWindow: (width, height) => {
    ipcRenderer.send('resize-window', width, height);
  },
  
  // AI Chat streaming
  chatStream: (options) => ipcRenderer.invoke('chat-stream', options),
  chatCancel: (requestId) => ipcRenderer.invoke('chat-cancel', { requestId }),
  onChatChunk: (callback) => {
    const handler = (event, data) => callback(data);
    ipcRenderer.on('chat-chunk', handler);
    return () => ipcRenderer.removeListener('chat-chunk', handler);
  },
  
  // Platform detection
  platform: process.platform,
  
  // Check if running in Electron
  isElectron: true
});

// Notify the renderer that preload is ready
window.addEventListener('DOMContentLoaded', () => {
  console.log('Electron preload script loaded');
});

