interface ChatStreamOptions {
  messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>;
  provider: 'ollama' | 'openai' | 'anthropic' | 'google';
  model: string;
  apiKey?: string;
  ollamaEndpoint?: string;
  requestId: string;
}

interface ChatChunk {
  content: string;
  done: boolean;
  fullContent?: string;
}

interface ChatStreamResult {
  success: boolean;
  error?: string;
}

interface ElectronAPI {
  // Clipboard operations
  readClipboard: () => Promise<string>;
  readClipboardFormats: () => Promise<{
    formats: string[];
    text: string;
    html: string | null;
  }>;
  
  // File system operations
  readRecentFiles: () => Promise<Array<{
    name: string;
    path: string;
    modified: Date;
    size: number;
    extension: string;
  }>>;
  readFileContent: (filePath: string) => Promise<{
    success: boolean;
    content?: string;
    error?: string;
  }>;
  
  // App info
  getAppPaths: () => Promise<{
    userData: string;
    documents: string;
    desktop: string;
    home: string;
  }>;
  
  // Ollama status
  checkOllama: () => Promise<{
    running: boolean;
    models: Array<{ name: string; modified_at: string; size: number }>;
  }>;
  
  // Window controls
  windowMinimize: () => Promise<void>;
  windowMaximize: () => Promise<void>;
  windowClose: () => Promise<void>;
  
  // AI Chat streaming
  chatStream: (options: ChatStreamOptions) => Promise<ChatStreamResult>;
  chatCancel: (requestId: string) => Promise<{ success: boolean; error?: string }>;
  onChatChunk: (callback: (chunk: ChatChunk) => void) => () => void;
  
  // Platform detection
  platform: string;
  isElectron: boolean;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};

