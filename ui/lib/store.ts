"use client";

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { SenterGoal, SenterSuggestion, SenterActivity, SenterStatus, SenterAwayInfo } from './senter-api';

// Types
export type AIProvider = 'ollama' | 'openai' | 'anthropic' | 'google';

export interface Settings {
  provider: AIProvider;
  model: string;
  apiKey?: string;
  ollamaEndpoint: string;
  ollamaModels: string[];
  theme: 'dark' | 'light' | 'system';
  whisperModelLoaded: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: Date;
  imageData?: string; // Base64 image data for vision
}

export interface JournalEntry {
  id: string;
  content: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Task {
  id: string;
  title: string;
  completed: boolean;
  createdAt: Date;
}

interface AppState {
  // Settings
  settings: Settings;
  updateSettings: (settings: Partial<Settings>) => void;
  
  // Current conversation
  currentConversationId: string | null;
  setCurrentConversationId: (id: string | null) => void;
  
  // Conversations
  conversations: Conversation[];
  addConversation: (conversation: Conversation) => void;
  updateConversation: (id: string, updates: Partial<Conversation>) => void;
  deleteConversation: (id: string) => void;
  
  // Messages (in-memory, for current session)
  messages: Message[];
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  setMessages: (messages: Message[]) => void;
  
  // Journal entries
  journalEntries: JournalEntry[];
  addJournalEntry: (entry: JournalEntry) => void;
  updateJournalEntry: (id: string, updates: Partial<JournalEntry>) => void;
  deleteJournalEntry: (id: string) => void;
  
  // Tasks
  tasks: Task[];
  addTask: (task: Task) => void;
  toggleTask: (id: string) => void;
  deleteTask: (id: string) => void;
  
  // UI State
  isSettingsOpen: boolean;
  setIsSettingsOpen: (open: boolean) => void;
  isChatViewOpen: boolean;
  setIsChatViewOpen: (open: boolean) => void;
  isContextMenuOpen: boolean;
  setIsContextMenuOpen: (open: boolean) => void;
  isCameraOpen: boolean;
  setIsCameraOpen: (open: boolean) => void;
  
  // Connection status
  ollamaConnected: boolean;
  setOllamaConnected: (connected: boolean) => void;

  // Senter state
  senterConnected: boolean;
  setSenterConnected: (connected: boolean) => void;
  senterGoals: SenterGoal[];
  setSenterGoals: (goals: SenterGoal[]) => void;
  senterSuggestions: SenterSuggestion[];
  setSenterSuggestions: (suggestions: SenterSuggestion[]) => void;
  senterActivity: SenterActivity | null;
  setSenterActivity: (activity: SenterActivity | null) => void;
  senterStatus: SenterStatus | null;
  setSenterStatus: (status: SenterStatus | null) => void;
  senterAwayInfo: SenterAwayInfo | null;
  setSenterAwayInfo: (info: SenterAwayInfo | null) => void;
  showAwayModal: boolean;
  setShowAwayModal: (show: boolean) => void;
}

// Default settings
const defaultSettings: Settings = {
  provider: 'ollama',
  model: 'llama3.2:1b',
  ollamaEndpoint: 'http://localhost:11434',
  ollamaModels: [],
  theme: 'dark',
  whisperModelLoaded: false,
};

// Generate unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Settings
      settings: defaultSettings,
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),
      
      // Current conversation
      currentConversationId: null,
      setCurrentConversationId: (id) => set({ currentConversationId: id }),
      
      // Conversations
      conversations: [],
      addConversation: (conversation) =>
        set((state) => ({
          conversations: [conversation, ...state.conversations],
        })),
      updateConversation: (id, updates) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, ...updates } : c
          ),
        })),
      deleteConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.filter((c) => c.id !== id),
          messages: state.messages.filter((m) => m.conversationId !== id),
        })),
      
      // Messages
      messages: [],
      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message],
        })),
      clearMessages: () => set({ messages: [] }),
      setMessages: (messages) => set({ messages }),
      
      // Journal entries
      journalEntries: [],
      addJournalEntry: (entry) =>
        set((state) => ({
          journalEntries: [entry, ...state.journalEntries],
        })),
      updateJournalEntry: (id, updates) =>
        set((state) => ({
          journalEntries: state.journalEntries.map((e) =>
            e.id === id ? { ...e, ...updates, updatedAt: new Date() } : e
          ),
        })),
      deleteJournalEntry: (id) =>
        set((state) => ({
          journalEntries: state.journalEntries.filter((e) => e.id !== id),
        })),
      
      // Tasks
      tasks: [],
      addTask: (task) =>
        set((state) => ({
          tasks: [task, ...state.tasks],
        })),
      toggleTask: (id) =>
        set((state) => ({
          tasks: state.tasks.map((t) =>
            t.id === id ? { ...t, completed: !t.completed } : t
          ),
        })),
      deleteTask: (id) =>
        set((state) => ({
          tasks: state.tasks.filter((t) => t.id !== id),
        })),
      
      // UI State
      isSettingsOpen: false,
      setIsSettingsOpen: (open) => set({ isSettingsOpen: open }),
      isChatViewOpen: false,
      setIsChatViewOpen: (open) => set({ isChatViewOpen: open }),
      isContextMenuOpen: false,
      setIsContextMenuOpen: (open) => set({ isContextMenuOpen: open }),
      isCameraOpen: false,
      setIsCameraOpen: (open) => set({ isCameraOpen: open }),
      
      // Connection status
      ollamaConnected: false,
      setOllamaConnected: (connected) => set({ ollamaConnected: connected }),

      // Senter state
      senterConnected: false,
      setSenterConnected: (connected) => set({ senterConnected: connected }),
      senterGoals: [],
      setSenterGoals: (goals) => set({ senterGoals: goals }),
      senterSuggestions: [],
      setSenterSuggestions: (suggestions) => set({ senterSuggestions: suggestions }),
      senterActivity: null,
      setSenterActivity: (activity) => set({ senterActivity: activity }),
      senterStatus: null,
      setSenterStatus: (status) => set({ senterStatus: status }),
      senterAwayInfo: null,
      setSenterAwayInfo: (info) => set({ senterAwayInfo: info }),
      showAwayModal: false,
      setShowAwayModal: (show) => set({ showAwayModal: show }),
    }),
    {
      name: 'senter-storage',
      partialize: (state) => ({
        settings: state.settings,
        conversations: state.conversations,
        messages: state.messages,
        journalEntries: state.journalEntries,
        tasks: state.tasks,
      }),
    }
  )
);

// Utility hook for creating new items
export function createConversation(title?: string): Conversation {
  const now = new Date();
  return {
    id: generateId(),
    title: title || `Conversation ${now.toLocaleDateString()}`,
    createdAt: now,
    updatedAt: now,
  };
}

export function createMessage(
  conversationId: string,
  role: 'user' | 'assistant' | 'system',
  content: string,
  imageData?: string
): Message {
  return {
    id: generateId(),
    conversationId,
    role,
    content,
    createdAt: new Date(),
    imageData,
  };
}

export function createJournalEntry(content: string): JournalEntry {
  const now = new Date();
  return {
    id: generateId(),
    content,
    createdAt: now,
    updatedAt: now,
  };
}

export function createTask(title: string): Task {
  return {
    id: generateId(),
    title,
    completed: false,
    createdAt: new Date(),
  };
}

