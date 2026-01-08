// Message types for chat
export interface Message {
  id: string
  type: 'user' | 'bot'
  content: string
  timestamp: Date
}

export interface ConversationThread {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

// Research types (matching Python backend)
export interface ResearchResult {
  id: number
  topic: string
  summary: string
  sources: ResearchSource[]
  confidence: number
  createdAt: Date
  viewed: boolean
}

export interface ResearchSource {
  url: string
  title: string
  snippet?: string
}

// Context item types
export type ContextItemType = 'web' | 'file' | 'clipboard' | 'image' | 'code'

export interface ContextItem {
  id: string
  type: ContextItemType
  title: string
  description?: string
  content?: string
  url?: string
  createdAt: Date
}

// Navigation types
export type NavItemId = 'conversation' | 'vision' | 'research' | 'settings'

export interface NavItem {
  id: NavItemId
  label: string
  icon: string
  description?: string
  hasSubmenu?: boolean
}

// Backend status types
export interface BackendStatus {
  status: 'healthy' | 'degraded' | 'unreachable'
  unreadCount: number
  components?: Record<string, boolean>
}

// Tab types for chat view
export type ChatViewTab = 'history' | 'journal' | 'tasks' | 'insights' | 'goals'

// Task types
export interface Task {
  id: string
  title: string
  status: 'pending' | 'in_progress' | 'completed'
  dueDate?: Date
  subtasks?: Task[]
}

// Journal entry types
export interface JournalEntry {
  id: string
  date: Date
  summary: string
  mood?: 'positive' | 'neutral' | 'negative'
  content?: string
}

// Context filter types
export type ContextFilter = 'all' | 'web' | 'files' | 'clipboard'
