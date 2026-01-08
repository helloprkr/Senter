import { contextBridge, ipcRenderer } from 'electron'

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('api', {
  // Senter backend communication
  sendQuery: (text: string, conversationId?: string) =>
    ipcRenderer.invoke('senter:query', text, conversationId),
  getResearch: (limit?: number) =>
    ipcRenderer.invoke('senter:research', limit),
  getStatus: () =>
    ipcRenderer.invoke('senter:status'),
  getHealth: () =>
    ipcRenderer.invoke('senter:health'),
  getConversations: (conversationId?: string, includeMessages?: boolean) =>
    ipcRenderer.invoke('senter:getConversations', conversationId, includeMessages),
  saveConversation: (conversation: {
    id?: string
    messages: Array<{ role: string; content: string }>
    focus?: string
    created?: string
  }) => ipcRenderer.invoke('senter:saveConversation', conversation),
  getTasks: (limit?: number) => ipcRenderer.invoke('senter:getTasks', limit),
  getJournal: (date?: string, limit?: number) => ipcRenderer.invoke('senter:getJournal', date, limit),
  generateJournal: (date?: string) => ipcRenderer.invoke('senter:generateJournal', date),

  // P3-001: Context Sources
  getContextSources: () => ipcRenderer.invoke('senter:getContextSources'),
  addContextSource: (source: {
    type: string
    title: string
    path: string
    description?: string
  }) => ipcRenderer.invoke('senter:addContextSource', source),
  removeContextSource: (id: string) => ipcRenderer.invoke('senter:removeContextSource', id),

  // P4-001: Voice Input
  startVoiceInput: () => ipcRenderer.invoke('senter:startVoiceInput'),
  stopVoiceInput: () => ipcRenderer.invoke('senter:stopVoiceInput'),
  getAudioStatus: () => ipcRenderer.invoke('senter:getAudioStatus'),

  // P4-002: Gaze Detection
  getGazeStatus: () => ipcRenderer.invoke('senter:getGazeStatus'),

  // V3-004: Insights (patterns, preferences, stats)
  getInsights: (days?: number) => ipcRenderer.invoke('senter:getInsights', days),

  // V3-005: Goals
  getGoals: (status?: string, limit?: number) => ipcRenderer.invoke('senter:getGoals', status, limit),
  updateGoal: (goalId: string, status: string) => ipcRenderer.invoke('senter:updateGoal', goalId, status),

  // Window controls
  minimizeWindow: () => ipcRenderer.send('window:minimize'),
  maximizeWindow: () => ipcRenderer.send('window:maximize'),
  closeWindow: () => ipcRenderer.send('window:close'),

  // Event listeners
  onResearchUpdate: (callback: (data: unknown) => void) => {
    ipcRenderer.on('senter:research-update', (_, data) => callback(data))
  },
  onStatusChange: (callback: (status: unknown) => void) => {
    ipcRenderer.on('senter:status-change', (_, status) => callback(status))
  },
})

// Type definitions for the exposed API
declare global {
  interface Window {
    api: {
      sendQuery: (text: string, conversationId?: string) => Promise<unknown>
      getResearch: (limit?: number) => Promise<unknown>
      getStatus: () => Promise<unknown>
      getHealth: () => Promise<unknown>
      getConversations: (conversationId?: string, includeMessages?: boolean) => Promise<unknown>
      saveConversation: (conversation: {
        id?: string
        messages: Array<{ role: string; content: string }>
        focus?: string
        created?: string
      }) => Promise<unknown>
      getTasks: (limit?: number) => Promise<unknown>
      getJournal: (date?: string, limit?: number) => Promise<unknown>
      generateJournal: (date?: string) => Promise<unknown>
      getContextSources: () => Promise<unknown>
      addContextSource: (source: {
        type: string
        title: string
        path: string
        description?: string
      }) => Promise<unknown>
      removeContextSource: (id: string) => Promise<unknown>
      startVoiceInput: () => Promise<unknown>
      stopVoiceInput: () => Promise<unknown>
      getAudioStatus: () => Promise<unknown>
      getGazeStatus: () => Promise<unknown>
      getInsights: (days?: number) => Promise<unknown>
      getGoals: (status?: string, limit?: number) => Promise<unknown>
      updateGoal: (goalId: string, status: string) => Promise<unknown>
      minimizeWindow: () => void
      maximizeWindow: () => void
      closeWindow: () => void
      onResearchUpdate: (callback: (data: unknown) => void) => void
      onStatusChange: (callback: (status: unknown) => void) => void
    }
  }
}
