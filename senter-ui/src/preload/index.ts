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
      minimizeWindow: () => void
      maximizeWindow: () => void
      closeWindow: () => void
      onResearchUpdate: (callback: (data: unknown) => void) => void
      onStatusChange: (callback: (status: unknown) => void) => void
    }
  }
}
