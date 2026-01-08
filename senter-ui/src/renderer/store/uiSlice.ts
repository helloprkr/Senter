import { type StateCreator } from 'zustand'

export interface UISlice {
  // Panel visibility states
  isContextMenuOpen: boolean
  isChatViewOpen: boolean
  isSidebarOpen: boolean
  isListening: boolean
  isSettingsOpen: boolean

  // Panel actions - with mutual exclusivity
  toggleContextMenu: () => void
  toggleChatView: () => void
  toggleSidebar: () => void
  setListening: (listening: boolean) => void
  toggleSettings: () => void
  closeAllPanels: () => void
}

export const createUISlice: StateCreator<UISlice, [], [], UISlice> = (set) => ({
  // Initial states
  isContextMenuOpen: false,
  isChatViewOpen: false,
  isSidebarOpen: true,
  isListening: false,
  isSettingsOpen: false,

  // Actions with mutual exclusivity for context menu and chat view
  toggleContextMenu: () =>
    set((state) => ({
      isContextMenuOpen: !state.isContextMenuOpen,
      // Close chat view when opening context menu
      isChatViewOpen: state.isContextMenuOpen ? state.isChatViewOpen : false,
    })),

  toggleChatView: () =>
    set((state) => ({
      isChatViewOpen: !state.isChatViewOpen,
      // Close context menu when opening chat view
      isContextMenuOpen: state.isChatViewOpen ? state.isContextMenuOpen : false,
    })),

  toggleSidebar: () =>
    set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

  setListening: (listening) =>
    set({ isListening: listening }),

  toggleSettings: () =>
    set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),

  closeAllPanels: () =>
    set({
      isContextMenuOpen: false,
      isChatViewOpen: false,
      isSettingsOpen: false,
    }),
})
