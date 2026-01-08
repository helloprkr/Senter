import { type StateCreator } from 'zustand'
import { type ContextItem, type ContextFilter } from '@/types'

export interface ContextSlice {
  // State
  contextItems: ContextItem[]
  activeFilter: ContextFilter
  activeContext: ContextItem[]
  contextSearchQuery: string
  contextLoading: boolean

  // Actions
  addContextItem: (item: ContextItem) => void
  removeContextItem: (id: string) => void
  setFilter: (filter: ContextFilter) => void
  addToActiveContext: (item: ContextItem) => void
  removeFromActiveContext: (id: string) => void
  clearActiveContext: () => void
  setContextSearchQuery: (query: string) => void
  setContextItems: (items: ContextItem[]) => void
  setContextLoading: (loading: boolean) => void
}

export const createContextSlice: StateCreator<ContextSlice, [], [], ContextSlice> = (set) => ({
  // Initial states
  contextItems: [],
  activeFilter: 'all',
  activeContext: [],
  contextSearchQuery: '',
  contextLoading: false,

  // Actions
  addContextItem: (item) =>
    set((state) => ({
      contextItems: [item, ...state.contextItems],
    })),

  removeContextItem: (id) =>
    set((state) => ({
      contextItems: state.contextItems.filter((item) => item.id !== id),
      activeContext: state.activeContext.filter((item) => item.id !== id),
    })),

  setFilter: (filter) =>
    set({ activeFilter: filter }),

  addToActiveContext: (item) =>
    set((state) => {
      // Don't add duplicates
      if (state.activeContext.some((i) => i.id === item.id)) {
        return state
      }
      return { activeContext: [...state.activeContext, item] }
    }),

  removeFromActiveContext: (id) =>
    set((state) => ({
      activeContext: state.activeContext.filter((item) => item.id !== id),
    })),

  clearActiveContext: () =>
    set({ activeContext: [] }),

  setContextSearchQuery: (query) =>
    set({ contextSearchQuery: query }),

  // P3-001: New actions for daemon integration
  setContextItems: (items) =>
    set({ contextItems: items }),

  setContextLoading: (loading) =>
    set({ contextLoading: loading }),
})
