import { type StateCreator } from 'zustand'
import { type ResearchResult } from '@/types'

export interface ResearchSlice {
  // State
  researchResults: ResearchResult[]
  selectedResearch: ResearchResult | null
  unreadCount: number
  isResearching: boolean

  // Actions
  setResearchResults: (results: ResearchResult[]) => void
  addResearch: (result: ResearchResult) => void
  selectResearch: (id: number | null) => void
  markAsRead: (id: number) => void
  clearResearch: () => void
  setResearching: (researching: boolean) => void
}

export const createResearchSlice: StateCreator<ResearchSlice, [], [], ResearchSlice> = (set, get) => ({
  // Initial states
  researchResults: [],
  selectedResearch: null,
  unreadCount: 0,
  isResearching: false,

  // Actions
  setResearchResults: (results) =>
    set({
      researchResults: results,
      unreadCount: results.filter((r) => !r.viewed).length,
    }),

  addResearch: (result) =>
    set((state) => ({
      researchResults: [result, ...state.researchResults],
      unreadCount: state.unreadCount + (result.viewed ? 0 : 1),
    })),

  selectResearch: (id) => {
    if (id === null) {
      set({ selectedResearch: null })
      return
    }
    const research = get().researchResults.find((r) => r.id === id)
    if (research) {
      set({ selectedResearch: research })
      // Auto-mark as read when selected
      if (!research.viewed) {
        get().markAsRead(id)
      }
    }
  },

  markAsRead: (id) =>
    set((state) => ({
      researchResults: state.researchResults.map((r) =>
        r.id === id ? { ...r, viewed: true } : r
      ),
      unreadCount: Math.max(0, state.unreadCount - 1),
    })),

  clearResearch: () =>
    set({
      researchResults: [],
      selectedResearch: null,
      unreadCount: 0,
    }),

  setResearching: (researching) =>
    set({ isResearching: researching }),
})
