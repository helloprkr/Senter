import { type StateCreator } from 'zustand'
import { type NavItemId } from '@/types'

export interface NavigationSlice {
  // State
  activeNavItem: NavItemId
  isSettingsExpanded: boolean
  searchQuery: string
  isSearchFocused: boolean

  // Actions
  setActiveNavItem: (item: NavItemId) => void
  toggleSettingsExpanded: () => void
  setSearchQuery: (query: string) => void
  setSearchFocused: (focused: boolean) => void
  clearSearch: () => void
}

export const createNavigationSlice: StateCreator<NavigationSlice, [], [], NavigationSlice> = (set) => ({
  // Initial states
  activeNavItem: 'conversation',
  isSettingsExpanded: false,
  searchQuery: '',
  isSearchFocused: false,

  // Actions
  setActiveNavItem: (item) =>
    set({ activeNavItem: item }),

  toggleSettingsExpanded: () =>
    set((state) => ({ isSettingsExpanded: !state.isSettingsExpanded })),

  setSearchQuery: (query) =>
    set({ searchQuery: query }),

  setSearchFocused: (focused) =>
    set({ isSearchFocused: focused }),

  clearSearch: () =>
    set({ searchQuery: '', isSearchFocused: false }),
})
