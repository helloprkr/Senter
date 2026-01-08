import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { createUISlice, type UISlice } from './uiSlice'
import { createChatSlice, type ChatSlice } from './chatSlice'
import { createResearchSlice, type ResearchSlice } from './researchSlice'
import { createNavigationSlice, type NavigationSlice } from './navigationSlice'
import { createContextSlice, type ContextSlice } from './contextSlice'

// Combined store type
export type AppStore = UISlice & ChatSlice & ResearchSlice & NavigationSlice & ContextSlice

// Create the combined store
export const useStore = create<AppStore>()(
  devtools(
    persist(
      (...a) => ({
        ...createUISlice(...a),
        ...createChatSlice(...a),
        ...createResearchSlice(...a),
        ...createNavigationSlice(...a),
        ...createContextSlice(...a),
      }),
      {
        name: 'senter-storage',
        // Only persist certain states
        partialize: (state) => ({
          isSidebarOpen: state.isSidebarOpen,
          activeNavItem: state.activeNavItem,
          activeTab: state.activeTab,
        }),
      }
    ),
    { name: 'Senter Store' }
  )
)

// Re-export slice types for convenience
export type { UISlice } from './uiSlice'
export type { ChatSlice } from './chatSlice'
export type { ResearchSlice } from './researchSlice'
export type { NavigationSlice } from './navigationSlice'
export type { ContextSlice } from './contextSlice'
