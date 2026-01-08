import { useEffect, useCallback } from 'react'
import { useStore } from '@/store'

export function useKeyboardNavigation() {
  const {
    toggleSidebar,
    toggleContextMenu,
    toggleChatView,
    isContextMenuOpen,
    isChatViewOpen,
  } = useStore()

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Global shortcuts with Cmd/Ctrl
      if (e.metaKey || e.ctrlKey) {
        switch (e.key) {
          case 'b':
            // Toggle sidebar
            e.preventDefault()
            toggleSidebar()
            break
          case 'k':
            // Toggle context menu
            e.preventDefault()
            toggleContextMenu()
            break
          case 'j':
            // Toggle chat view
            e.preventDefault()
            toggleChatView()
            break
          case '/':
            // Focus input
            e.preventDefault()
            const input = document.querySelector('textarea')
            input?.focus()
            break
        }
      }

      // Escape to close menus
      if (e.key === 'Escape') {
        if (isChatViewOpen) {
          toggleChatView()
        } else if (isContextMenuOpen) {
          toggleContextMenu()
        }
      }
    },
    [toggleSidebar, toggleContextMenu, toggleChatView, isContextMenuOpen, isChatViewOpen]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

export const KEYBOARD_SHORTCUTS = [
  { keys: ['⌘', 'B'], description: 'Toggle sidebar' },
  { keys: ['⌘', 'K'], description: 'Toggle context menu' },
  { keys: ['⌘', 'J'], description: 'Toggle chat history' },
  { keys: ['⌘', '/'], description: 'Focus input' },
  { keys: ['⌘', '⇧', 'S'], description: 'Show/hide Senter (global)' },
  { keys: ['Esc'], description: 'Close menus' },
]
