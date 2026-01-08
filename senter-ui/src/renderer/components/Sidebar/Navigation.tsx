import { motion, AnimatePresence } from 'framer-motion'
import { MessageCircle, Camera, Search, Settings, ChevronDown } from 'lucide-react'
import { useStore } from '@/store'
import { type NavItemId } from '@/types'

const navItems = [
  { id: 'conversation' as NavItemId, icon: MessageCircle, label: 'Conversation' },
  { id: 'vision' as NavItemId, icon: Camera, label: 'Vision' },
  { id: 'research' as NavItemId, icon: Search, label: 'Research' },
  { id: 'settings' as NavItemId, icon: Settings, label: 'Settings', hasSubmenu: true },
]

export function Navigation() {
  const {
    activeNavItem,
    setActiveNavItem,
    isSettingsExpanded,
    toggleSettingsExpanded,
  } = useStore()

  const handleNavClick = (item: typeof navItems[0]) => {
    if (item.hasSubmenu) {
      toggleSettingsExpanded()
    } else {
      setActiveNavItem(item.id)
    }
  }

  return (
    <nav className="flex-1 px-4 space-y-1 overflow-y-auto">
      {navItems.map((item) => (
        <div key={item.id}>
          <motion.button
            onClick={() => handleNavClick(item)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
              activeNavItem === item.id && !item.hasSubmenu
                ? 'bg-brand-accent/20 border-l-2 border-brand-accent'
                : 'hover:bg-brand-primary/30 hover:translate-x-1'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <item.icon className="w-5 h-5 text-brand-light" strokeWidth={1.5} />
            <span className="text-sm font-light text-brand-light">{item.label}</span>
            {item.hasSubmenu && (
              <ChevronDown
                className={`w-4 h-4 ml-auto text-brand-light/60 transition-transform ${
                  isSettingsExpanded ? 'rotate-180' : ''
                }`}
                strokeWidth={1.5}
              />
            )}
          </motion.button>

          {/* Settings Submenu */}
          <AnimatePresence>
            {item.hasSubmenu && isSettingsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="ml-4 mt-1 space-y-1 overflow-hidden"
              >
                <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-brand-light/5 text-sm text-brand-light/80">
                  Preferences
                </button>
                <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-brand-light/5 text-sm text-brand-light/80">
                  API Keys
                </button>
                <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-brand-light/5 text-sm text-brand-light/80">
                  Integrations
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </nav>
  )
}
