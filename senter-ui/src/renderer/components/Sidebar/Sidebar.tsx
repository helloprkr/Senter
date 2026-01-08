import { motion, AnimatePresence } from 'framer-motion'
import { MessageCircle, X } from 'lucide-react'
import { useStore } from '@/store'
import { SearchBar } from './SearchBar'
import { Navigation } from './Navigation'
import { UserProfile } from './UserProfile'

interface SidebarProps {
  onSettingsClick?: () => void
}

export function Sidebar({ onSettingsClick }: SidebarProps) {
  const { isSidebarOpen, toggleSidebar } = useStore()

  return (
    <AnimatePresence>
      {isSidebarOpen && (
        <motion.div
          initial={{ x: -368, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: -368, opacity: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="fixed left-0 top-0 h-screen w-[368px] z-20"
        >
          <div className="h-full bg-brand-light/[0.08] backdrop-blur-3xl border-r border-brand-light/10 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-6">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="absolute inset-0 bg-brand-primary/20 rounded-xl blur-sm" />
                  <div className="relative w-10 h-10 bg-brand-primary/20 backdrop-blur-sm rounded-xl flex items-center justify-center border border-brand-primary/30">
                    <MessageCircle className="w-5 h-5 text-brand-light" strokeWidth={1.5} />
                  </div>
                </div>
                <h1 className="text-xl font-medium text-brand-light tracking-wide">SENTER</h1>
              </div>
              <button
                onClick={toggleSidebar}
                className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/[0.08] hover:bg-brand-light/10 transition-colors"
              >
                <X className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
              </button>
            </div>

            {/* Search Bar */}
            <SearchBar />

            {/* Navigation */}
            <Navigation />

            {/* User Profile */}
            <UserProfile onSettingsClick={onSettingsClick} />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
