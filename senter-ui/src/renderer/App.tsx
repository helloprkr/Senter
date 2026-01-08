import { useState } from 'react'
import { BackgroundEffects, ConnectionStatus } from '@/components/ui'
import { Sidebar } from '@/components/Sidebar'
import { InputBar } from '@/components/InputBar'
import { ContextMenu } from '@/components/ContextMenu'
import { ChatView } from '@/components/ChatView'
import { SettingsPanel } from '@/components/Settings'
import { useStore } from '@/store'
import { useKeyboardNavigation, useSenter } from '@/hooks'

function App() {
  const { isSidebarOpen } = useStore()
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const { isConnected, connect } = useSenter()

  // Enable keyboard shortcuts
  useKeyboardNavigation()

  return (
    <div className="min-h-screen bg-brand-dark relative overflow-hidden font-manrope">
      {/* Background Effects */}
      <BackgroundEffects />

      {/* Connection Status Banner */}
      <ConnectionStatus
        isConnected={isConnected}
        onReconnect={connect}
        className="fixed top-0 left-0 right-0 z-50"
      />

      {/* Sidebar */}
      <Sidebar onSettingsClick={() => setIsSettingsOpen(true)} />

      {/* Main Content Area */}
      <div
        className="relative z-10 flex items-center justify-center min-h-screen p-6 transition-all duration-300"
        style={{
          marginLeft: isSidebarOpen ? '368px' : '0',
        }}
      >
        <div className="relative max-w-lg w-full">
          {/* Context Menu (Expands Upward) */}
          <ContextMenu />

          {/* Input Bar */}
          <InputBar />

          {/* Chat View (Expands Downward) */}
          <ChatView />
        </div>
      </div>

      {/* Settings Panel */}
      <SettingsPanel isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  )
}

export default App
