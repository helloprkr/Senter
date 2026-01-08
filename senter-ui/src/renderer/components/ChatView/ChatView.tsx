import { useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import { useStore } from '@/store'
import { GlassPanel } from '@/components/ui'
import { MessageBubble } from './MessageBubble'
import { TypingIndicator } from './TypingIndicator'
import { RightPanel } from './RightPanel'

export function ChatView() {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const {
    isChatViewOpen,
    toggleChatView,
    messages,
    isTyping,
  } = useStore()

  // V3-009: Handle research suggestion click
  const handleResearchClick = useCallback(async (topic: string) => {
    try {
      console.log('[ChatView] Triggering research for:', topic)
      const response = await window.api.addResearchTask({
        description: `Research: ${topic}`,
        goal_id: 'suggested_research',
      })

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean }
        if (!res.success) {
          throw new Error('Research task creation failed')
        }
      }
    } catch (error) {
      console.error('[ChatView] Failed to create research task:', error)
      throw error
    }
  }, [])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isChatViewOpen && messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isChatViewOpen])

  return (
    <AnimatePresence>
      {isChatViewOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0, y: -20 }}
          animate={{ opacity: 1, height: 500, y: 0 }}
          exit={{ opacity: 0, height: 0, y: -20 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="absolute top-full left-0 right-0 mt-3 overflow-hidden z-30"
        >
          <GlassPanel withGlow className="h-full">
            <div className="flex h-full">
              {/* Main Chat Area */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Chat Header */}
                <div className="p-4 border-b border-brand-light/[0.08] flex items-center justify-between">
                  <h3 className="text-lg font-medium text-brand-light">Conversation</h3>
                  <button
                    onClick={toggleChatView}
                    className="w-6 h-6 rounded flex items-center justify-center hover:bg-brand-light/10 transition-colors"
                  >
                    <X className="w-4 h-4 text-brand-light/50" strokeWidth={1.5} />
                  </button>
                </div>

                {/* Messages Container */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  {messages.length === 0 ? (
                    <div className="text-center text-brand-light/40 py-8">
                      No messages yet. Start a conversation!
                    </div>
                  ) : (
                    messages.map((message) => (
                      <MessageBubble
                        key={message.id}
                        message={message}
                        onResearchClick={handleResearchClick}  // V3-009
                      />
                    ))
                  )}
                  {isTyping && <TypingIndicator />}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              {/* Right Panel with Tabs */}
              <RightPanel />
            </div>
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
