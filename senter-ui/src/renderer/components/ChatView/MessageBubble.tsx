import { useState } from 'react'
import { motion } from 'framer-motion'
import { Search, Check, Loader2 } from 'lucide-react'
import { type Message } from '@/types'

interface MessageBubbleProps {
  message: Message
  onResearchClick?: (topic: string) => Promise<void>  // V3-009
}

export function MessageBubble({ message, onResearchClick }: MessageBubbleProps) {
  const isUser = message.type === 'user'
  const [researchPending, setResearchPending] = useState(false)
  const [researchQueued, setResearchQueued] = useState(false)

  // V3-009: Handle research suggestion click
  const handleResearchClick = async () => {
    if (!message.suggestedResearch || !onResearchClick) return

    setResearchPending(true)
    try {
      await onResearchClick(message.suggestedResearch.topic)
      setResearchQueued(true)
    } catch (error) {
      console.error('[MessageBubble] Research trigger failed:', error)
    } finally {
      setResearchPending(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-[70%] px-4 py-3 ${
          isUser
            ? 'bg-brand-accent/20 rounded-2xl rounded-tr-sm'
            : 'bg-brand-light/10 rounded-2xl rounded-tl-sm'
        }`}
      >
        <p className="text-sm text-brand-light">{message.content}</p>

        {/* V3-009: Research Suggestion CTA */}
        {!isUser && message.suggestedResearch && !researchQueued && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ delay: 0.5, duration: 0.3 }}
            className="mt-3 pt-3 border-t border-brand-light/10"
          >
            <button
              onClick={handleResearchClick}
              disabled={researchPending}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-accent/20 hover:bg-brand-accent/30 border border-brand-accent/30 transition-colors disabled:opacity-50 w-full"
            >
              {researchPending ? (
                <Loader2 className="w-3.5 h-3.5 text-brand-accent animate-spin" />
              ) : (
                <Search className="w-3.5 h-3.5 text-brand-accent" />
              )}
              <span className="text-xs text-brand-accent font-medium">
                {message.suggestedResearch.prompt}
              </span>
            </button>
            <p className="text-xs text-brand-light/40 mt-1 pl-1">
              {message.suggestedResearch.reason}
            </p>
          </motion.div>
        )}

        {/* V3-009: Research queued confirmation */}
        {!isUser && researchQueued && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 flex items-center gap-1.5 text-xs text-green-400"
          >
            <Check className="w-3 h-3" />
            Research queued! Check Tasks tab for progress.
          </motion.div>
        )}

        <span className="text-xs text-brand-light/50 mt-1 block">
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </motion.div>
  )
}
