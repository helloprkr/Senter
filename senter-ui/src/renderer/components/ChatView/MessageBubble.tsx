import { motion } from 'framer-motion'
import { type Message } from '@/types'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.type === 'user'

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
