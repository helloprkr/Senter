import { motion } from 'framer-motion'
import { AlertCircle, RefreshCw, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { GlassPanel } from './GlassPanel'
import { IconButton } from './IconButton'

interface ErrorDisplayProps {
  title?: string
  message: string
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
}

export function ErrorDisplay({
  title = 'Something went wrong',
  message,
  onRetry,
  onDismiss,
  className,
}: ErrorDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn('w-full', className)}
    >
      <GlassPanel className="p-4 border-brand-accent/30">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-brand-accent/20 flex items-center justify-center">
            <AlertCircle className="w-4 h-4 text-brand-accent" strokeWidth={1.5} />
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium text-brand-light">{title}</h4>
            <p className="mt-1 text-xs text-brand-light/60">{message}</p>
          </div>
          <div className="flex items-center gap-1">
            {onRetry && (
              <IconButton
                icon={RefreshCw}
                size="sm"
                onClick={onRetry}
                className="hover:bg-brand-primary/20"
              />
            )}
            {onDismiss && (
              <IconButton
                icon={X}
                size="sm"
                onClick={onDismiss}
                className="hover:bg-brand-light/10"
              />
            )}
          </div>
        </div>
      </GlassPanel>
    </motion.div>
  )
}

interface ConnectionStatusProps {
  isConnected: boolean
  isConnecting?: boolean
  onReconnect?: () => void
  className?: string
}

export function ConnectionStatus({
  isConnected,
  isConnecting = false,
  onReconnect,
  className,
}: ConnectionStatusProps) {
  if (isConnected) return null

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className={cn('overflow-hidden', className)}
    >
      <div className="px-4 py-2 bg-brand-accent/10 border-b border-brand-accent/20 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              'w-2 h-2 rounded-full',
              isConnecting ? 'bg-yellow-500 animate-pulse' : 'bg-brand-accent'
            )}
          />
          <span className="text-xs text-brand-light/70">
            {isConnecting ? 'Connecting to Senter...' : 'Disconnected from Senter'}
          </span>
        </div>
        {!isConnecting && onReconnect && (
          <button
            onClick={onReconnect}
            className="text-xs text-brand-accent hover:text-brand-accent/80 transition-colors"
          >
            Reconnect
          </button>
        )}
      </div>
    </motion.div>
  )
}

interface ToastProps {
  type: 'success' | 'error' | 'info' | 'warning'
  message: string
  onClose?: () => void
}

export function Toast({ type, message, onClose }: ToastProps) {
  const colors = {
    success: 'bg-green-500/20 border-green-500/30 text-green-400',
    error: 'bg-brand-accent/20 border-brand-accent/30 text-brand-accent',
    info: 'bg-blue-500/20 border-blue-500/30 text-blue-400',
    warning: 'bg-yellow-500/20 border-yellow-500/30 text-yellow-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 50, scale: 0.95 }}
      className={cn(
        'fixed bottom-6 right-6 px-4 py-3 rounded-xl border backdrop-blur-xl z-50',
        colors[type]
      )}
    >
      <div className="flex items-center gap-3">
        <span className="text-sm">{message}</span>
        {onClose && (
          <button onClick={onClose} className="hover:opacity-70 transition-opacity">
            <X className="w-4 h-4" strokeWidth={1.5} />
          </button>
        )}
      </div>
    </motion.div>
  )
}
