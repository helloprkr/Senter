import { forwardRef } from 'react'
import { motion, type HTMLMotionProps } from 'framer-motion'
import { type LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface IconButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  icon: LucideIcon
  isActive?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const sizeClasses = {
  sm: 'w-6 h-6',
  md: 'w-8 h-8',
  lg: 'w-10 h-10',
}

const iconSizeClasses = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
}

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon: Icon, isActive = false, size = 'md', className, disabled, ...props }, ref) => {
    return (
      <motion.button
        ref={ref}
        whileHover={{ scale: disabled ? 1 : 1.05 }}
        whileTap={{ scale: disabled ? 1 : 0.95 }}
        disabled={disabled}
        className={cn(
          sizeClasses[size],
          'rounded-lg flex items-center justify-center transition-all duration-200',
          'bg-brand-light/[0.08] border border-brand-light/10',
          'hover:bg-brand-primary/10',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          isActive && 'bg-brand-primary/20 border-brand-primary/30',
          className
        )}
        {...props}
      >
        <Icon
          className={cn(
            iconSizeClasses[size],
            isActive ? 'text-brand-accent' : 'text-brand-light/70'
          )}
          strokeWidth={1.5}
        />
      </motion.button>
    )
  }
)

IconButton.displayName = 'IconButton'

export { IconButton }
export type { IconButtonProps }
