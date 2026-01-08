import { forwardRef, type ReactNode, type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface GlassPanelProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  withGlow?: boolean
  variant?: 'primary' | 'secondary'
}

const GlassPanel = forwardRef<HTMLDivElement, GlassPanelProps>(
  ({ children, className, withGlow = false, variant = 'primary', ...props }, ref) => {
    const baseStyles = variant === 'primary'
      ? 'bg-brand-light/[0.08] backdrop-blur-3xl border border-brand-light/10 rounded-3xl'
      : 'bg-brand-light/[0.05] backdrop-blur-2xl border border-brand-light/[0.08] rounded-2xl'

    const glowStyles = withGlow
      ? 'shadow-[inset_0_1px_0_0_rgba(250,250,250,0.1),_0_0_40px_rgba(18,53,36,0.2)]'
      : ''

    return (
      <div
        ref={ref}
        className={cn(baseStyles, glowStyles, className)}
        {...props}
      >
        {children}
      </div>
    )
  }
)

GlassPanel.displayName = 'GlassPanel'

export { GlassPanel }
export type { GlassPanelProps }
