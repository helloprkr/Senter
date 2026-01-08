import { memo } from 'react'

const BackgroundEffects = memo(function BackgroundEffects() {
  return (
    <>
      {/* Gradient overlays */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_25%,#123524,transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_75%_75%,#D64933,transparent_70%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_48%,rgba(18,53,36,0.1)_50%,transparent_52%)]" />
      </div>

      {/* Animated glow orbs */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-brand-accent/[0.03] rounded-full blur-3xl" />
      </div>
    </>
  )
})

export { BackgroundEffects }
