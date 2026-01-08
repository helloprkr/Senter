import { Settings } from 'lucide-react'

interface UserProfileProps {
  onSettingsClick?: () => void
}

export function UserProfile({ onSettingsClick }: UserProfileProps) {
  return (
    <div className="p-4 border-t border-brand-light/10">
      <div className="flex items-center gap-3">
        {/* Avatar */}
        <div className="w-10 h-10 rounded-full bg-brand-primary/30 flex items-center justify-center text-brand-light font-medium">
          U
        </div>

        {/* User Info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-brand-light truncate">User</p>
          <div className="flex items-center gap-2">
            {/* Status indicator */}
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-xs text-brand-light/60">Online</span>
          </div>
        </div>

        {/* Settings Button */}
        <button
          onClick={onSettingsClick}
          className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/[0.08] hover:bg-brand-light/10 transition-colors"
        >
          <Settings className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
        </button>
      </div>
    </div>
  )
}
