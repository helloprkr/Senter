import { motion, AnimatePresence } from 'framer-motion'
import { X, Keyboard, Bell, Mic, Eye, Info } from 'lucide-react'
import { GlassPanel, IconButton } from '@/components/ui'
import { KEYBOARD_SHORTCUTS } from '@/hooks'

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
}

export function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed right-0 top-0 h-screen w-[400px] z-50"
          >
            <GlassPanel className="h-full rounded-none border-l border-brand-light/10">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-brand-light/10">
                <h2 className="text-lg font-medium text-brand-light">Settings</h2>
                <IconButton icon={X} onClick={onClose} />
              </div>

              {/* Content */}
              <div className="p-6 overflow-y-auto h-[calc(100%-80px)]">
                {/* Keyboard Shortcuts Section */}
                <SettingsSection title="Keyboard Shortcuts" icon={Keyboard}>
                  <div className="space-y-3">
                    {KEYBOARD_SHORTCUTS.map((shortcut, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span className="text-sm text-brand-light/70">{shortcut.description}</span>
                        <div className="flex items-center gap-1">
                          {shortcut.keys.map((key, j) => (
                            <kbd
                              key={j}
                              className="px-2 py-1 text-xs bg-brand-light/10 rounded border border-brand-light/20 text-brand-light/80"
                            >
                              {key}
                            </kbd>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </SettingsSection>

                {/* Notifications Section */}
                <SettingsSection title="Notifications" icon={Bell}>
                  <SettingsToggle
                    label="Desktop notifications"
                    description="Show notifications for research results"
                    defaultChecked={true}
                  />
                  <SettingsToggle
                    label="Sound effects"
                    description="Play sounds for events"
                    defaultChecked={false}
                  />
                </SettingsSection>

                {/* Voice Section */}
                <SettingsSection title="Voice" icon={Mic}>
                  <SettingsToggle
                    label="Voice activation"
                    description="Enable push-to-talk for voice input"
                    defaultChecked={true}
                  />
                  <SettingsToggle
                    label="Text-to-speech"
                    description="Read responses aloud"
                    defaultChecked={false}
                  />
                </SettingsSection>

                {/* Appearance Section */}
                <SettingsSection title="Appearance" icon={Eye}>
                  <SettingsToggle
                    label="Compact mode"
                    description="Use smaller UI elements"
                    defaultChecked={false}
                  />
                  <SettingsToggle
                    label="Show sidebar by default"
                    description="Keep sidebar open on launch"
                    defaultChecked={true}
                  />
                </SettingsSection>

                {/* About Section */}
                <SettingsSection title="About" icon={Info}>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-brand-light/70">Version</span>
                      <span className="text-sm text-brand-light">1.0.0</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-brand-light/70">Build</span>
                      <span className="text-sm text-brand-light font-mono">2026.01.08</span>
                    </div>
                  </div>
                </SettingsSection>
              </div>
            </GlassPanel>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

interface SettingsSectionProps {
  title: string
  icon: React.ElementType
  children: React.ReactNode
}

function SettingsSection({ title, icon: Icon, children }: SettingsSectionProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 mb-4">
        <Icon className="w-4 h-4 text-brand-accent" strokeWidth={1.5} />
        <h3 className="text-sm font-medium text-brand-light">{title}</h3>
      </div>
      <div className="space-y-4">{children}</div>
    </div>
  )
}

interface SettingsToggleProps {
  label: string
  description: string
  defaultChecked?: boolean
  onChange?: (checked: boolean) => void
}

function SettingsToggle({ label, description, defaultChecked = false, onChange }: SettingsToggleProps) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <input
        type="checkbox"
        defaultChecked={defaultChecked}
        onChange={(e) => onChange?.(e.target.checked)}
        className="mt-1 w-4 h-4 rounded border-brand-light/30 bg-brand-light/10 text-brand-accent focus:ring-brand-accent/50"
      />
      <div>
        <div className="text-sm text-brand-light group-hover:text-brand-light/90 transition-colors">
          {label}
        </div>
        <div className="text-xs text-brand-light/50">{description}</div>
      </div>
    </label>
  )
}
