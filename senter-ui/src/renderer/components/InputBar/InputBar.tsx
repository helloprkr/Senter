import { useRef, type KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageCircle, Mic, Camera, Send, History, Menu } from 'lucide-react'
import { useStore } from '@/store'
import { GlassPanel, IconButton } from '@/components/ui'
import { RainbowBorder } from './RainbowBorder'
import { useSenter } from '@/hooks'

export function InputBar() {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { sendQuery } = useSenter()

  const {
    inputValue,
    setInputValue,
    isListening,
    setListening,
    isContextMenuOpen,
    toggleContextMenu,
    isChatViewOpen,
    toggleChatView,
  } = useStore()

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    const message = inputValue.trim()
    setInputValue('')

    // Auto-resize textarea back to single line
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    // Send to backend (this adds user message and bot response)
    await sendQuery(message)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value)

    // Auto-resize
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 96) + 'px'
  }

  return (
    <div className="relative max-w-lg w-full">
      {/* Outer Glow */}
      <div className="absolute -inset-1 bg-gradient-to-r from-brand-primary/20 via-brand-light/10 to-brand-accent/20 rounded-3xl blur-xl opacity-60" />
      <div className="absolute -inset-2 bg-brand-primary/5 rounded-3xl blur-2xl" />

      {/* Main Container with Rainbow Border when listening */}
      <div className="relative">
        <AnimatePresence>
          {isListening && <RainbowBorder />}
        </AnimatePresence>

        <GlassPanel withGlow className="overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-brand-light/[0.08]">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-brand-primary/20 rounded-xl blur-sm" />
                <div className="relative w-10 h-10 bg-brand-primary/20 backdrop-blur-sm rounded-xl flex items-center justify-center border border-brand-primary/30">
                  <MessageCircle className="w-5 h-5 text-brand-light" strokeWidth={1.5} />
                </div>
              </div>
              <h1 className="text-xl font-medium text-brand-light tracking-wide">SENTER</h1>
            </div>
          </div>

          {/* Input Bar */}
          <div className="p-6">
            <div className="relative">
              <div className="absolute inset-0 bg-brand-light/[0.06] backdrop-blur-sm rounded-2xl border border-brand-light/[0.08]" />
              <div className="relative flex items-center gap-3 p-4">
                {/* Context Menu Button */}
                <IconButton
                  icon={Menu}
                  isActive={isContextMenuOpen}
                  onClick={toggleContextMenu}
                />

                {/* Input Field */}
                <textarea
                  ref={textareaRef}
                  value={inputValue}
                  onChange={handleTextareaChange}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask me anything..."
                  className="flex-1 bg-transparent text-brand-light placeholder-brand-light/40 outline-none font-normal text-sm resize-none min-h-[20px] max-h-24"
                  rows={1}
                />

                {/* Action Buttons */}
                <div className="flex items-center gap-2">
                  {/* Microphone Button */}
                  <motion.button
                    onMouseDown={() => setListening(true)}
                    onMouseUp={() => setListening(false)}
                    onMouseLeave={() => setListening(false)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                      isListening
                        ? 'bg-brand-accent/20 border border-brand-accent/30'
                        : 'bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10'
                    }`}
                  >
                    <Mic
                      className={`w-4 h-4 ${isListening ? 'text-brand-accent' : 'text-brand-light/70'}`}
                      strokeWidth={1.5}
                    />
                  </motion.button>

                  {/* Camera Button */}
                  <IconButton icon={Camera} />

                  {/* Send Button */}
                  <motion.button
                    onClick={handleSendMessage}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={!inputValue.trim()}
                    className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-accent/90 hover:bg-brand-accent disabled:opacity-50 disabled:cursor-not-allowed border border-brand-accent/20 transition-all duration-200"
                  >
                    <Send className="w-4 h-4 text-brand-light" strokeWidth={1.5} />
                  </motion.button>

                  {/* Chat History Button */}
                  <IconButton
                    icon={History}
                    isActive={isChatViewOpen}
                    onClick={toggleChatView}
                  />
                </div>
              </div>
            </div>
          </div>
        </GlassPanel>
      </div>
    </div>
  )
}
