"use client"

import type React from "react"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { MessageCircle, Mic, Camera, History, Menu, Send } from "lucide-react"
import { ContextMenu } from "./components/context-menu"
import { ChatView } from "./components/chat-view"

interface Message {
  id: string
  type: "user" | "bot"
  content: string
  timestamp: Date
}

export default function Component() {
  const [isContextMenuOpen, setIsContextMenuOpen] = useState(false)
  const [isChatViewOpen, setIsChatViewOpen] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [isListening, setIsListening] = useState(false)

  const handleSendMessage = () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    }

    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: "bot",
      content: "I'm processing your request. This is a demo response.",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage, botMessage])
    setInputValue("")
    setIsChatViewOpen(true)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="min-h-screen bg-brand-dark relative overflow-hidden font-manrope">
      {/* Background Effects */}
      <div className="fixed inset-0 opacity-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_25%,#123524,transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_75%_75%,#D64933,transparent_70%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_48%,rgba(18,53,36,0.1)_50%,transparent_52%)]" />
      </div>

      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-brand-accent/3 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 flex items-center justify-center min-h-screen p-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="relative max-w-lg w-full"
        >
          {/* Outer Glow */}
          <div className="absolute -inset-1 bg-gradient-to-r from-brand-primary/20 via-brand-light/10 to-brand-accent/20 rounded-3xl blur-xl opacity-60" />
          <div className="absolute -inset-2 bg-brand-primary/5 rounded-3xl blur-2xl" />

          {/* Main Container */}
          <div className="relative">
            {/* Context Menu */}
            <ContextMenu isOpen={isContextMenuOpen} onClose={() => setIsContextMenuOpen(false)} />

            {/* Main Input Panel */}
            <div className="relative bg-brand-light/[0.08] backdrop-blur-3xl border border-brand-light/15 rounded-3xl shadow-2xl overflow-hidden">
              <div className="absolute inset-0 rounded-3xl border border-brand-primary/10 pointer-events-none" />

              {/* Rainbow border animation when listening */}
              <AnimatePresence>
                {isListening && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 rounded-3xl border-2 border-transparent bg-gradient-to-r from-red-500 via-yellow-500 via-green-500 via-blue-500 via-indigo-500 to-purple-500 animate-pulse"
                    style={{
                      background:
                        "linear-gradient(45deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #9400d3)",
                      backgroundSize: "400% 400%",
                      animation: "rainbow 2s linear infinite",
                    }}
                  />
                )}
              </AnimatePresence>

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

              {/* Enhanced Input Bar */}
              <div className="p-6">
                <div className="relative">
                  <div className="absolute inset-0 bg-brand-light/[0.06] backdrop-blur-sm rounded-2xl border border-brand-light/[0.08]" />
                  <div className="relative flex items-center gap-3 p-4">
                    {/* Context Menu Button */}
                    <motion.button
                      onClick={() => setIsContextMenuOpen(!isContextMenuOpen)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                        isContextMenuOpen
                          ? "bg-brand-primary/20 border border-brand-primary/30"
                          : "bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10"
                      }`}
                    >
                      <Menu
                        className={`w-4 h-4 ${isContextMenuOpen ? "text-brand-accent" : "text-brand-light/70"}`}
                        strokeWidth={1.5}
                      />
                    </motion.button>

                    {/* Input Field */}
                    <textarea
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask me anything..."
                      className="flex-1 bg-transparent text-brand-light placeholder-brand-light/40 outline-none font-normal text-sm resize-none min-h-[20px] max-h-24"
                      rows={1}
                    />

                    {/* Action Buttons */}
                    <div className="flex items-center gap-2">
                      {/* Microphone Button */}
                      <motion.button
                        onMouseDown={() => setIsListening(true)}
                        onMouseUp={() => setIsListening(false)}
                        onMouseLeave={() => setIsListening(false)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                          isListening
                            ? "bg-brand-accent/20 border border-brand-accent/30"
                            : "bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10"
                        }`}
                      >
                        <Mic
                          className={`w-4 h-4 ${isListening ? "text-brand-accent" : "text-brand-light/70"}`}
                          strokeWidth={1.5}
                        />
                      </motion.button>

                      {/* Camera Button */}
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95, backgroundColor: "rgba(214, 73, 51, 0.2)" }}
                        className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10 transition-all duration-200"
                      >
                        <Camera className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
                      </motion.button>

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
                      <motion.button
                        onClick={() => setIsChatViewOpen(!isChatViewOpen)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                          isChatViewOpen
                            ? "bg-brand-primary/20 border border-brand-primary/30"
                            : "bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10"
                        }`}
                      >
                        <History
                          className={`w-4 h-4 ${isChatViewOpen ? "text-brand-accent" : "text-brand-light/70"}`}
                          strokeWidth={1.5}
                        />
                      </motion.button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Chat View */}
            <ChatView isOpen={isChatViewOpen} messages={messages} onClose={() => setIsChatViewOpen(false)} />
          </div>
        </motion.div>
      </div>

      <style jsx>{`
        @keyframes rainbow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </div>
  )
}
