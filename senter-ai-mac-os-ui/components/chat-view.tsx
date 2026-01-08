"use client"

import { motion, AnimatePresence } from "framer-motion"
import { MessageCircle, BookOpen, CheckSquare } from "lucide-react"
import { useState } from "react"

interface Message {
  id: string
  type: "user" | "bot"
  content: string
  timestamp: Date
}

interface ChatViewProps {
  isOpen: boolean
  messages: Message[]
  onClose: () => void
}

export function ChatView({ isOpen, messages, onClose }: ChatViewProps) {
  const [activeTab, setActiveTab] = useState("chat")

  const tabs = [
    { id: "chat", icon: MessageCircle, label: "Chat History" },
    { id: "journal", icon: BookOpen, label: "Journal" },
    { id: "tasks", icon: CheckSquare, label: "Task List" },
  ]

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20, height: 0 }}
          animate={{ opacity: 1, y: 0, height: "auto" }}
          exit={{ opacity: 0, y: -20, height: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="absolute top-full left-0 right-0 mt-2 overflow-hidden"
        >
          <div className="relative bg-brand-light/[0.08] backdrop-blur-3xl border border-brand-light/15 rounded-2xl shadow-2xl">
            <div className="absolute inset-0 rounded-2xl border border-brand-primary/10 pointer-events-none" />

            <div className="flex">
              {/* Content Area */}
              <div className="flex-1 p-4">
                {activeTab === "chat" && (
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {messages.length === 0 ? (
                      <div className="text-center text-brand-light/60 py-8">
                        <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No messages yet. Start a conversation!</p>
                      </div>
                    ) : (
                      messages.map((message) => (
                        <motion.div
                          key={message.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`p-3 rounded-lg ${
                            message.type === "user"
                              ? "bg-brand-primary/10 border border-brand-primary/20 ml-8"
                              : "bg-brand-light/[0.04] border border-brand-light/[0.06] mr-8"
                          }`}
                        >
                          <div className="text-brand-light text-sm font-medium mb-1">
                            {message.type === "user" ? "You" : "Assistant"}
                          </div>
                          <div className="text-brand-light/80 text-sm">{message.content}</div>
                          <div className="text-brand-light/40 text-xs mt-2">
                            {message.timestamp.toLocaleTimeString()}
                          </div>
                        </motion.div>
                      ))
                    )}
                  </div>
                )}

                {activeTab === "journal" && (
                  <div className="text-center text-brand-light/60 py-8">
                    <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Journal feature coming soon</p>
                  </div>
                )}

                {activeTab === "tasks" && (
                  <div className="text-center text-brand-light/60 py-8">
                    <CheckSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Task list feature coming soon</p>
                  </div>
                )}
              </div>

              {/* Sidebar */}
              <div className="w-16 bg-brand-primary/10 backdrop-blur-sm border-l border-brand-light/10 rounded-r-2xl p-2">
                <div className="space-y-2">
                  {tabs.map((tab) => (
                    <motion.button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 ${
                        activeTab === tab.id
                          ? "bg-brand-primary/20 border border-brand-primary/30"
                          : "bg-brand-light/[0.06] hover:bg-brand-primary/10 border border-brand-light/10"
                      }`}
                    >
                      <tab.icon
                        className={`w-5 h-5 ${activeTab === tab.id ? "text-brand-accent" : "text-brand-light/70"}`}
                        strokeWidth={1.5}
                      />
                    </motion.button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
