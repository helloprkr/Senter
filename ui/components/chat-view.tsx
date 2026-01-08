"use client"

import { motion, AnimatePresence } from "framer-motion"
import { MessageCircle, BookOpen, CheckSquare, Plus, Trash2, Check, Target } from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { useStore, createJournalEntry, createTask } from "@/lib/store"
import { GoalsPanel } from "./goals-panel"
import { SenterGoal } from "@/lib/senter-api"

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
  goals?: SenterGoal[]
}

export function ChatView({ isOpen, messages, onClose, goals = [] }: ChatViewProps) {
  const [activeTab, setActiveTab] = useState("chat")
  const [newJournalEntry, setNewJournalEntry] = useState("")
  const [newTaskTitle, setNewTaskTitle] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const { 
    journalEntries, 
    addJournalEntry, 
    deleteJournalEntry,
    tasks,
    addTask,
    toggleTask,
    deleteTask
  } = useStore()

  const tabs = [
    { id: "chat", icon: MessageCircle, label: "Chat History" },
    { id: "goals", icon: Target, label: "Goals" },
    { id: "journal", icon: BookOpen, label: "Journal" },
    { id: "tasks", icon: CheckSquare, label: "Task List" },
  ]

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, activeTab])

  const handleAddJournalEntry = () => {
    if (!newJournalEntry.trim()) return
    addJournalEntry(createJournalEntry(newJournalEntry.trim()))
    setNewJournalEntry("")
  }

  const handleAddTask = () => {
    if (!newTaskTitle.trim()) return
    addTask(createTask(newTaskTitle.trim()))
    setNewTaskTitle("")
  }

  const handleKeyPress = (e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      action()
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95, height: 0 }}
          animate={{ opacity: 1, scale: 1, height: "auto" }}
          exit={{ opacity: 0, scale: 0.95, height: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="w-full overflow-hidden"
        >
          <div className="relative glass-panel-light" data-ui-panel>
            <div className="absolute inset-0 rounded-2xl border border-brand-primary/10 pointer-events-none" />

            <div className="flex">
              {/* Content Area */}
              <div className="flex-1 p-4">
                {/* Chat Tab */}
                {activeTab === "chat" && (
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {messages.length === 0 ? (
                      <div className="text-center text-brand-light/60 py-8">
                        <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No messages yet. Start a conversation!</p>
                      </div>
                    ) : (
                      <>
                        {messages.map((message) => (
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
                              {message.type === "user" ? "You" : "Senter"}
                            </div>
                            <div className="text-brand-light/80 text-sm whitespace-pre-wrap">
                              {message.content}
                            </div>
                            <div className="text-brand-light/40 text-xs mt-2">
                              {message.timestamp.toLocaleTimeString()}
                            </div>
                          </motion.div>
                        ))}
                        <div ref={messagesEndRef} />
                      </>
                    )}
                  </div>
                )}

                {/* Goals Tab */}
                {activeTab === "goals" && (
                  <GoalsPanel goals={goals} />
                )}

                {/* Journal Tab */}
                {activeTab === "journal" && (
                  <div className="space-y-3">
                    {/* New Entry Input */}
                    <div className="flex gap-2">
                      <textarea
                        value={newJournalEntry}
                        onChange={(e) => setNewJournalEntry(e.target.value)}
                        onKeyPress={(e) => handleKeyPress(e, handleAddJournalEntry)}
                        placeholder="Write a journal entry..."
                        className="flex-1 px-3 py-2 rounded-lg bg-brand-light/5 border border-brand-light/10 text-brand-light placeholder-brand-light/40 outline-none focus:border-brand-primary/50 text-sm resize-none"
                        rows={2}
                      />
                      <motion.button
                        onClick={handleAddJournalEntry}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={!newJournalEntry.trim()}
                        className="px-3 rounded-lg bg-brand-accent text-brand-light disabled:opacity-50"
                      >
                        <Plus className="w-4 h-4" />
                      </motion.button>
                    </div>

                    {/* Entries List */}
                    <div className="max-h-48 overflow-y-auto space-y-2">
                      {journalEntries.length === 0 ? (
                        <div className="text-center text-brand-light/60 py-4">
                          <BookOpen className="w-6 h-6 mx-auto mb-2 opacity-50" />
                          <p className="text-xs">No journal entries yet</p>
                        </div>
                      ) : (
                        journalEntries.map((entry) => (
                          <motion.div
                            key={entry.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="p-3 rounded-lg bg-brand-light/[0.04] border border-brand-light/[0.06] group"
                          >
                            <div className="flex justify-between items-start">
                              <p className="text-brand-light/80 text-sm flex-1 whitespace-pre-wrap">
                                {entry.content}
                              </p>
                              <motion.button
                                onClick={() => deleteJournalEntry(entry.id)}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                                className="opacity-0 group-hover:opacity-100 p-1 text-brand-light/40 hover:text-red-400 transition-all"
                              >
                                <Trash2 className="w-3 h-3" />
                              </motion.button>
                            </div>
                            <div className="text-brand-light/40 text-xs mt-2">
                              {new Date(entry.createdAt).toLocaleDateString()}
                            </div>
                          </motion.div>
                        ))
                      )}
                    </div>
                  </div>
                )}

                {/* Tasks Tab */}
                {activeTab === "tasks" && (
                  <div className="space-y-3">
                    {/* New Task Input */}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={newTaskTitle}
                        onChange={(e) => setNewTaskTitle(e.target.value)}
                        onKeyPress={(e) => handleKeyPress(e, handleAddTask)}
                        placeholder="Add a new task..."
                        className="flex-1 px-3 py-2 rounded-lg bg-brand-light/5 border border-brand-light/10 text-brand-light placeholder-brand-light/40 outline-none focus:border-brand-primary/50 text-sm"
                      />
                      <motion.button
                        onClick={handleAddTask}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={!newTaskTitle.trim()}
                        className="px-3 rounded-lg bg-brand-accent text-brand-light disabled:opacity-50"
                      >
                        <Plus className="w-4 h-4" />
                      </motion.button>
                    </div>

                    {/* Tasks List */}
                    <div className="max-h-48 overflow-y-auto space-y-2">
                      {tasks.length === 0 ? (
                        <div className="text-center text-brand-light/60 py-4">
                          <CheckSquare className="w-6 h-6 mx-auto mb-2 opacity-50" />
                          <p className="text-xs">No tasks yet</p>
                        </div>
                      ) : (
                        tasks.map((task) => (
                          <motion.div
                            key={task.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex items-center gap-3 p-3 rounded-lg bg-brand-light/[0.04] border border-brand-light/[0.06] group"
                          >
                            <motion.button
                              onClick={() => toggleTask(task.id)}
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                                task.completed
                                  ? 'bg-brand-accent border-brand-accent'
                                  : 'border-brand-light/30 hover:border-brand-accent/50'
                              }`}
                            >
                              {task.completed && <Check className="w-3 h-3 text-brand-light" />}
                            </motion.button>
                            <span className={`flex-1 text-sm ${
                              task.completed 
                                ? 'text-brand-light/40 line-through' 
                                : 'text-brand-light/80'
                            }`}>
                              {task.title}
                            </span>
                            <motion.button
                              onClick={() => deleteTask(task.id)}
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              className="opacity-0 group-hover:opacity-100 p-1 text-brand-light/40 hover:text-red-400 transition-all"
                            >
                              <Trash2 className="w-3 h-3" />
                            </motion.button>
                          </motion.div>
                        ))
                      )}
                    </div>
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
