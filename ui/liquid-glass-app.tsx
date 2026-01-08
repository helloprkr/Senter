"use client"

import type React from "react"
import { useState, useEffect, useRef, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { MessageCircle, Mic, Camera, History, Menu, Send, Settings, Loader2, ImageIcon, X } from "lucide-react"
import { ContextMenu } from "./components/context-menu"
import { ChatView } from "./components/chat-view"
import { SettingsModal } from "./components/settings-modal"
import { CameraModal } from "./components/camera-modal"
import { ActivityBadge } from "./components/activity-badge"
import { AwayModal } from "./components/away-modal"
import { SuggestionsToast } from "./components/suggestions-toast"
import { useStore } from "./lib/store"
import { useAudioRecording } from "./hooks/use-audio-recording"
import { useElectronChat } from "./hooks/use-electron-chat"
import { useSenter } from "./lib/use-senter"
import senterAPI from "./lib/senter-api"

// Extend Window interface for Electron API
declare global {
  interface Window {
    electronAPI?: {
      readClipboard: () => Promise<string>;
      readClipboardFormats: () => Promise<{ formats: string[]; text: string; html: string | null }>;
      readRecentFiles: () => Promise<Array<{ name: string; path: string; extension: string; modified: Date }>>;
      readFileContent: (filePath: string) => Promise<{ success: boolean; content?: string; error?: string }>;
      getAppPaths: () => Promise<{ userData: string; documents: string; desktop: string; home: string }>;
      checkOllama: () => Promise<{ running: boolean; models: Array<{ name: string }> }>;
      windowMinimize: () => void;
      windowMaximize: () => void;
      windowClose: () => void;
      setIgnoreMouseEvents: (ignore: boolean, options?: { forward?: boolean }) => void;
      resizeWindow: (width: number, height: number) => void;
      chatStream: (options: unknown) => Promise<{ success: boolean; error?: string }>;
      chatCancel: (requestId: string) => Promise<{ success: boolean; error?: string }>;
      onChatChunk: (callback: (data: { content: string; done: boolean; fullContent?: string }) => void) => () => void;
      platform: string;
      isElectron: boolean;
    };
  }
}

export default function Component() {
  const {
    settings,
    isSettingsOpen,
    setIsSettingsOpen,
    isChatViewOpen,
    setIsChatViewOpen,
    isContextMenuOpen,
    setIsContextMenuOpen,
    ollamaConnected,
    setOllamaConnected,
    setSenterSuggestions,
  } = useStore()

  // Senter integration
  const {
    connected: senterConnected,
    goals: senterGoals,
    suggestions: senterSuggestions,
    activity: senterActivity,
    awayInfo: senterAwayInfo,
    showAwayModal,
    setShowAwayModal,
  } = useSenter()

  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [isMounted, setIsMounted] = useState(false) // For hydration fix
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // AI Chat integration - uses IPC in Electron, fetch in browser
  const { messages, input, setInput, handleSubmit, isLoading, error, append, setMessages } = useElectronChat({
    provider: settings.provider,
    model: settings.model,
    apiKey: settings.apiKey,
    ollamaEndpoint: settings.ollamaEndpoint,
    onFinish: () => {
      // Clear captured image after sending
      setCapturedImage(null)
    },
    onError: (err) => {
      console.error('Chat error:', err)
    },
  })

  // Audio recording
  const { 
    isRecording, 
    isSupported: isAudioSupported,
    transcript,
    interimTranscript,
    startRecording, 
    stopRecording,
    error: audioError 
  } = useAudioRecording({
    onTranscript: (text, isFinal) => {
      if (isFinal) {
        setInput(prev => prev + text + ' ')
      }
    },
  })

  // Check Ollama connection on mount
  useEffect(() => {
    const checkOllama = async () => {
      try {
        if (window.electronAPI) {
          const result = await window.electronAPI.checkOllama()
          setOllamaConnected(result.running)
        } else {
          const response = await fetch(`${settings.ollamaEndpoint}/api/tags`)
          setOllamaConnected(response.ok)
        }
      } catch {
        setOllamaConnected(false)
      }
    }

    checkOllama()
    const interval = setInterval(checkOllama, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [settings.ollamaEndpoint, setOllamaConnected])

  // Hydration fix - ensures client-specific state matches server
  useEffect(() => {
    setIsMounted(true)
  }, [])

  // Click-through for transparent window gaps
  // This allows clicking the desktop between UI panels
  useEffect(() => {
    if (!window.electronAPI?.setIgnoreMouseEvents) return

    const handleMouseMove = (e: MouseEvent) => {
      // Use elementFromPoint for accurate hit detection
      // This queries what's visually at the mouse position, not event bubbling
      const element = document.elementFromPoint(e.clientX, e.clientY)

      // If no element found, or we're over root/body, allow click-through
      if (!element) {
        window.electronAPI?.setIgnoreMouseEvents(true, { forward: true })
        return
      }

      const tagName = element.tagName.toUpperCase()

      // Check if we're over root elements (transparent gaps)
      const isRootElement =
        tagName === 'HTML' ||
        tagName === 'BODY' ||
        element.id === '__next' ||
        element.id === 'root'

      // Check if we're over a UI panel or any interactive element inside one
      const isOverPanel = element.closest('[data-ui-panel]') !== null

      // Check if we're over any interactive element
      const isInteractive =
        tagName === 'BUTTON' ||
        tagName === 'INPUT' ||
        tagName === 'TEXTAREA' ||
        tagName === 'SELECT' ||
        tagName === 'A' ||
        element.getAttribute('role') === 'button' ||
        element.hasAttribute('onclick')

      if ((isRootElement || element.hasAttribute('data-transparent-gap')) && !isOverPanel && !isInteractive) {
        // Allow clicks to pass through to desktop
        window.electronAPI?.setIgnoreMouseEvents(true, { forward: true })
      } else {
        // Capture clicks for the app UI
        window.electronAPI?.setIgnoreMouseEvents(false)
      }
    }

    // Add listener with passive option for performance
    document.addEventListener('mousemove', handleMouseMove, { passive: true })

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      // Reset to capture all events when component unmounts
      window.electronAPI?.setIgnoreMouseEvents(false)
    }
  }, [])

  // Handle form submission
  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() && !capturedImage) return

    const messageText = input.trim()
    setInput('')
    setIsChatViewOpen(true)

    // If Senter backend is connected, use it
    if (senterConnected) {
      // Add user message to chat view
      append({
        role: 'user' as const,
        content: capturedImage ? `[Image attached]\n\n${messageText}` : messageText,
      })
      setCapturedImage(null)

      try {
        const response = await senterAPI.chat(messageText)

        // Add bot message
        append({
          role: 'assistant' as const,
          content: response.response,
        })

        // Update suggestions if any
        if (response.suggestions?.length > 0) {
          setSenterSuggestions(response.suggestions)
        }
      } catch (err) {
        console.error('Senter chat error:', err)
        // Fall back to local AI if Senter fails
        append({
          role: 'assistant' as const,
          content: 'Senter backend unavailable. Using local AI...',
        })
      }
    } else {
      // Fall back to local Ollama/OpenAI
      if (capturedImage) {
        const messageContent = messageText
          ? `[Image attached]\n\n${messageText}`
          : '[Image attached]'

        append({
          role: 'user',
          content: messageContent,
        })
        setCapturedImage(null)
      } else {
        append({
          role: 'user',
          content: messageText,
        })
      }
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSubmit(e as unknown as React.FormEvent)
    }
  }

  // Handle microphone toggle
  const handleMicrophoneClick = () => {
    if (isRecording) {
      const finalText = stopRecording()
      if (finalText) {
        setInput(prev => prev + finalText)
      }
    } else {
      startRecording()
    }
  }

  // Handle context menu selection
  const handleContextSelect = (context: { content: string; title: string }) => {
    setInput(prev => {
      const newContent = `Context: ${context.title}\n${context.content}\n\n${prev}`
      return newContent
    })
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  // Handle camera capture
  const handleCameraCapture = (imageData: string) => {
    setCapturedImage(imageData)
    setIsCameraOpen(false)
  }

  // Convert AI SDK messages to ChatView format
  const chatViewMessages = messages.map(msg => ({
    id: msg.id,
    type: msg.role as "user" | "bot",
    content: msg.content,
    timestamp: new Date(),
  }))

  return (
    <div className="relative overflow-visible font-manrope" data-transparent-gap>
      {/* NORMAL FLOW LAYOUT: All panels in flex column - NO absolute positioning outside bounds */}
      {/* This ensures nothing extends outside body (which has overflow:hidden for transparency) */}
      <div className="relative z-10 flex flex-col items-center justify-start p-6 pt-10" data-transparent-gap>
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col max-w-lg w-full gap-2"
        >
          {/* Context Menu - ABOVE main panel in normal document flow */}
          <ContextMenu
            isOpen={isContextMenuOpen}
            onClose={() => setIsContextMenuOpen(false)}
            onSelectContext={handleContextSelect}
          />

          {/* Main Container */}
          <div className="relative">

            {/* Main Input Panel */}
            <div className="relative glass-panel overflow-hidden" data-ui-panel>
              <div className="absolute inset-0 rounded-3xl border border-brand-primary/10 pointer-events-none" />

              {/* Rainbow border animation when listening */}
              <AnimatePresence>
                {isRecording && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 rounded-3xl"
                    style={{
                      background: "linear-gradient(45deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #9400d3)",
                      backgroundSize: "400% 400%",
                      animation: "rainbow 2s linear infinite",
                      padding: "2px",
                    }}
                  >
                    <div className="absolute inset-[2px] bg-[#0a0a0a] rounded-3xl" />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-brand-light/[0.08] draggable-region relative z-10">
                <div className="flex items-center gap-3 non-draggable">
                  <div className="relative">
                    <div className="absolute inset-0 bg-brand-primary/20 rounded-xl blur-sm" />
                    <div className="relative w-10 h-10 bg-brand-primary/20 backdrop-blur-sm rounded-xl flex items-center justify-center border border-brand-primary/30">
                      <MessageCircle className="w-5 h-5 text-brand-light" strokeWidth={1.5} />
                    </div>
                  </div>
                  <div>
                    <h1 className="text-xl font-medium text-brand-light tracking-wide">SENTER</h1>
                    {/* Connection indicator */}
                    <div className="flex items-center gap-1.5">
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        senterConnected ? 'bg-green-500' : 'bg-yellow-500'
                      }`} />
                      <span className="text-[10px] text-brand-light/50">
                        {senterConnected ? 'Connected' : 'Local Only'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Activity Badge */}
                <div className="flex items-center gap-2">
                  <ActivityBadge activity={senterActivity} connected={senterConnected} />
                </div>

                {/* Settings Button */}
                <motion.button
                  onClick={() => setIsSettingsOpen(true)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10 transition-all duration-200 non-draggable"
                >
                  <Settings className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
                </motion.button>
              </div>

              {/* Captured Image Preview */}
              <AnimatePresence>
                {capturedImage && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="px-6 pt-4 relative z-10"
                  >
                    <div className="relative inline-block">
                      <img 
                        src={capturedImage} 
                        alt="Captured" 
                        className="h-20 rounded-lg border border-brand-light/20"
                      />
                      <motion.button
                        onClick={() => setCapturedImage(null)}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-brand-accent flex items-center justify-center"
                      >
                        <X className="w-3 h-3 text-brand-light" />
                      </motion.button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Interim transcript display */}
              <AnimatePresence>
                {(isRecording && interimTranscript) && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="px-6 pt-4 relative z-10"
                  >
                    <div className="px-3 py-2 rounded-lg bg-brand-accent/10 border border-brand-accent/20">
                      <p className="text-sm text-brand-light/70 italic">{interimTranscript}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Error display */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="px-6 pt-4 relative z-10"
                  >
                    <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
                      <p className="text-sm text-red-400">{error.message}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Enhanced Input Bar */}
              <form onSubmit={onSubmit} className="p-6 relative z-10">
                <div className="relative">
                  <div className="absolute inset-0 bg-brand-light/[0.06] backdrop-blur-sm rounded-2xl border border-brand-light/[0.08]" />
                  <div className="relative flex items-center gap-3 p-4">
                    {/* Context Menu Button */}
                    <motion.button
                      type="button"
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
                      ref={textareaRef}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyPress}
                      placeholder={isRecording ? "Listening..." : "Ask me anything..."}
                      className="flex-1 bg-transparent text-brand-light placeholder-brand-light/40 outline-none font-normal text-sm resize-none min-h-[20px] max-h-24"
                      rows={1}
                      disabled={isLoading}
                    />

                    {/* Action Buttons */}
                    <div className="flex items-center gap-2">
                      {/* Microphone Button */}
                      <motion.button
                        type="button"
                        onClick={handleMicrophoneClick}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={!isMounted || !isAudioSupported}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                          isRecording
                            ? "bg-brand-accent/20 border border-brand-accent/30"
                            : "bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10"
                        } disabled:opacity-50`}
                      >
                        <Mic
                          className={`w-4 h-4 ${isRecording ? "text-brand-accent animate-pulse" : "text-brand-light/70"}`}
                          strokeWidth={1.5}
                        />
                      </motion.button>

                      {/* Camera Button */}
                      <motion.button
                        type="button"
                        onClick={() => setIsCameraOpen(true)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 ${
                          capturedImage
                            ? "bg-brand-primary/20 border border-brand-primary/30"
                            : "bg-brand-light/[0.08] hover:bg-brand-primary/10 border border-brand-light/10"
                        }`}
                      >
                        {capturedImage ? (
                          <ImageIcon className="w-4 h-4 text-brand-accent" strokeWidth={1.5} />
                        ) : (
                          <Camera className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
                        )}
                      </motion.button>

                      {/* Send Button */}
                      <motion.button
                        type="submit"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        disabled={(!input.trim() && !capturedImage) || isLoading}
                        className="w-8 h-8 rounded-lg flex items-center justify-center bg-brand-accent/90 hover:bg-brand-accent disabled:opacity-50 disabled:cursor-not-allowed border border-brand-accent/20 transition-all duration-200"
                      >
                        {isLoading ? (
                          <Loader2 className="w-4 h-4 text-brand-light animate-spin" strokeWidth={1.5} />
                        ) : (
                          <Send className="w-4 h-4 text-brand-light" strokeWidth={1.5} />
                        )}
                      </motion.button>

                      {/* Chat History Button */}
                      <motion.button
                        type="button"
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
              </form>
            </div>

          </div>

          {/* Chat View - BELOW main panel in normal document flow */}
          <ChatView
            isOpen={isChatViewOpen}
            messages={chatViewMessages}
            onClose={() => setIsChatViewOpen(false)}
            goals={senterGoals}
          />
        </motion.div>
      </div>

      {/* Settings Modal */}
      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />

      {/* Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
      />

      {/* Away Modal */}
      <AwayModal
        isOpen={showAwayModal}
        onClose={() => setShowAwayModal(false)}
        awayInfo={senterAwayInfo}
      />

      {/* Suggestions Toast */}
      <SuggestionsToast
        suggestions={senterSuggestions}
        onDismiss={(index) => {
          const newSuggestions = [...senterSuggestions]
          newSuggestions.splice(index, 1)
          setSenterSuggestions(newSuggestions)
        }}
        onAccept={(suggestion) => {
          setInput(suggestion.content)
          const newSuggestions = senterSuggestions.filter(s => s !== suggestion)
          setSenterSuggestions(newSuggestions)
        }}
      />

      <style jsx>{`
        @keyframes rainbow {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .draggable-region {
          -webkit-app-region: drag;
        }
        .non-draggable {
          -webkit-app-region: no-drag;
        }
      `}</style>
    </div>
  )
}
