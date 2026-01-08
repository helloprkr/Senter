"use client"

import { useState, useCallback, useRef, useEffect } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  createdAt?: Date
}

interface UseElectronChatOptions {
  provider: 'ollama' | 'openai' | 'anthropic' | 'google'
  model: string
  apiKey?: string
  ollamaEndpoint?: string
  onFinish?: () => void
  onError?: (error: Error) => void
}

interface UseElectronChatReturn {
  messages: Message[]
  input: string
  setInput: (input: string) => void
  handleSubmit: (e: React.FormEvent) => void
  append: (message: Omit<Message, 'id' | 'createdAt'>) => Promise<void>
  isLoading: boolean
  error: Error | null
  setMessages: (messages: Message[]) => void
  stop: () => void
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

export function useElectronChat(options: UseElectronChatOptions): UseElectronChatReturn {
  const { provider, model, apiKey, ollamaEndpoint, onFinish, onError } = options
  
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  
  const currentRequestId = useRef<string | null>(null)
  const unsubscribeRef = useRef<(() => void) | null>(null)
  const assistantMessageRef = useRef<string>('')

  // Clean up listener on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
      }
    }
  }, [])

  const isElectron = typeof window !== 'undefined' && window.electronAPI?.isElectron

  const streamViaElectron = useCallback(async (messagesToSend: Message[]) => {
    if (!window.electronAPI) {
      throw new Error('Electron API not available')
    }

    const requestId = generateId()
    currentRequestId.current = requestId
    assistantMessageRef.current = ''

    // Create assistant message placeholder
    const assistantId = generateId()
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      createdAt: new Date()
    }])

    // Set up chunk listener
    unsubscribeRef.current = window.electronAPI.onChatChunk((chunk) => {
      if (chunk.content) {
        assistantMessageRef.current += chunk.content
        setMessages(prev => prev.map(msg => 
          msg.id === assistantId 
            ? { ...msg, content: assistantMessageRef.current }
            : msg
        ))
      }
      
      if (chunk.done) {
        setIsLoading(false)
        currentRequestId.current = null
        if (unsubscribeRef.current) {
          unsubscribeRef.current()
          unsubscribeRef.current = null
        }
        onFinish?.()
      }
    })

    // Start the stream
    const result = await window.electronAPI.chatStream({
      messages: messagesToSend.map(m => ({ role: m.role, content: m.content })),
      provider,
      model,
      apiKey,
      ollamaEndpoint,
      requestId
    })

    if (!result.success) {
      throw new Error(result.error || 'Chat stream failed')
    }
  }, [provider, model, apiKey, ollamaEndpoint, onFinish])

  const streamViaFetch = useCallback(async (messagesToSend: Message[]) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messagesToSend.map(m => ({ role: m.role, content: m.content })),
        provider,
        model,
        apiKey,
        ollamaEndpoint
      })
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.error || `HTTP ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body')
    }

    const decoder = new TextDecoder()
    const assistantId = generateId()
    let fullContent = ''

    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      createdAt: new Date()
    }])

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      // Parse the AI SDK stream format
      const lines = chunk.split('\n').filter(line => line.trim())
      
      for (const line of lines) {
        // AI SDK format: 0:"text chunk"
        if (line.startsWith('0:"')) {
          try {
            // Extract the text between 0:" and "
            const text = line.slice(3, -1)
              .replace(/\\n/g, '\n')
              .replace(/\\"/g, '"')
              .replace(/\\\\/g, '\\')
            fullContent += text
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, content: fullContent }
                : msg
            ))
          } catch (e) {
            // Skip parsing errors
          }
        }
      }
    }

    setIsLoading(false)
    onFinish?.()
  }, [provider, model, apiKey, ollamaEndpoint, onFinish])

  const append = useCallback(async (message: Omit<Message, 'id' | 'createdAt'>) => {
    setError(null)
    setIsLoading(true)

    const newMessage: Message = {
      ...message,
      id: generateId(),
      createdAt: new Date()
    }

    const newMessages = [...messages, newMessage]
    setMessages(newMessages)

    try {
      if (isElectron) {
        await streamViaElectron(newMessages)
      } else {
        await streamViaFetch(newMessages)
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      setIsLoading(false)
      onError?.(error)
    }
  }, [messages, isElectron, streamViaElectron, streamViaFetch, onError])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    
    const trimmedInput = input.trim()
    setInput('')
    append({ role: 'user', content: trimmedInput })
  }, [input, isLoading, append])

  const stop = useCallback(() => {
    if (currentRequestId.current && window.electronAPI) {
      window.electronAPI.chatCancel(currentRequestId.current)
    }
    setIsLoading(false)
    currentRequestId.current = null
    if (unsubscribeRef.current) {
      unsubscribeRef.current()
      unsubscribeRef.current = null
    }
  }, [])

  return {
    messages,
    input,
    setInput,
    handleSubmit,
    append,
    isLoading,
    error,
    setMessages,
    stop
  }
}
