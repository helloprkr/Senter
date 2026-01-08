import { useCallback, useEffect, useRef } from 'react'
import { useStore } from '@/store'
import { type Message, type ResearchResult, type ConversationThread } from '@/types'

interface UseSenterReturn {
  sendQuery: (text: string) => Promise<void>
  fetchResearch: () => Promise<void>
  fetchConversations: () => Promise<void>
  saveCurrentConversation: () => Promise<void>
  checkHealth: () => Promise<void>
  connect: () => Promise<void>
  isConnected: boolean
}

export function useSenter(): UseSenterReturn {
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const {
    messages,
    currentConversationId,
    addMessage,
    setTyping,
    setResearchResults,
    setResearching,
    setConversations,
    setCurrentConversation,
  } = useStore()

  const isConnected = useRef(false)

  const sendQuery = useCallback(async (text: string): Promise<void> => {
    try {
      console.log('[useSenter] Sending query:', text)
      console.log('[useSenter] window.api exists:', !!window.api)
      console.log('[useSenter] window.api.sendQuery exists:', !!window.api?.sendQuery)

      setTyping(true)

      // Add user message immediately
      const userMessage: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: text,
        timestamp: new Date(),
      }
      addMessage(userMessage)

      // Send to backend
      console.log('[useSenter] Calling window.api.sendQuery...')
      const response = await window.api.sendQuery(text)
      console.log('[useSenter] Got response:', response)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { response?: string }; error?: string }
        console.log('[useSenter] Parsed response - success:', res.success, 'data:', res.data)
        if (res.success && res.data?.response) {
          const botMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: 'bot',
            content: res.data.response,
            timestamp: new Date(),
          }
          addMessage(botMessage)
        } else {
          // Fallback response
          const botMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: 'bot',
            content: res.error || 'I received your message but couldn\'t process it.',
            timestamp: new Date(),
          }
          addMessage(botMessage)
        }
      } else {
        console.log('[useSenter] Response format unexpected:', response)
      }
    } catch (error) {
      console.error('[useSenter] Failed to send query:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'Sorry, I couldn\'t connect to the Senter backend. Please make sure it\'s running.',
        timestamp: new Date(),
      }
      addMessage(errorMessage)
    } finally {
      setTyping(false)
    }
  }, [addMessage, setTyping])

  const fetchResearch = useCallback(async (): Promise<void> => {
    try {
      setResearching(true)
      const response = await window.api.getResearch(20)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: ResearchResult[] }
        if (res.success && Array.isArray(res.data)) {
          setResearchResults(res.data)
        }
      }
    } catch (error) {
      console.error('Failed to fetch research:', error)
    } finally {
      setResearching(false)
    }
  }, [setResearchResults, setResearching])

  const checkHealth = useCallback(async (): Promise<void> => {
    try {
      const response = await window.api.getHealth()
      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean }
        isConnected.current = res.success
      }
    } catch {
      isConnected.current = false
    }
  }, [])

  const fetchConversations = useCallback(async (): Promise<void> => {
    try {
      console.log('[useSenter] Fetching conversations...')
      const response = await window.api.getConversations(undefined, true)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as {
          success: boolean
          data?: {
            conversations?: Array<{
              id: string
              messages?: Array<{ role: string; content: string }>
              summary?: string
              created?: string
              focus?: string
            }>
          }
        }

        if (res.success && res.data?.conversations) {
          // Convert daemon format to ConversationThread format
          const threads: ConversationThread[] = res.data.conversations.map((conv) => ({
            id: conv.id,
            title: conv.summary || 'Untitled',
            messages: (conv.messages || []).map((msg, idx) => ({
              id: `${conv.id}-${idx}`,
              type: msg.role === 'user' ? 'user' : 'bot',
              content: msg.content,
              timestamp: new Date(conv.created || Date.now()),
            })) as Message[],
            createdAt: new Date(conv.created || Date.now()),
            updatedAt: new Date(conv.created || Date.now()),
          }))

          console.log('[useSenter] Loaded conversations:', threads.length)
          setConversations(threads)
        }
      }
    } catch (error) {
      console.error('[useSenter] Failed to fetch conversations:', error)
    }
  }, [setConversations])

  const saveCurrentConversation = useCallback(async (): Promise<void> => {
    if (messages.length === 0) return

    try {
      console.log('[useSenter] Saving conversation...')

      // Convert Message format to daemon format
      const daemonMessages = messages.map((msg) => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.content,
      }))

      const conversation = {
        id: currentConversationId || undefined,
        messages: daemonMessages,
        focus: 'general',
        created: messages[0]?.timestamp?.toISOString(),
      }

      const response = await window.api.saveConversation(conversation)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as {
          success: boolean
          data?: { conversation_id?: string }
        }

        if (res.success && res.data?.conversation_id) {
          console.log('[useSenter] Saved conversation:', res.data.conversation_id)
          // Update current conversation ID if it was new
          if (!currentConversationId) {
            setCurrentConversation(res.data.conversation_id)
          }
          // Refresh conversation list
          await fetchConversations()
        }
      }
    } catch (error) {
      console.error('[useSenter] Failed to save conversation:', error)
    }
  }, [messages, currentConversationId, setCurrentConversation, fetchConversations])

  const connect = useCallback(async (): Promise<void> => {
    await checkHealth()
    if (isConnected.current) {
      await fetchResearch()
      await fetchConversations()
    }
  }, [checkHealth, fetchResearch, fetchConversations])

  // Set up polling for research updates and load conversations on mount
  useEffect(() => {
    // Initial fetch
    fetchResearch()
    fetchConversations()
    checkHealth()

    // Poll every 30 seconds
    pollIntervalRef.current = setInterval(() => {
      fetchResearch()
      checkHealth()
    }, 30000)

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [fetchResearch, fetchConversations, checkHealth])

  // Auto-save conversation when messages change (after bot response)
  const lastMessageCountRef = useRef(0)
  useEffect(() => {
    // Only save when we have messages and the count increased by 2 (user + bot)
    if (messages.length > 0 && messages.length >= lastMessageCountRef.current + 2) {
      const lastMessage = messages[messages.length - 1]
      // Save after bot response
      if (lastMessage?.type === 'bot') {
        saveCurrentConversation()
      }
    }
    lastMessageCountRef.current = messages.length
  }, [messages, saveCurrentConversation])

  return {
    sendQuery,
    fetchResearch,
    fetchConversations,
    saveCurrentConversation,
    checkHealth,
    connect,
    isConnected: isConnected.current,
  }
}
