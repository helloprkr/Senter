import { type StateCreator } from 'zustand'
import { type Message, type ConversationThread, type ChatViewTab } from '@/types'

export interface ChatSlice {
  // State
  messages: Message[]
  inputValue: string
  isTyping: boolean
  currentConversationId: string | null
  conversations: ConversationThread[]
  activeTab: ChatViewTab

  // Actions
  addMessage: (message: Message) => void
  setInputValue: (value: string) => void
  setTyping: (typing: boolean) => void
  clearMessages: () => void
  setCurrentConversation: (id: string | null) => void
  setActiveTab: (tab: ChatViewTab) => void
  loadConversation: (id: string) => void
  setConversations: (conversations: ConversationThread[]) => void
  startNewConversation: () => void
}

export const createChatSlice: StateCreator<ChatSlice, [], [], ChatSlice> = (set, get) => ({
  // Initial states
  messages: [],
  inputValue: '',
  isTyping: false,
  currentConversationId: null,
  conversations: [],
  activeTab: 'history',

  // Actions
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  setInputValue: (value) =>
    set({ inputValue: value }),

  setTyping: (typing) =>
    set({ isTyping: typing }),

  clearMessages: () =>
    set({ messages: [], currentConversationId: null }),

  setCurrentConversation: (id) =>
    set({ currentConversationId: id }),

  setActiveTab: (tab) =>
    set({ activeTab: tab }),

  loadConversation: (id) => {
    const conversation = get().conversations.find((c) => c.id === id)
    if (conversation) {
      set({
        messages: conversation.messages,
        currentConversationId: id,
      })
    }
  },

  setConversations: (conversations) =>
    set({ conversations }),

  startNewConversation: () =>
    set({
      messages: [],
      currentConversationId: null,
    }),
})
