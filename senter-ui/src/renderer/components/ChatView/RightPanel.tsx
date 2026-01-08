import { useState, useMemo, useCallback } from 'react'
import { useStore } from '@/store'
import { type ChatViewTab } from '@/types'
import { Search, X } from 'lucide-react'

const tabs: { id: ChatViewTab; label: string }[] = [
  { id: 'history', label: 'History' },
  { id: 'journal', label: 'Journal' },
  { id: 'tasks', label: 'Tasks' },
]

export function RightPanel() {
  const { activeTab, setActiveTab, conversations, loadConversation, currentConversationId, startNewConversation } = useStore()
  const [searchQuery, setSearchQuery] = useState('')

  // P1-004: Filter conversations by search term
  const filteredConversations = useMemo(() => {
    if (!searchQuery.trim()) return conversations

    const query = searchQuery.toLowerCase()
    return conversations.filter((conv) => {
      // Match title
      if (conv.title.toLowerCase().includes(query)) return true

      // Match message content
      return conv.messages.some((msg) =>
        msg.content.toLowerCase().includes(query)
      )
    })
  }, [conversations, searchQuery])

  const handleClearSearch = useCallback(() => {
    setSearchQuery('')
  }, [])

  return (
    <div className="w-64 border-l border-brand-light/10 flex flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-brand-light/10">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-3 text-xs font-medium transition-all ${
              activeTab === tab.id
                ? 'text-brand-light bg-brand-primary/20 border-b-2 border-brand-accent'
                : 'text-brand-light/60 hover:text-brand-light hover:bg-brand-light/5'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {activeTab === 'history' && (
          <div className="space-y-2">
            {/* P1-004: Search Bar */}
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-brand-light/40" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search conversations..."
                className="w-full pl-7 pr-7 py-1.5 text-xs bg-brand-light/5 border border-brand-light/10 rounded-lg text-brand-light placeholder-brand-light/40 focus:outline-none focus:border-brand-primary/30"
              />
              {searchQuery && (
                <button
                  onClick={handleClearSearch}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-brand-light/40 hover:text-brand-light"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            {/* New Conversation Button */}
            <button
              onClick={startNewConversation}
              className={`w-full text-left p-2 rounded-lg transition-colors ${
                !currentConversationId
                  ? 'bg-brand-accent/20 border border-brand-accent/30'
                  : 'bg-brand-light/5 hover:bg-brand-light/10'
              }`}
            >
              <p className={`text-xs font-medium ${
                !currentConversationId ? 'text-brand-accent' : 'text-brand-light'
              }`}>
                + New Conversation
              </p>
            </button>

            {filteredConversations.length === 0 ? (
              <div className="text-center text-brand-light/40 text-xs py-4">
                {searchQuery ? 'No matching conversations' : 'No conversation history'}
              </div>
            ) : (
              filteredConversations.map((conv) => {
                const isActive = currentConversationId === conv.id
                const dateStr = conv.updatedAt instanceof Date
                  ? conv.updatedAt.toLocaleDateString()
                  : new Date(conv.updatedAt).toLocaleDateString()

                return (
                  <button
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-brand-primary/20 border border-brand-primary/30'
                        : 'hover:bg-brand-light/5'
                    }`}
                  >
                    <p className={`text-xs font-medium truncate ${
                      isActive ? 'text-brand-accent' : 'text-brand-light'
                    }`}>
                      {conv.title}
                    </p>
                    <p className="text-xs text-brand-light/50 mt-1">
                      {dateStr}
                    </p>
                  </button>
                )
              })
            )}
          </div>
        )}

        {activeTab === 'journal' && (
          <div className="text-center text-brand-light/40 text-xs py-4">
            Journal entries coming soon
          </div>
        )}

        {activeTab === 'tasks' && (
          <div className="text-center text-brand-light/40 text-xs py-4">
            Tasks coming soon
          </div>
        )}
      </div>
    </div>
  )
}
