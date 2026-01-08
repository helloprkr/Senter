import { useStore } from '@/store'
import { type ChatViewTab } from '@/types'

const tabs: { id: ChatViewTab; label: string }[] = [
  { id: 'history', label: 'History' },
  { id: 'journal', label: 'Journal' },
  { id: 'tasks', label: 'Tasks' },
]

export function RightPanel() {
  const { activeTab, setActiveTab, conversations, loadConversation, currentConversationId, startNewConversation } = useStore()

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

            {conversations.length === 0 ? (
              <div className="text-center text-brand-light/40 text-xs py-4">
                No conversation history
              </div>
            ) : (
              conversations.map((conv) => {
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
