import { useState, useMemo, useCallback, useEffect } from 'react'
import { useStore } from '@/store'
import { type ChatViewTab } from '@/types'
import { Search, X, Clock, CheckCircle, Loader2, BookOpen, RefreshCw } from 'lucide-react'

// P2-001: Task interface
interface DaemonTask {
  id: string
  title: string
  description: string
  status: 'pending' | 'completed' | 'error'
  timestamp: number
  datetime: string
  result?: string
}

// P2-004: Journal entry interface
interface JournalEntry {
  date: string
  summary: string
  conversations_count: number
  tasks_count: number
  topics: string[]
}

const tabs: { id: ChatViewTab; label: string }[] = [
  { id: 'history', label: 'History' },
  { id: 'journal', label: 'Journal' },
  { id: 'tasks', label: 'Tasks' },
]

export function RightPanel() {
  const { activeTab, setActiveTab, conversations, loadConversation, currentConversationId, startNewConversation } = useStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [tasks, setTasks] = useState<DaemonTask[]>([])
  const [tasksLoading, setTasksLoading] = useState(false)
  const [selectedTask, setSelectedTask] = useState<DaemonTask | null>(null)

  // P2-004: Journal state
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([])
  const [journalLoading, setJournalLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)

  // P2-001: Fetch tasks when tab is active
  useEffect(() => {
    if (activeTab === 'tasks') {
      fetchTasks()
    }
  }, [activeTab])

  // P2-004: Fetch journal entries when tab is active
  useEffect(() => {
    if (activeTab === 'journal') {
      fetchJournal()
    }
  }, [activeTab])

  const fetchTasks = async () => {
    try {
      setTasksLoading(true)
      const response = await window.api.getTasks(20)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { tasks?: DaemonTask[] } }
        if (res.success && res.data?.tasks) {
          setTasks(res.data.tasks)
        }
      }
    } catch (error) {
      console.error('[RightPanel] Failed to fetch tasks:', error)
    } finally {
      setTasksLoading(false)
    }
  }

  // P2-004: Fetch journal entries
  const fetchJournal = async () => {
    try {
      setJournalLoading(true)
      const response = await window.api.getJournal(undefined, 7)

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { entries?: JournalEntry[] } }
        if (res.success && res.data?.entries) {
          setJournalEntries(res.data.entries)
        }
      }
    } catch (error) {
      console.error('[RightPanel] Failed to fetch journal:', error)
    } finally {
      setJournalLoading(false)
    }
  }

  // P2-004: Generate today's journal entry
  const generateTodayEntry = async () => {
    try {
      setIsGenerating(true)
      const today = new Date().toISOString().split('T')[0]
      const response = await window.api.generateJournal(today)

      if (response && typeof response === 'object' && 'success' in response) {
        // Refresh journal entries after generation
        await fetchJournal()
      }
    } catch (error) {
      console.error('[RightPanel] Failed to generate journal:', error)
    } finally {
      setIsGenerating(false)
    }
  }

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
          <div className="space-y-2">
            {/* Generate Today's Entry Button */}
            <button
              onClick={generateTodayEntry}
              disabled={isGenerating}
              className="w-full flex items-center justify-center gap-2 p-2 rounded-lg bg-brand-accent/20 hover:bg-brand-accent/30 border border-brand-accent/30 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3 h-3 text-brand-accent ${isGenerating ? 'animate-spin' : ''}`} />
              <span className="text-xs font-medium text-brand-accent">
                {isGenerating ? 'Generating...' : "Generate Today's Entry"}
              </span>
            </button>

            {journalLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="w-4 h-4 text-brand-light/40 animate-spin" />
              </div>
            ) : journalEntries.length === 0 ? (
              <div className="text-center text-brand-light/40 text-xs py-4">
                <BookOpen className="w-6 h-6 mx-auto mb-2 opacity-50" />
                No journal entries yet.<br />
                Generate one for today!
              </div>
            ) : (
              journalEntries.map((entry) => (
                <div
                  key={entry.date}
                  className="p-3 rounded-lg bg-brand-light/5 border border-brand-light/10 hover:bg-brand-light/10 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <BookOpen className="w-3 h-3 text-brand-accent" />
                    <span className="text-xs font-medium text-brand-light">
                      {new Date(entry.date).toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric'
                      })}
                    </span>
                  </div>
                  <p className="text-xs text-brand-light/80 mb-2 line-clamp-3">
                    {entry.summary}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {entry.topics.slice(0, 3).map((topic, i) => (
                      <span
                        key={i}
                        className="text-xs px-1.5 py-0.5 rounded bg-brand-primary/20 text-brand-light/70"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-3 mt-2 text-xs text-brand-light/50">
                    <span>{entry.conversations_count} chats</span>
                    <span>{entry.tasks_count} tasks</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'tasks' && (
          <div className="space-y-2">
            {tasksLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="w-4 h-4 text-brand-light/40 animate-spin" />
              </div>
            ) : tasks.length === 0 ? (
              <div className="text-center text-brand-light/40 text-xs py-4">
                No tasks yet
              </div>
            ) : (
              <>
                {/* Selected task detail view */}
                {selectedTask && (
                  <div className="bg-brand-light/5 border border-brand-light/10 rounded-lg p-3 mb-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        selectedTask.status === 'completed'
                          ? 'bg-green-500/20 text-green-400'
                          : selectedTask.status === 'pending'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {selectedTask.status}
                      </span>
                      <button
                        onClick={() => setSelectedTask(null)}
                        className="text-brand-light/40 hover:text-brand-light"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                    <p className="text-xs font-medium text-brand-light mb-1">
                      {selectedTask.title}
                    </p>
                    <p className="text-xs text-brand-light/60 mb-2">
                      {selectedTask.description}
                    </p>
                    {selectedTask.result && (
                      <div className="text-xs text-brand-light/70 bg-brand-dark/30 rounded p-2 max-h-32 overflow-y-auto">
                        {selectedTask.result}
                      </div>
                    )}
                  </div>
                )}

                {/* Task list */}
                {tasks.map((task) => (
                  <button
                    key={task.id}
                    onClick={() => setSelectedTask(task)}
                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                      selectedTask?.id === task.id
                        ? 'bg-brand-primary/20 border border-brand-primary/30'
                        : 'hover:bg-brand-light/5'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {task.status === 'completed' ? (
                        <CheckCircle className="w-3 h-3 text-green-400 mt-0.5 flex-shrink-0" />
                      ) : task.status === 'pending' ? (
                        <Clock className="w-3 h-3 text-yellow-400 mt-0.5 flex-shrink-0" />
                      ) : (
                        <X className="w-3 h-3 text-red-400 mt-0.5 flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium text-brand-light truncate">
                          {task.title}
                        </p>
                        <p className="text-xs text-brand-light/50">
                          {new Date(task.datetime).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
