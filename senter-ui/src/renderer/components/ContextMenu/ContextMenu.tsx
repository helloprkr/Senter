import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '@/store'
import { GlassPanel } from '@/components/ui'
import { FilterSidebar } from './FilterSidebar'
import { ContextItemCard } from './ContextItemCard'
import { Search, Plus, FileText, Globe, Loader2 } from 'lucide-react'
import { type ContextItem, type ContextItemType } from '@/types'

// P3-001: Response types
interface DaemonSource {
  id: string
  type: string
  title: string
  path: string
  description: string
  content_preview: string
}

export function ContextMenu() {
  const {
    isContextMenuOpen,
    contextItems,
    activeFilter,
    contextSearchQuery,
    setContextSearchQuery,
    setContextItems,
    contextLoading,
    setContextLoading,
    addContextItem,
    removeContextItem,
  } = useStore()

  const [showAddDialog, setShowAddDialog] = useState(false)
  const [addType, setAddType] = useState<'file' | 'web'>('file')
  const [newPath, setNewPath] = useState('')
  const [newTitle, setNewTitle] = useState('')
  const [isAdding, setIsAdding] = useState(false)

  // P3-001: Load context sources from daemon
  const loadSources = useCallback(async () => {
    try {
      setContextLoading(true)
      const response = await window.api.getContextSources()

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { sources?: DaemonSource[] } }
        if (res.success && res.data?.sources) {
          // Convert daemon sources to ContextItems
          const items: ContextItem[] = res.data.sources.map((s) => ({
            id: s.id,
            type: s.type as ContextItemType,
            title: s.title,
            description: s.description || s.path,
            content: s.content_preview,
            createdAt: new Date(),
          }))
          setContextItems(items)
        }
      }
    } catch (error) {
      console.error('[ContextMenu] Failed to load sources:', error)
    } finally {
      setContextLoading(false)
    }
  }, [setContextItems, setContextLoading])

  // Load sources when menu opens
  useEffect(() => {
    if (isContextMenuOpen) {
      loadSources()
    }
  }, [isContextMenuOpen, loadSources])

  // P3-001: Add new context source
  const handleAddSource = async () => {
    if (!newPath.trim()) return

    try {
      setIsAdding(true)
      const response = await window.api.addContextSource({
        type: addType,
        title: newTitle.trim() || (addType === 'file' ? newPath.split('/').pop() || 'Untitled' : 'Web Source'),
        path: newPath.trim(),
        description: '',
      })

      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { source?: DaemonSource; added?: boolean } }
        if (res.success && res.data?.source) {
          // Add to local state
          const newItem: ContextItem = {
            id: res.data.source.id,
            type: res.data.source.type as ContextItemType,
            title: res.data.source.title,
            description: res.data.source.description || res.data.source.path,
            content: res.data.source.content_preview,
            createdAt: new Date(),
          }
          addContextItem(newItem)
          setShowAddDialog(false)
          setNewPath('')
          setNewTitle('')
        }
      }
    } catch (error) {
      console.error('[ContextMenu] Failed to add source:', error)
    } finally {
      setIsAdding(false)
    }
  }

  // P3-001: Remove context source
  const handleRemoveSource = async (id: string) => {
    try {
      const response = await window.api.removeContextSource(id)
      if (response && typeof response === 'object' && 'success' in response) {
        const res = response as { success: boolean; data?: { removed?: boolean } }
        if (res.success && res.data?.removed) {
          removeContextItem(id)
        }
      }
    } catch (error) {
      console.error('[ContextMenu] Failed to remove source:', error)
    }
  }

  // Filter items based on active filter and search
  const filteredItems = contextItems.filter((item) => {
    const matchesFilter = activeFilter === 'all' || item.type === activeFilter
    const matchesSearch =
      !contextSearchQuery ||
      item.title.toLowerCase().includes(contextSearchQuery.toLowerCase()) ||
      item.description?.toLowerCase().includes(contextSearchQuery.toLowerCase())
    return matchesFilter && matchesSearch
  })

  return (
    <AnimatePresence>
      {isContextMenuOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0, y: 20 }}
          animate={{ opacity: 1, height: 400, y: 0 }}
          exit={{ opacity: 0, height: 0, y: 20 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="absolute bottom-full left-0 right-0 mb-3 overflow-hidden z-10"
        >
          <GlassPanel withGlow className="h-full">
            <div className="flex h-full">
              {/* Filter Sidebar */}
              <FilterSidebar />

              {/* Context Items */}
              <div className="flex-1 flex flex-col">
                {/* Search Bar */}
                <div className="p-4 border-b border-brand-light/10">
                  <div className="relative">
                    <Search
                      className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-brand-light/60"
                      strokeWidth={1.5}
                    />
                    <input
                      type="text"
                      placeholder="Search context items..."
                      value={contextSearchQuery}
                      onChange={(e) => setContextSearchQuery(e.target.value)}
                      className="w-full bg-brand-light/5 rounded-lg py-2 pl-9 pr-4 text-sm text-brand-light placeholder-brand-light/40 outline-none"
                    />
                  </div>
                </div>

                {/* Items List */}
                <div className="flex-1 overflow-y-auto p-4">
                  {contextLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-5 h-5 text-brand-light/40 animate-spin" />
                    </div>
                  ) : filteredItems.length === 0 ? (
                    <div className="text-center text-brand-light/40 py-8">
                      {contextItems.length === 0
                        ? 'No context items yet. Add files or URLs as context.'
                        : 'No items match your search'}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {filteredItems.map((item) => (
                        <ContextItemCard
                          key={item.id}
                          item={item}
                          onRemove={() => handleRemoveSource(item.id)}
                        />
                      ))}
                    </div>
                  )}
                </div>

                {/* P3-001: Add Source Button */}
                <div className="p-4 border-t border-brand-light/10">
                  {!showAddDialog ? (
                    <button
                      onClick={() => setShowAddDialog(true)}
                      className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-brand-accent/20 hover:bg-brand-accent/30 border border-brand-accent/30 transition-colors"
                    >
                      <Plus className="w-4 h-4 text-brand-accent" />
                      <span className="text-sm font-medium text-brand-accent">Add Context Source</span>
                    </button>
                  ) : (
                    <div className="space-y-3">
                      {/* Type Selection */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => setAddType('file')}
                          className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-colors ${
                            addType === 'file'
                              ? 'bg-brand-primary/30 border border-brand-primary/50'
                              : 'bg-brand-light/5 border border-brand-light/10 hover:bg-brand-light/10'
                          }`}
                        >
                          <FileText className="w-4 h-4 text-brand-light" />
                          <span className="text-xs text-brand-light">File</span>
                        </button>
                        <button
                          onClick={() => setAddType('web')}
                          className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-colors ${
                            addType === 'web'
                              ? 'bg-brand-primary/30 border border-brand-primary/50'
                              : 'bg-brand-light/5 border border-brand-light/10 hover:bg-brand-light/10'
                          }`}
                        >
                          <Globe className="w-4 h-4 text-brand-light" />
                          <span className="text-xs text-brand-light">URL</span>
                        </button>
                      </div>

                      {/* Path Input */}
                      <input
                        type="text"
                        placeholder={addType === 'file' ? '/path/to/file.md' : 'https://example.com/docs'}
                        value={newPath}
                        onChange={(e) => setNewPath(e.target.value)}
                        className="w-full bg-brand-light/5 rounded-lg py-2 px-3 text-sm text-brand-light placeholder-brand-light/40 outline-none border border-brand-light/10 focus:border-brand-primary/30"
                      />

                      {/* Title Input (Optional) */}
                      <input
                        type="text"
                        placeholder="Title (optional)"
                        value={newTitle}
                        onChange={(e) => setNewTitle(e.target.value)}
                        className="w-full bg-brand-light/5 rounded-lg py-2 px-3 text-sm text-brand-light placeholder-brand-light/40 outline-none border border-brand-light/10 focus:border-brand-primary/30"
                      />

                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            setShowAddDialog(false)
                            setNewPath('')
                            setNewTitle('')
                          }}
                          className="flex-1 py-2 rounded-lg bg-brand-light/5 hover:bg-brand-light/10 border border-brand-light/10 transition-colors"
                        >
                          <span className="text-sm text-brand-light/70">Cancel</span>
                        </button>
                        <button
                          onClick={handleAddSource}
                          disabled={!newPath.trim() || isAdding}
                          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-brand-accent/80 hover:bg-brand-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                          {isAdding ? (
                            <Loader2 className="w-4 h-4 text-brand-light animate-spin" />
                          ) : (
                            <Plus className="w-4 h-4 text-brand-light" />
                          )}
                          <span className="text-sm font-medium text-brand-light">Add</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
