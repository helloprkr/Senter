"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Search, Globe, FileText, Clipboard, Grid3X3, RefreshCw, File, Image, Code } from "lucide-react"
import { useState, useEffect, useCallback } from "react"

interface ContextMenuProps {
  isOpen: boolean
  onClose: () => void
  onSelectContext?: (context: ContextItem) => void
}

interface ContextItem {
  id: string
  type: 'clipboard' | 'file' | 'web'
  title: string
  content: string
  path?: string
  extension?: string
  modified?: Date
}

const getFileIcon = (extension: string) => {
  const imageExts = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'];
  const codeExts = ['.js', '.ts', '.tsx', '.jsx', '.py', '.rb', '.go', '.rs'];
  
  if (imageExts.includes(extension)) return Image;
  if (codeExts.includes(extension)) return Code;
  return File;
}

export function ContextMenu({ isOpen, onClose, onSelectContext }: ContextMenuProps) {
  const [activeFilter, setActiveFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [contextItems, setContextItems] = useState<{
    clipboard: ContextItem[]
    files: ContextItem[]
    web: ContextItem[]
    all: ContextItem[]
  }>({
    clipboard: [],
    files: [],
    web: [],
    all: []
  })

  const filters = [
    { id: "clipboard", icon: Clipboard, label: "Clipboard" },
    { id: "files", icon: FileText, label: "Files" },
    { id: "web", icon: Globe, label: "Web" },
    { id: "all", icon: Grid3X3, label: "All" },
  ]

  const loadContextData = useCallback(async () => {
    setIsLoading(true)
    
    const clipboardItems: ContextItem[] = []
    const fileItems: ContextItem[] = []
    const webItems: ContextItem[] = []
    
    try {
      // Load clipboard content
      if (window.electronAPI) {
        // Running in Electron - use IPC
        const clipboardText = await window.electronAPI.readClipboard()
        if (clipboardText && clipboardText.trim()) {
          const isUrl = clipboardText.startsWith('http://') || clipboardText.startsWith('https://')
          clipboardItems.push({
            id: 'clipboard-current',
            type: 'clipboard',
            title: isUrl ? 'Copied URL' : 'Copied Text',
            content: clipboardText.length > 200 ? clipboardText.substring(0, 200) + '...' : clipboardText,
          })
        }
        
        // Load recent files
        const recentFiles = await window.electronAPI.readRecentFiles()
        if (recentFiles && Array.isArray(recentFiles)) {
          recentFiles.forEach((file: { name: string; path: string; extension: string; modified: Date }, index: number) => {
            fileItems.push({
              id: `file-${index}`,
              type: 'file',
              title: file.name,
              content: file.path,
              path: file.path,
              extension: file.extension,
              modified: file.modified,
            })
          })
        }
      } else {
        // Fallback for browser - use clipboard API
        try {
          const clipboardText = await navigator.clipboard.readText()
          if (clipboardText && clipboardText.trim()) {
            const isUrl = clipboardText.startsWith('http://') || clipboardText.startsWith('https://')
            clipboardItems.push({
              id: 'clipboard-current',
              type: 'clipboard',
              title: isUrl ? 'Copied URL' : 'Copied Text',
              content: clipboardText.length > 200 ? clipboardText.substring(0, 200) + '...' : clipboardText,
            })
          }
        } catch (e) {
          // Clipboard access denied
          clipboardItems.push({
            id: 'clipboard-denied',
            type: 'clipboard',
            title: 'Clipboard Access',
            content: 'Click to allow clipboard access',
          })
        }
        
        // Browser placeholder files
        fileItems.push({
          id: 'file-placeholder',
          type: 'file',
          title: 'File Access',
          content: 'Run as desktop app for file access',
          extension: '.txt',
        })
      }
      
      // Placeholder web items (would need browser extension for real data)
      webItems.push({
        id: 'web-placeholder',
        type: 'web',
        title: 'Recent Searches',
        content: 'Web context coming soon',
      })
      
    } catch (error) {
      console.error('Error loading context data:', error)
    }
    
    // Combine all items
    const allItems = [...clipboardItems, ...fileItems, ...webItems]
    
    setContextItems({
      clipboard: clipboardItems,
      files: fileItems,
      web: webItems,
      all: allItems,
    })
    
    setIsLoading(false)
  }, [])

  useEffect(() => {
    if (isOpen) {
      loadContextData()
    }
  }, [isOpen, loadContextData])

  const handleSelectItem = (item: ContextItem) => {
    onSelectContext?.(item)
    onClose()
  }

  const filteredItems = contextItems[activeFilter as keyof typeof contextItems]
    .filter(
      (item) =>
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.content.toLowerCase().includes(searchQuery.toLowerCase())
    )

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
              {/* Filter Sidebar */}
              <div className="w-16 bg-brand-primary/10 backdrop-blur-sm border-r border-brand-light/10 rounded-l-2xl p-2">
                <div className="space-y-2">
                  {filters.map((filter) => (
                    <motion.button
                      key={filter.id}
                      onClick={() => setActiveFilter(filter.id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 ${
                        activeFilter === filter.id
                          ? "bg-brand-primary/20 border border-brand-primary/30"
                          : "bg-brand-light/[0.06] hover:bg-brand-primary/10 border border-brand-light/10"
                      }`}
                    >
                      <filter.icon
                        className={`w-5 h-5 ${
                          activeFilter === filter.id ? "text-brand-accent" : "text-brand-light/70"
                        }`}
                        strokeWidth={1.5}
                      />
                    </motion.button>
                  ))}
                  
                  {/* Refresh button */}
                  <motion.button
                    onClick={loadContextData}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={isLoading}
                    className="w-12 h-12 rounded-xl flex items-center justify-center bg-brand-light/[0.06] hover:bg-brand-primary/10 border border-brand-light/10 transition-all duration-200 disabled:opacity-50"
                  >
                    <RefreshCw
                      className={`w-5 h-5 text-brand-light/70 ${isLoading ? 'animate-spin' : ''}`}
                      strokeWidth={1.5}
                    />
                  </motion.button>
                </div>
              </div>

              {/* Content Area */}
              <div className="flex-1 p-4">
                {/* Search Bar */}
                <div className="relative mb-4">
                  <div className="absolute inset-0 bg-brand-light/[0.06] backdrop-blur-sm rounded-xl border border-brand-light/[0.08]" />
                  <div className="relative flex items-center gap-3 p-3">
                    <Search className="w-4 h-4 text-brand-light/60" strokeWidth={1.5} />
                    <input
                      type="text"
                      placeholder="Filter context..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="flex-1 bg-transparent text-brand-light placeholder-brand-light/40 outline-none font-normal text-sm"
                    />
                  </div>
                </div>

                {/* Context Items */}
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {isLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <RefreshCw className="w-5 h-5 text-brand-light/50 animate-spin" />
                    </div>
                  ) : filteredItems.length === 0 ? (
                    <div className="text-center py-8 text-brand-light/50 text-sm">
                      No context items found
                    </div>
                  ) : (
                    filteredItems.map((item, index) => {
                      const IconComponent = item.extension ? getFileIcon(item.extension) : 
                        item.type === 'clipboard' ? Clipboard :
                        item.type === 'web' ? Globe : FileText
                      
                      return (
                        <motion.div
                          key={item.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          onClick={() => handleSelectItem(item)}
                          className="p-3 bg-brand-light/[0.04] hover:bg-brand-primary/[0.08] rounded-lg border border-brand-light/[0.06] cursor-pointer transition-all duration-200 group"
                        >
                          <div className="flex items-start gap-3">
                            <div className="w-8 h-8 rounded-lg bg-brand-light/10 flex items-center justify-center flex-shrink-0">
                              <IconComponent className="w-4 h-4 text-brand-light/70" strokeWidth={1.5} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-brand-light font-medium text-sm mb-1 truncate">
                                {item.title}
                              </div>
                              <div className="text-brand-light/60 text-xs truncate">
                                {item.content}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )
                    })
                  )}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
