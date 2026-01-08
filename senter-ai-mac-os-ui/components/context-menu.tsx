"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Search, Globe, FileText, Clipboard, Grid3X3 } from "lucide-react"
import { useState } from "react"

interface ContextMenuProps {
  isOpen: boolean
  onClose: () => void
}

export function ContextMenu({ isOpen, onClose }: ContextMenuProps) {
  const [activeFilter, setActiveFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")

  const filters = [
    { id: "web", icon: Globe, label: "Web" },
    { id: "files", icon: FileText, label: "Files" },
    { id: "clipboard", icon: Clipboard, label: "Clipboard" },
    { id: "all", icon: Grid3X3, label: "All" },
  ]

  const contextItems = {
    web: [
      { title: "Recent Search", content: "Machine learning tutorials" },
      { title: "Bookmarked Page", content: "React documentation" },
    ],
    files: [
      { title: "Document.pdf", content: "Project specifications" },
      { title: "Notes.txt", content: "Meeting notes from yesterday" },
    ],
    clipboard: [
      { title: "Copied Text", content: "const handleSubmit = () => {..." },
      { title: "URL", content: "https://example.com/api/docs" },
    ],
    all: [
      { title: "Recent Search", content: "Machine learning tutorials" },
      { title: "Document.pdf", content: "Project specifications" },
      { title: "Copied Text", content: "const handleSubmit = () => {..." },
    ],
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 20, height: 0 }}
          animate={{ opacity: 1, y: 0, height: "auto" }}
          exit={{ opacity: 0, y: 20, height: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="absolute bottom-full left-0 right-0 mb-2 overflow-hidden"
        >
          <div className="relative bg-brand-light/[0.08] backdrop-blur-3xl border border-brand-light/15 rounded-2xl shadow-2xl">
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
                  {contextItems[activeFilter as keyof typeof contextItems]
                    .filter(
                      (item) =>
                        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                        item.content.toLowerCase().includes(searchQuery.toLowerCase()),
                    )
                    .map((item, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="p-3 bg-brand-light/[0.04] hover:bg-brand-primary/[0.08] rounded-lg border border-brand-light/[0.06] cursor-pointer transition-all duration-200"
                      >
                        <div className="text-brand-light font-medium text-sm mb-1">{item.title}</div>
                        <div className="text-brand-light/60 text-xs truncate">{item.content}</div>
                      </motion.div>
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
