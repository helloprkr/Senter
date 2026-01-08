"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X, Sparkles, ArrowRight } from "lucide-react"
import { SenterSuggestion } from "@/lib/senter-api"

interface SuggestionsToastProps {
  suggestions: SenterSuggestion[]
  onDismiss: (index: number) => void
  onAccept: (suggestion: SenterSuggestion) => void
}

export function SuggestionsToast({ suggestions, onDismiss, onAccept }: SuggestionsToastProps) {
  const visibleSuggestions = suggestions.slice(0, 3)

  return (
    <div className="fixed bottom-4 right-4 z-40 space-y-2 max-w-sm">
      <AnimatePresence mode="popLayout">
        {visibleSuggestions.map((suggestion, index) => (
          <motion.div
            key={`${suggestion.type}-${index}`}
            initial={{ opacity: 0, x: 100, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.9 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            layout
            className="glass-panel-light rounded-xl border border-brand-primary/20 overflow-hidden"
          >
            <div className="p-3">
              <div className="flex items-start gap-3">
                <div className="p-1.5 rounded-lg bg-brand-accent/20 flex-shrink-0">
                  <Sparkles className="w-4 h-4 text-brand-accent" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-brand-accent capitalize">
                      {suggestion.type}
                    </span>
                    <motion.button
                      onClick={() => onDismiss(index)}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="p-1 rounded hover:bg-brand-light/10 text-brand-light/40"
                    >
                      <X className="w-3 h-3" />
                    </motion.button>
                  </div>
                  <p className="text-sm text-brand-light/80 mt-1">
                    {suggestion.content}
                  </p>
                  {suggestion.reason && (
                    <p className="text-xs text-brand-light/50 mt-1">
                      {suggestion.reason}
                    </p>
                  )}
                  <motion.button
                    onClick={() => onAccept(suggestion)}
                    whileHover={{ scale: 1.02, x: 2 }}
                    whileTap={{ scale: 0.98 }}
                    className="mt-2 flex items-center gap-1 text-xs text-brand-accent hover:text-brand-accent/80"
                  >
                    <span>Try this</span>
                    <ArrowRight className="w-3 h-3" />
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
