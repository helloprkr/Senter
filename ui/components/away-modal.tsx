"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X, Clock, Search, Lightbulb, ExternalLink } from "lucide-react"
import { SenterAwayInfo } from "@/lib/senter-api"

interface AwayModalProps {
  isOpen: boolean
  onClose: () => void
  awayInfo: SenterAwayInfo | null
}

export function AwayModal({ isOpen, onClose, awayInfo }: AwayModalProps) {
  if (!awayInfo) return null

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-md mx-auto z-50"
          >
            <div className="glass-panel-light rounded-2xl overflow-hidden border border-brand-primary/20">
              {/* Header */}
              <div className="p-4 border-b border-brand-light/10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-brand-accent/20">
                    <Clock className="w-5 h-5 text-brand-accent" />
                  </div>
                  <div>
                    <h3 className="text-brand-light font-medium">Welcome Back!</h3>
                    <p className="text-xs text-brand-light/50">
                      You were away for {awayInfo.duration_hours.toFixed(1)} hours
                    </p>
                  </div>
                </div>
                <motion.button
                  onClick={onClose}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  className="p-2 rounded-lg hover:bg-brand-light/10 text-brand-light/50"
                >
                  <X className="w-4 h-4" />
                </motion.button>
              </div>

              {/* Content */}
              <div className="p-4 max-h-[60vh] overflow-y-auto space-y-4">
                {/* Summary */}
                {awayInfo.summary && (
                  <div className="p-3 rounded-lg bg-brand-light/[0.04] border border-brand-light/[0.06]">
                    <p className="text-brand-light/80 text-sm">{awayInfo.summary}</p>
                  </div>
                )}

                {/* Research */}
                {awayInfo.research.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Search className="w-4 h-4 text-brand-accent" />
                      <span className="text-sm font-medium text-brand-light">
                        Research Completed
                      </span>
                    </div>
                    <div className="space-y-2">
                      {awayInfo.research.map((item, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="p-3 rounded-lg bg-brand-light/[0.04] border border-brand-light/[0.06]"
                        >
                          <div className="text-sm text-brand-light font-medium mb-1">
                            {item.goal}
                          </div>
                          <p className="text-xs text-brand-light/60 mb-2">
                            {item.summary}
                          </p>
                          {item.sources.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {item.sources.map((source, i) => (
                                <a
                                  key={i}
                                  href={source}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-brand-primary/20 text-brand-accent hover:bg-brand-primary/30 transition-colors"
                                >
                                  <ExternalLink className="w-3 h-3" />
                                  Source {i + 1}
                                </a>
                              ))}
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Insights */}
                {awayInfo.insights.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <Lightbulb className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm font-medium text-brand-light">
                        Insights
                      </span>
                    </div>
                    <div className="space-y-2">
                      {awayInfo.insights.map((insight, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.2 + index * 0.1 }}
                          className="flex items-start gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20"
                        >
                          <div className="w-1.5 h-1.5 rounded-full bg-yellow-400 mt-1.5 flex-shrink-0" />
                          <p className="text-xs text-brand-light/80">{insight}</p>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-brand-light/10">
                <motion.button
                  onClick={onClose}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full py-2 rounded-lg bg-brand-accent text-brand-light font-medium text-sm"
                >
                  Got it, thanks!
                </motion.button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
