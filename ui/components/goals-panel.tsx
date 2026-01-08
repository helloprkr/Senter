"use client"

import { motion } from "framer-motion"
import { Target, TrendingUp, Zap, BookOpen, Code, Briefcase } from "lucide-react"
import { SenterGoal } from "@/lib/senter-api"

interface GoalsPanelProps {
  goals: SenterGoal[]
}

const categoryIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  learning: BookOpen,
  coding: Code,
  career: Briefcase,
  health: TrendingUp,
  default: Target,
}

const categoryColors: Record<string, string> = {
  learning: "text-blue-400",
  coding: "text-green-400",
  career: "text-purple-400",
  health: "text-red-400",
  default: "text-brand-accent",
}

export function GoalsPanel({ goals }: GoalsPanelProps) {
  if (goals.length === 0) {
    return (
      <div className="text-center text-brand-light/60 py-8">
        <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No active goals detected yet.</p>
        <p className="text-xs mt-1 text-brand-light/40">
          Senter will learn your goals from conversations.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3 max-h-64 overflow-y-auto">
      {goals.map((goal, index) => {
        const Icon = categoryIcons[goal.category] || categoryIcons.default
        const colorClass = categoryColors[goal.category] || categoryColors.default

        return (
          <motion.div
            key={goal.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="p-3 rounded-lg bg-brand-light/[0.04] border border-brand-light/[0.06]"
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-lg bg-brand-light/5 ${colorClass}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-brand-light text-sm font-medium truncate">
                    {goal.description}
                  </span>
                  {goal.source === "inferred" && (
                    <span title="AI-inferred goal">
                      <Zap className="w-3 h-3 text-yellow-400 flex-shrink-0" />
                    </span>
                  )}
                </div>

                {/* Progress bar */}
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-brand-light/50 mb-1">
                    <span className="capitalize">{goal.category}</span>
                    <span>{Math.round(goal.progress * 100)}%</span>
                  </div>
                  <div className="h-1.5 bg-brand-light/10 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${goal.progress * 100}%` }}
                      transition={{ duration: 0.5, ease: "easeOut" }}
                      className="h-full bg-gradient-to-r from-brand-primary to-brand-accent rounded-full"
                    />
                  </div>
                </div>

                {goal.confidence && (
                  <div className="mt-1 text-xs text-brand-light/40">
                    Confidence: {Math.round(goal.confidence * 100)}%
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
