"use client"

import { motion } from "framer-motion"
import { Activity, Code, Globe, FileText, Gamepad2, Play, HelpCircle } from "lucide-react"
import { SenterActivity } from "@/lib/senter-api"

interface ActivityBadgeProps {
  activity: SenterActivity | null
  connected: boolean
}

const contextIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  coding: Code,
  browsing: Globe,
  writing: FileText,
  gaming: Gamepad2,
  media: Play,
  unknown: HelpCircle,
}

const contextColors: Record<string, string> = {
  coding: "bg-green-500/20 text-green-400 border-green-500/30",
  browsing: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  writing: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  gaming: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  media: "bg-red-500/20 text-red-400 border-red-500/30",
  unknown: "bg-gray-500/20 text-gray-400 border-gray-500/30",
}

export function ActivityBadge({ activity, connected }: ActivityBadgeProps) {
  if (!connected) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/20 border border-red-500/30"
      >
        <div className="w-2 h-2 rounded-full bg-red-500" />
        <span className="text-xs text-red-400">Senter Offline</span>
      </motion.div>
    )
  }

  if (!activity) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-primary/20 border border-brand-primary/30"
      >
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-2 h-2 rounded-full bg-brand-accent"
        />
        <span className="text-xs text-brand-light/70">Watching...</span>
      </motion.div>
    )
  }

  const Icon = contextIcons[activity.context] || contextIcons.unknown
  const colorClass = contextColors[activity.context] || contextColors.unknown

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${colorClass}`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span className="text-xs capitalize">{activity.context}</span>
      {activity.project && (
        <>
          <div className="w-1 h-1 rounded-full bg-current opacity-50" />
          <span className="text-xs opacity-70 truncate max-w-[100px]">
            {activity.project}
          </span>
        </>
      )}
      <div className="w-1 h-1 rounded-full bg-current opacity-50" />
      <span className="text-xs opacity-50">{activity.duration_minutes}m</span>
    </motion.div>
  )
}
