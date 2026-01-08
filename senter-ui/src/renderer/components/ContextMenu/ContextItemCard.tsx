import { Globe, FileText, Copy, Image, Code, X } from 'lucide-react'
import { type ContextItem, type ContextItemType } from '@/types'
import { useStore } from '@/store'

const iconMap: Record<ContextItemType, typeof Globe> = {
  web: Globe,
  file: FileText,
  clipboard: Copy,
  image: Image,
  code: Code,
}

interface ContextItemCardProps {
  item: ContextItem
  onRemove?: () => void  // P3-001: Optional callback for daemon removal
}

export function ContextItemCard({ item, onRemove }: ContextItemCardProps) {
  const { removeContextItem, addToActiveContext } = useStore()
  const Icon = iconMap[item.type]

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (onRemove) {
      onRemove()  // P3-001: Call daemon removal first
    } else {
      removeContextItem(item.id)  // Fallback to store-only removal
    }
  }

  return (
    <div
      className="bg-brand-light/5 rounded-lg p-3 hover:bg-brand-light/10 transition-colors cursor-pointer"
      onClick={() => addToActiveContext(item)}
    >
      <div className="flex items-start gap-3">
        <Icon className="w-5 h-5 mt-1 text-brand-light/60" strokeWidth={1.5} />
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-sm text-brand-light truncate">{item.title}</h4>
          {item.description && (
            <p className="text-xs text-brand-light/60 truncate mt-1">
              {item.description}
            </p>
          )}
        </div>
        <button
          onClick={handleRemove}
          className="p-1 rounded hover:bg-brand-light/10 transition-colors"
        >
          <X className="w-4 h-4 text-brand-light/40" strokeWidth={1.5} />
        </button>
      </div>
    </div>
  )
}
