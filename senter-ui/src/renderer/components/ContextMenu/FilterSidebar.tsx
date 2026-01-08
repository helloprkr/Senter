import { useStore } from '@/store'
import { type ContextFilter } from '@/types'

const filters: { id: ContextFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'web', label: 'Web' },
  { id: 'files', label: 'Files' },
  { id: 'clipboard', label: 'Clipboard' },
]

export function FilterSidebar() {
  const { activeFilter, setFilter } = useStore()

  return (
    <div className="w-48 border-r border-brand-light/10 p-4">
      <h3 className="text-sm font-medium text-brand-light/80 mb-4">Context Sources</h3>
      <div className="space-y-2">
        {filters.map((filter) => (
          <button
            key={filter.id}
            onClick={() => setFilter(filter.id)}
            className={`w-full text-left px-4 py-2 rounded-lg transition-all text-sm ${
              activeFilter === filter.id
                ? 'bg-brand-primary/40 border-l-2 border-brand-accent text-brand-light'
                : 'text-brand-light/70 hover:bg-brand-light/5'
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>
    </div>
  )
}
