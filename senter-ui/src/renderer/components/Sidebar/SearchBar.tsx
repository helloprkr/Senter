import { useEffect, useRef } from 'react'
import { Search } from 'lucide-react'
import { useStore } from '@/store'

export function SearchBar() {
  const inputRef = useRef<HTMLInputElement>(null)
  const { searchQuery, setSearchQuery, isSearchFocused, setSearchFocused } = useStore()

  // Handle ⌘K keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }
      if (e.key === 'Escape' && isSearchFocused) {
        inputRef.current?.blur()
        setSearchFocused(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isSearchFocused, setSearchFocused])

  return (
    <div className="px-6 mb-4">
      <div className="relative">
        <Search
          className={`absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 transition-colors ${
            isSearchFocused ? 'text-brand-accent' : 'text-brand-light/60'
          }`}
          strokeWidth={1.5}
        />
        <input
          ref={inputRef}
          type="text"
          placeholder="Search (⌘K)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onFocus={() => setSearchFocused(true)}
          onBlur={() => setSearchFocused(false)}
          className={`w-full bg-brand-light/5 rounded-xl py-3 pl-10 pr-4 text-sm text-brand-light placeholder-brand-light/40 outline-none transition-all ${
            isSearchFocused ? 'bg-brand-light/10 ring-2 ring-brand-accent/50' : ''
          }`}
        />
      </div>
    </div>
  )
}
