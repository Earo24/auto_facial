import { Bell, Search, User } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'

export function Header() {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-background-400 bg-background-200/95 backdrop-blur supports-[backdrop-filter]:bg-background-200/60">
      {/* Search */}
      <div className="flex items-center gap-4">
        <div className="relative w-80">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <Input
            type="search"
            placeholder="搜索视频、角色..."
            className="pl-10"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm">
          <Bell size={18} />
        </Button>

        <Button variant="ghost" size="sm">
          <User size={18} />
        </Button>
      </div>
    </header>
  )
}
