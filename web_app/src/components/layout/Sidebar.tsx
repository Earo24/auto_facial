import { NavLink, useLocation } from 'react-router-dom'
import {
  Home,
  Upload,
  Users,
  Film,
  Settings,
  ChevronLeft,
  ChevronRight,
  Clapperboard,
} from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: '仪表板', href: '/', icon: Home },
  { name: '剧集管理', href: '/series', icon: Clapperboard },
  { name: '视频列表', href: '/videos', icon: Film },
  { name: '视频处理', href: '/processing', icon: Upload },
  { name: '聚类标注', href: '/clustering', icon: Users },
  { name: '系统设置', href: '/settings', icon: Settings },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen bg-background-200 border-r border-background-400 transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b border-background-400 px-4">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
              <span className="text-lg font-bold">AF</span>
            </div>
            <span className="text-lg font-semibold">AutoFacial</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            'rounded-lg p-1.5 text-gray-400 hover:bg-background-300 hover:text-white transition-colors',
            collapsed && 'mx-auto'
          )}
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="space-y-1 p-3">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200 cursor-pointer',
                isActive
                  ? 'bg-primary-600/20 text-primary-400'
                  : 'text-gray-400 hover:bg-background-300 hover:text-white'
              )}
            >
              <item.icon size={20} className={cn('flex-shrink-0', isActive && 'text-primary-400')} />
              {!collapsed && <span>{item.name}</span>}
            </NavLink>
          )
        })}
      </nav>

      {/* Version info */}
      {!collapsed && (
        <div className="absolute bottom-0 left-0 right-0 border-t border-background-400 p-4">
          <p className="text-xs text-gray-500">版本 0.1.0</p>
        </div>
      )}
    </aside>
  )
}
