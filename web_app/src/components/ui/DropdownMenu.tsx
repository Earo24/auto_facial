import { forwardRef, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface DropdownMenuProps {
  trigger: ReactNode
  children: ReactNode
  align?: 'start' | 'center' | 'end'
}

export function DropdownMenu({ trigger, children, align = 'start' }: DropdownMenuProps) {
  return (
    <div className="relative inline-block text-left">
      {trigger}
      <div className={cn(
        "absolute z-50 mt-2 w-56 rounded-lg bg-background-200 border border-background-400 shadow-card-lg focus:outline-none",
        align === 'end' && 'right-0',
        align === 'center' && 'left-1/2 -translate-x-1/2',
        align === 'start' && 'left-0'
      )}>
        <div className="py-1">
          {children}
        </div>
      </div>
    </div>
  )
}

export const DropdownMenuItem = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement>
>(({ className, children, ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      'flex w-full items-center px-4 py-2 text-sm text-gray-300 hover:bg-background-300 hover:text-white cursor-pointer transition-colors',
      className
    )}
    {...props}
  >
    {children}
  </button>
))

DropdownMenuItem.displayName = 'DropdownMenuItem'
