import { forwardRef, type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export const Badge = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & { variant?: 'primary' | 'success' | 'warning' | 'error' }>(
  ({ className, variant = 'primary', ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'badge',
        {
          'badge-primary': variant === 'primary',
          'badge-success': variant === 'success',
          'badge-warning': variant === 'warning',
          'badge-error': variant === 'error',
        },
        className
      )}
      {...props}
    />
  )
)

Badge.displayName = 'Badge'
