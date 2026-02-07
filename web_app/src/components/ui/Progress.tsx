import { forwardRef } from 'react'
import { cn } from '@/lib/utils'

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number
  max?: number
  showLabel?: boolean
}

export const Progress = forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, max = 100, showLabel = false, ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

    return (
      <div ref={ref} className={cn('space-y-1', className)} {...props}>
        {showLabel && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Progress</span>
            <span className="text-gray-300">{Math.round(percentage)}%</span>
          </div>
        )}
        <div className="progress-bar">
          <div
            className="progress-value"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    )
  }
)

Progress.displayName = 'Progress'
