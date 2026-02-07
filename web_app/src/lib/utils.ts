import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`
}

export function formatTimestamp(timestamp: number): string {
  return timestamp.toFixed(2)
}

export function getQualityColor(quality: number): string {
  if (quality >= 0.8) return 'text-success'
  if (quality >= 0.6) return 'text-warning'
  return 'text-error'
}

export function getQualityBadgeColor(quality: number): string {
  if (quality >= 0.8) return 'badge-success'
  if (quality >= 0.6) return 'badge-warning'
  return 'badge-error'
}
