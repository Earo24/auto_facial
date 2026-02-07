import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Film,
  Users,
  Clock,
  CheckCircle,
  Eye,
  Search,
  Filter,
  Calendar,
  BarChart2,
  Clapperboard,
  Link2,
  Unlink,
  X,
} from 'lucide-react'
import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import { api, VideoInfo } from '@/services/api'
import { cn } from '@/lib/utils'

interface Series {
  series_id: string
  name: string
  year?: number
}

export default function Videos() {
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<'all' | 'completed' | 'processing'>('all')
  const [series, setSeries] = useState<Series[]>([])
  const [showSeriesDialog, setShowSeriesDialog] = useState(false)
  const [selectedVideo, setSelectedVideo] = useState<VideoInfo | null>(null)

  useEffect(() => {
    loadVideos()
    loadSeries()
  }, [])

  useEffect(() => {
    // 定期刷新视频列表（每5秒）
    const interval = setInterval(loadVideos, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadVideos = async () => {
    try {
      const data = await api.getVideos()
      setVideos(data)
    } catch (error) {
      console.error('加载视频失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadSeries = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/series')
      const data = await res.json()
      setSeries(data.series || [])
    } catch (error) {
      console.error('加载剧集失败:', error)
    }
  }

  const handleAssociate = (video: VideoInfo) => {
    setSelectedVideo(video)
    setShowSeriesDialog(true)
  }

  const handleDisassociate = async (videoId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/videos/${videoId}/series`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ series_id: null })
      })
      if (res.ok) {
        loadVideos()
      }
    } catch (error) {
      console.error('解除关联失败:', error)
    }
  }

  const handleSelectSeries = async (seriesId: string) => {
    if (!selectedVideo) return

    try {
      const res = await fetch(`http://localhost:8000/api/videos/${selectedVideo.video_id}/series`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ series_id: seriesId })
      })
      if (res.ok) {
        setShowSeriesDialog(false)
        loadVideos()
      }
    } catch (error) {
      console.error('关联失败:', error)
    }
  }

  const getStatus = (video: VideoInfo) => {
    if (video.detected_faces > 0) {
      return { text: '已完成', variant: 'success' as const }
    } else if (video.processed_frames > 0) {
      return { text: '处理中', variant: 'warning' as const }
    } else {
      return { text: '待处理', variant: 'error' as const }
    }
  }

  const filteredVideos = videos
    .filter(v => {
      // 搜索过滤
      if (searchQuery && !v.filename.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false
      }
      // 状态过滤
      if (filterStatus === 'completed' && v.detected_faces === 0) return false
      if (filterStatus === 'processing' && v.processed_frames === 0) return false
      if (filterStatus === 'processing' && v.detected_faces > 0) return false
      return true
    })
    .sort((a, b) => {
      // 按创建时间倒序
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })

  const stats = {
    total: videos.length,
    completed: videos.filter(v => v.detected_faces > 0).length,
    processing: videos.filter(v => v.processed_frames > 0 && v.detected_faces === 0).length,
    pending: videos.filter(v => v.processed_frames === 0).length,
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">视频列表</h1>
          <p className="text-gray-400">查看和管理所有已上传的视频</p>
        </div>
        <Link to="/processing">
          <Button variant="accent" className="gap-2">
            <Film size={18} />
            上传视频
          </Button>
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-600/20 p-2.5 text-blue-400">
              <Film size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">视频总数</p>
              <p className="text-xl font-bold text-gray-100">{stats.total}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-green-600/20 p-2.5 text-green-400">
              <CheckCircle size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">已完成</p>
              <p className="text-xl font-bold text-gray-100">{stats.completed}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-yellow-600/20 p-2.5 text-yellow-400">
              <Clock size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">处理中</p>
              <p className="text-xl font-bold text-gray-100">{stats.processing}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-gray-600/20 p-2.5 text-gray-400">
              <Users size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">待处理</p>
              <p className="text-xl font-bold text-gray-100">{stats.pending}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Search and Filter */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
              <Input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="搜索视频..."
                className="pl-10"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter size={18} className="text-gray-500" />
              <select
                value={filterStatus}
                onChange={e => setFilterStatus(e.target.value as any)}
                className="bg-background-500 border border-background-500 rounded-lg px-3 py-2 text-gray-100 focus:ring-2 focus:ring-primary-500"
              >
                <option value="all">全部状态</option>
                <option value="completed">已完成</option>
                <option value="processing">处理中</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Video List */}
      {loading ? (
        <div className="text-center py-16 text-gray-500">
          <Clock className="mx-auto h-12 w-12 mb-4 animate-spin" />
          <p>加载中...</p>
        </div>
      ) : filteredVideos.length === 0 ? (
        <Card>
          <CardContent className="p-16 text-center">
            <Film className="mx-auto h-16 w-16 text-gray-600 mb-4" />
            <p className="text-gray-500 mb-2">
              {searchQuery ? '没有找到匹配的视频' : '暂无视频'}
            </p>
            {!searchQuery && (
              <Link to="/processing">
                <Button variant="primary" className="mt-4">
                  上传第一个视频
                </Button>
              </Link>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredVideos.map((video) => {
            const status = getStatus(video)
            return (
              <Card key={video.video_id} className="hover:border-primary-500/50 transition-colors">
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    {/* Video Icon */}
                    <div className="rounded-lg bg-background-300 p-3">
                      <Film className="text-gray-400" size={24} />
                    </div>

                    {/* Video Info */}
                    <div className="flex-1 min-w-0">
                      {/* 剧集信息 - 更突出显示 */}
                      {video.series_name ? (
                        <div className="flex items-center gap-2 mb-2">
                          <Link to={`/series/${video.series_id}/characters`} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary-600/10 border border-primary-500/30 hover:bg-primary-600/20 transition-colors">
                            <Clapperboard size={14} className="text-primary-400" />
                            <span className="text-sm font-medium text-primary-300">
                              {video.series_name}
                              {video.series_year && ` (${video.series_year})`}
                            </span>
                          </Link>
                          <span className="text-gray-500">›</span>
                          <span className="text-sm text-gray-400">{video.filename}</span>
                          <Badge variant={status.variant}>{status.text}</Badge>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-gray-400 text-sm">未关联剧集</span>
                          <span className="text-gray-500">·</span>
                          <p className="font-medium text-gray-100 truncate">{video.filename}</p>
                          <Badge variant={status.variant}>{status.text}</Badge>
                        </div>
                      )}

                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock size={14} />
                          {Math.round(video.duration / 60)} 分钟
                        </span>
                        {video.detected_faces > 0 && (
                          <>
                            <span className="flex items-center gap-1">
                              <Users size={14} />
                              {video.detected_faces} 人脸
                            </span>
                            <span className="flex items-center gap-1">
                              <CheckCircle size={14} />
                              {video.characters_found} 角色
                            </span>
                          </>
                        )}
                      </div>

                      {video.detected_faces > 0 && (
                        <div className="mt-2 flex items-center gap-2 text-xs">
                          <span className="text-gray-500">
                            上传于: {new Date(video.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      {video.series_name ? (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDisassociate(video.video_id)}
                          className="gap-1 hover:bg-red-600/10 hover:text-red-400 hover:border-red-500/50"
                        >
                          <Unlink size={14} />
                          解除关联
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleAssociate(video)}
                          className="gap-1"
                        >
                          <Link2 size={14} />
                          关联剧集
                        </Button>
                      )}

                      {video.detected_faces > 0 ? (
                        <Link to={`/clustering?video=${video.video_id}`}>
                          <Button variant="primary" size="sm" className="gap-1">
                            <Eye size={16} />
                            查看结果
                          </Button>
                        </Link>
                      ) : video.processed_frames > 0 ? (
                        <Button variant="outline" size="sm" disabled>
                          <Clock size={16} className="animate-spin" />
                          处理中
                        </Button>
                      ) : (
                        <Button variant="outline" size="sm" disabled>
                          等待处理
                        </Button>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {/* 关联剧集对话框 */}
      {showSeriesDialog && selectedVideo && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-gray-100">关联剧集</h3>
              <button
                onClick={() => setShowSeriesDialog(false)}
                className="text-gray-400 hover:text-gray-200"
              >
                <X size={20} />
              </button>
            </div>

            <p className="text-gray-400 mb-4">
              选择要关联到视频 <span className="text-primary-400 font-medium">{selectedVideo.filename}</span> 的剧集
            </p>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {series.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  暂无剧集，请先创建剧集
                </div>
              ) : (
                series.map(s => (
                  <button
                    key={s.series_id}
                    onClick={() => handleSelectSeries(s.series_id)}
                    className="w-full flex items-center gap-3 p-3 rounded-lg border border-background-500 hover:border-primary-500/50 hover:bg-background-300 transition-colors text-left"
                  >
                    <Clapperboard size={18} className="text-primary-400" />
                    <div className="flex-1">
                      <p className="font-medium text-gray-100">{s.name}</p>
                      <p className="text-sm text-gray-400">{s.year}年</p>
                    </div>
                  </button>
                ))
              )}
            </div>

            <div className="flex justify-end mt-6">
              <Button
                onClick={() => setShowSeriesDialog(false)}
                className="px-5 py-2.5 border border-background-500 text-gray-300 rounded-lg hover:bg-background-500 transition-colors"
              >
                取消
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
