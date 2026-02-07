import { useState, useEffect } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Users,
  Film,
  Clock,
  Image as ImageIcon,
  Filter,
  Search,
  User,
  Clapperboard,
  Eye,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from 'lucide-react'
import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'

interface FaceSample {
  sample_id: string
  video_id: string
  video_filename: string
  character_id: string
  image_path: string
  timestamp: number
  frame_number: number
  quality_score: number
}

interface SeriesCharacter {
  actor_id: string
  actor_name: string
  photo_path: string | null
  character_name: string
  role_order: number
  is_main_character: boolean
  total_samples: number
  videos_appeared: number
  first_appearance: number | null
  last_appearance: number | null
}

interface SeriesData {
  series_id: string
  series_name: string
  series_year: number | null
  description: string | null
  total_actors: number
  total_videos: number
  characters: SeriesCharacter[]
}

export default function SeriesCharacters() {
  const { seriesId } = useParams()
  const navigate = useNavigate()

  const [series, setSeries] = useState<SeriesData | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterMainOnly, setFilterMainOnly] = useState(false)
  const [selectedActor, setSelectedActor] = useState<SeriesCharacter | null>(null)
  const [expandedActors, setExpandedActors] = useState<Set<string>>(new Set())
  const [actorFaces, setActorFaces] = useState<Record<string, FaceSample[]>>({})
  const [loadingFaces, setLoadingFaces] = useState<Set<string>>(new Set())
  const [reclustering, setReclustering] = useState(false)
  const [reclusterMessage, setReclusterMessage] = useState<string | null>(null)

  useEffect(() => {
    if (seriesId) {
      loadSeriesCharacters(seriesId)
    }
  }, [seriesId])

  const loadSeriesCharacters = async (id: string) => {
    setLoading(true)
    try {
      const res = await fetch(`http://localhost:8000/api/series/${id}/characters`)
      const data = await res.json()
      if (data.error) {
        console.error('加载角色失败:', data.error)
        return
      }
      setSeries(data)
    } catch (error) {
      console.error('加载角色失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadActorFaces = async (actorId: string) => {
    if (actorFaces[actorId]) return // Already loaded

    setLoadingFaces(prev => new Set(prev).add(actorId))
    try {
      const res = await fetch(`http://localhost:8000/api/series/${seriesId}/faces?actor_id=${actorId}&limit=50`)
      const data = await res.json()
      if (data.samples) {
        setActorFaces(prev => ({ ...prev, [actorId]: data.samples }))
      }
    } catch (error) {
      console.error('加载人脸样本失败:', error)
    } finally {
      setLoadingFaces(prev => {
        const newSet = new Set(prev)
        newSet.delete(actorId)
        return newSet
      })
    }
  }

  const toggleActorExpansion = (actorId: string) => {
    setExpandedActors(prev => {
      const newSet = new Set(prev)
      if (newSet.has(actorId)) {
        newSet.delete(actorId)
      } else {
        newSet.add(actorId)
        loadActorFaces(actorId)
      }
      return newSet
    })
  }

  const handleRecluster = async () => {
    if (!series || series.total_videos === 0) {
      setReclusterMessage('该剧集暂无视频')
      return
    }

    // 获取第一个视频ID（实际应用中应该让用户选择）
    const videoId = series.total_videos > 0 ? await getFirstVideoId(seriesId) : null
    if (!videoId) {
      setReclusterMessage('未找到可用的视频')
      return
    }

    setReclustering(true)
    setReclusterMessage(null)

    try {
      const res = await fetch(`http://localhost:8000/api/series/${seriesId}/recluster`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
      })
      const data = await res.json()

      if (data.success) {
        setReclusterMessage(`重新聚类完成！匹配了 ${data.matched_clusters}/${data.total_clusters} 个簇`)
        // 刷新角色数据
        await loadSeriesCharacters(seriesId)
        // 清空缓存的演员人脸数据
        setActorFaces({})
        setExpandedActors(new Set())
      } else {
        setReclusterMessage(`重新聚类失败: ${data.error}`)
      }
    } catch (error) {
      console.error('重新聚类失败:', error)
      setReclusterMessage('重新聚类失败，请稍后重试')
    } finally {
      setReclustering(false)
      // 3秒后清除消息
      setTimeout(() => setReclusterMessage(null), 3000)
    }
  }

  const getFirstVideoId = async (seriesId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/videos`)
      const videos = await res.json()
      const seriesVideos = videos.filter((v: any) => v.series_id === seriesId)
      return seriesVideos.length > 0 ? seriesVideos[0].video_id : null
    } catch {
      return null
    }
  }

  const filteredCharacters = series?.characters.filter(char => {
    // 搜索过滤
    if (searchQuery && !char.actor_name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !char.character_name.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false
    }
    // 主演过滤
    if (filterMainOnly && !char.is_main_character) {
      return false
    }
    return true
  }).sort((a, b) => a.role_order - b.role_order) || []

  const formatTime = (seconds: number | null) => {
    if (!seconds) return '-'
    const minutes = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-t-primary-500 border-solid rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">加载角色数据中...</p>
        </div>
      </div>
    )
  }

  if (!series) {
    return (
      <div className="text-center py-16">
        <Clapperboard className="mx-auto h-16 w-16 text-gray-600 mb-4" />
        <p className="text-gray-500">未找到剧集信息</p>
        <Link to="/series">
          <Button variant="primary" className="mt-4">
            返回剧集管理
          </Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/series">
          <Button variant="ghost" size="sm">
            <ArrowLeft size={18} />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-100">{series.series_name}</h1>
          <p className="text-gray-400 flex items-center gap-2">
            {series.series_year && `${series.series_year}年`}
            {series.description && ` · ${series.description}`}
          </p>
        </div>
        <Button
          variant="accent"
          onClick={handleRecluster}
          disabled={reclustering}
          className="gap-2"
        >
          <RefreshCw size={18} className={reclustering ? 'animate-spin' : ''} />
          {reclustering ? '重新聚类中...' : '重新聚类'}
        </Button>
      </div>

      {/* Recluster Message */}
      {reclusterMessage && (
        <Card className={cn(
          "p-4 border-l-4",
          reclusterMessage.includes('完成') ? 'border-green-500 bg-green-600/10' :
          reclusterMessage.includes('失败') ? 'border-red-500 bg-red-600/10' :
          'border-blue-500 bg-blue-600/10'
        )}>
          <p className={cn(
            "text-sm",
            reclusterMessage.includes('完成') ? 'text-green-400' :
            reclusterMessage.includes('失败') ? 'text-red-400' :
            'text-blue-400'
          )}>
            {reclusterMessage}
          </p>
        </Card>
      )}

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-purple-600/20 p-2.5 text-purple-400">
              <Users size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">角色总数</p>
              <p className="text-xl font-bold text-gray-100">{series.total_actors}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-600/20 p-2.5 text-blue-400">
              <Film size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">视频数量</p>
              <p className="text-xl font-bold text-gray-100">{series.total_videos}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-green-600/20 p-2.5 text-green-400">
              <User size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">已标注角色</p>
              <p className="text-xl font-bold text-gray-100">{series.characters.filter(c => c.total_samples > 0).length}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-yellow-600/20 p-2.5 text-yellow-400">
              <Clock size={20} />
            </div>
            <div>
              <p className="text-xs text-gray-500">总样本数</p>
              <p className="text-xl font-bold text-gray-100">{series.characters.reduce((sum, c) => sum + c.total_samples, 0)}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
              <input
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="搜索演员或角色..."
                className="w-full bg-background-500 border border-background-500 rounded-lg px-10 py-2 text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={filterMainOnly}
                onChange={e => setFilterMainOnly(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">只看主演</span>
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Characters List */}
      {filteredCharacters.length === 0 ? (
        <Card>
          <CardContent className="p-16 text-center">
            <Users className="mx-auto h-16 w-16 text-gray-600 mb-4" />
            <p className="text-gray-500 mb-2">
              {searchQuery ? '没有找到匹配的角色' : '暂无角色数据'}
            </p>
            {!searchQuery && (
              <p className="text-sm text-gray-600">
                请先在"视频处理"页面上传视频并完成聚类标注
              </p>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCharacters.map((character) => {
            const isExpanded = expandedActors.has(character.actor_id)
            const faces = actorFaces[character.actor_id] || []
            const isLoading = loadingFaces.has(character.actor_id)

            return (
              <Card
                key={character.actor_id}
                className={cn(
                  'hover:border-primary-500/50 transition-all',
                  selectedActor?.actor_id === character.actor_id && 'border-primary-500 bg-primary-600/10',
                  isExpanded && 'md:col-span-2 lg:col-span-3'
                )}
              >
                <CardContent className="p-4">
                  {/* Actor Header */}
                  <div
                    className="flex items-start gap-4 cursor-pointer"
                    onClick={() => setSelectedActor(character)}
                  >
                    {/* Actor Photo */}
                    <div className="flex-shrink-0">
                      {character.photo_path ? (
                        <img
                          src={`http://localhost:8000/api/actor_photos/${character.actor_id}.jpg`}
                          alt={character.actor_name}
                          className="w-16 h-16 rounded-full object-cover border-2 border-background-500"
                        />
                      ) : (
                        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-600/20 to-primary-400/20 flex items-center justify-center">
                          <User className="text-primary-400" size={24} />
                        </div>
                      )}
                    </div>

                    {/* Actor Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-gray-100">{character.actor_name}</h3>
                        {character.is_main_character && (
                          <Badge variant="primary" size="sm">主演</Badge>
                        )}
                      </div>
                      <p className="text-sm text-gray-400">{character.character_name}</p>

                      {/* Stats */}
                      <div className="mt-3 space-y-1">
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <span className="flex items-center gap-1">
                            <ImageIcon size={12} />
                            {character.total_samples} 样本
                          </span>
                          <span className="flex items-center gap-1">
                            <Film size={12} />
                            {character.videos_appeared} 视频
                          </span>
                        </div>
                        {(character.first_appearance || character.last_appearance) && (
                          <div className="text-xs text-gray-500">
                            {formatTime(character.first_appearance)} - {formatTime(character.last_appearance)}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Expand Button */}
                    {character.total_samples > 0 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleActorExpansion(character.actor_id)
                        }}
                        className="flex-shrink-0 p-2 rounded-lg hover:bg-background-300 transition-colors"
                      >
                        {isExpanded ? (
                          <ChevronUp className="text-gray-400" size={20} />
                        ) : (
                          <ChevronDown className="text-gray-400" size={20} />
                        )}
                      </button>
                    )}
                  </div>

                  {/* Face Samples Grid */}
                  {isExpanded && character.total_samples > 0 && (
                    <div className="mt-4 pt-4 border-t border-background-500/50">
                      {isLoading ? (
                        <div className="text-center py-8 text-gray-500">
                          <div className="w-8 h-8 border-2 border-t-primary-500 border-solid rounded-full animate-spin mx-auto mb-2"></div>
                          <p className="text-sm">加载人脸样本中...</p>
                        </div>
                      ) : faces.length === 0 ? (
                        <div className="text-center py-8 text-gray-500">
                          <ImageIcon className="mx-auto h-12 w-12 mb-2 text-gray-600" />
                          <p className="text-sm">暂无人脸样本</p>
                        </div>
                      ) : (
                        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-2">
                          {faces.map((face) => (
                            <div
                              key={face.sample_id}
                              className="group relative aspect-square rounded-lg overflow-hidden bg-background-300 hover:ring-2 hover:ring-primary-500 transition-all cursor-pointer"
                              title={`${face.video_filename} - ${formatTime(face.timestamp)}`}
                            >
                              <img
                                src={`http://localhost:8000${face.image_path}`}
                                alt={`Face from ${face.video_filename}`}
                                className="w-full h-full object-cover"
                                loading="lazy"
                              />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center">
                                <span className="text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                  {face.quality_score?.toFixed(1)}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
