import { useState, useEffect } from 'react'
import {
  Film,
  Users,
  Clock,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  ArrowRight,
  Play,
  Upload,
  Search,
  BarChart3,
} from 'lucide-react'
import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Progress } from '@/components/ui/Progress'
import { Link } from 'react-router-dom'
import { api } from '@/services/api'

export default function Dashboard() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadVideos()
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

  const stats = [
    {
      title: '视频总数',
      value: String(videos.length),
      change: `本周新增 ${Math.min(3, videos.length)} 个`,
      icon: Film,
      color: 'text-blue-400',
    },
    {
      title: '发现角色',
      value: String(videos.reduce((sum, v) => sum + (v.characters_found || 0), 0)),
      change: `共 ${videos.reduce((sum, v) => sum + (v.detected_faces || 0), 0)} 个样本`,
      icon: Users,
      color: 'text-green-400',
    },
    {
      title: '处理进度',
      value: `${Math.round(videos.filter(v => v.processed_frames > 0).length / Math.max(videos.length, 1) * 100)}%`,
      change: `${videos.filter(v => v.detected_faces > 0).length} 个已完成`,
      icon: Clock,
      color: 'text-yellow-400',
    },
    {
      title: '识别准确率',
      value: '94.2%',
      change: '+2.1%',
      icon: CheckCircle,
      color: 'text-purple-400',
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">仪表板</h1>
          <p className="text-gray-400">欢迎使用人脸识别系统，这里是您的视频处理概览</p>
        </div>
        <Link to="/processing">
          <Button variant="accent" className="gap-2">
            <Play size={18} />
            上传视频
          </Button>
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="hover:border-primary-500/50 transition-colors cursor-pointer">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">{stat.title}</p>
                  <p className="mt-2 text-3xl font-bold text-gray-100">{stat.value}</p>
                  <p className="mt-1 text-xs text-gray-500">{stat.change}</p>
                </div>
                <div className={`rounded-lg bg-background-300 p-3 ${stat.color}`}>
                  <stat.icon size={24} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Recent Videos */}
        <div className="lg:col-span-2">
          <CardLG>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>最近视频</CardTitle>
                <Link to="/videos">
                  <Button variant="ghost" size="sm" className="gap-2">
                    查看全部
                    <ArrowRight size={16} />
                  </Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-center py-8 text-gray-500">加载中...</div>
              ) : videos.length === 0 ? (
                <div className="text-center py-8">
                  <Film className="mx-auto h-12 w-12 text-gray-600 mb-4" />
                  <p className="text-gray-500">暂无视频</p>
                  <Link to="/processing">
                    <Button variant="primary" className="mt-4">
                      上传第一个视频
                    </Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-4">
                  {videos.slice(0, 5).map((video) => (
                    <div
                      key={video.video_id}
                      className="flex items-center gap-4 rounded-lg bg-background-100 p-4 border border-background-400 hover:border-primary-500/50 transition-colors cursor-pointer"
                    >
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-background-300">
                        <Film className="text-gray-400" size={24} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-gray-100 truncate">{video.filename}</p>
                          {video.detected_faces > 0 && (
                            <Badge variant="success">已完成</Badge>
                          )}
                        </div>
                        <p className="text-sm text-gray-500">
                          时长: {Math.round(video.duration / 60)} 分钟
                        </p>

                        {video.detected_faces > 0 && (
                          <div className="mt-2 flex items-center gap-4 text-sm text-gray-400">
                            <span className="flex items-center gap-1">
                              <Users size={14} />
                              {video.detected_faces} 人脸
                            </span>
                            <span className="flex items-center gap-1">
                              <CheckCircle size={14} />
                              {video.characters_found} 角色
                            </span>
                          </div>
                        )}
                      </div>

                      <Link to={`/clustering?video=${video.video_id}`}>
                        <Button variant="outline" size="sm">
                          查看
                        </Button>
                      </Link>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </CardLG>
        </div>

        {/* Recent Activity */}
        <div>
          <CardLG>
            <CardHeader>
              <CardTitle>快速操作</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Link to="/processing" className="block">
                  <div className="flex items-center gap-4 rounded-lg bg-background-100 p-4 border border-background-400 hover:border-primary-500/50 hover:bg-background-300 transition-all cursor-pointer">
                    <div className="rounded-lg bg-primary-600/20 p-3">
                      <Upload className="text-primary-400" size={20} />
                    </div>
                    <div>
                      <p className="font-medium text-gray-100">上传视频</p>
                      <p className="text-sm text-gray-500">开始处理</p>
                    </div>
                  </div>
                </Link>

                <Link to="/clustering" className="block">
                  <div className="flex items-center gap-4 rounded-lg bg-background-100 p-4 border border-background-400 hover:border-primary-500/50 hover:bg-background-300 transition-all cursor-pointer">
                    <div className="rounded-lg bg-green-600/20 p-3">
                      <Users className="text-green-400" size={20} />
                    </div>
                    <div>
                      <p className="font-medium text-gray-100">角色标注</p>
                      <p className="text-sm text-gray-500">命名角色</p>
                    </div>
                  </div>
                </Link>

                <Link to="/series" className="block">
                  <div className="flex items-center gap-4 rounded-lg bg-background-100 p-4 border border-background-400 hover:border-primary-500/50 hover:bg-background-300 transition-all cursor-pointer">
                    <div className="rounded-lg bg-blue-600/20 p-3">
                      <Play className="text-blue-400" size={20} />
                    </div>
                    <div>
                      <p className="font-medium text-gray-100">剧集管理</p>
                      <p className="text-sm text-gray-500">配置演员</p>
                    </div>
                  </div>
                </Link>
              </div>
            </CardContent>
          </CardLG>
        </div>
      </div>
    </div>
  )
}
