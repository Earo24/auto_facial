import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Users,
  Image as ImageIcon,
  Merge,
  Split,
  Trash2,
  Search,
  Filter,
  AlertCircle,
  RefreshCw,
} from 'lucide-react'
import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { api, Cluster, VideoInfo } from '@/services/api'

export default function Clustering() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const videoId = searchParams.get('video')

  const [clusters, setClusters] = useState<Cluster[]>([])
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null)
  const [editingCluster, setEditingCluster] = useState<Cluster | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [mergeMode, setMergeMode] = useState(false)
  const [mergeSource, setMergeSource] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentVideo, setCurrentVideo] = useState<VideoInfo | null>(null)
  const [clusteringInProgress, setClusteringInProgress] = useState(false)

  // 加载当前视频信息和聚类数据
  useEffect(() => {
    if (videoId) {
      loadClusters(videoId)
      loadCurrentVideo(videoId)
    } else {
      // 如果没有指定视频，加载第一个可用视频
      loadFirstVideo()
    }
  }, [videoId])

  const loadFirstVideo = async () => {
    try {
      const videos = await api.getVideos()
      if (videos.length > 0) {
        const firstVideo = videos[0]
        setCurrentVideo(firstVideo)
        // 更新 URL 参数
        navigate(`/clustering?video=${firstVideo.video_id}`, { replace: true })
        await loadClusters(firstVideo.video_id)
      } else {
        setLoading(false)
      }
    } catch (error) {
      console.error('加载视频列表失败:', error)
      setLoading(false)
    }
  }

  const loadCurrentVideo = async (vid: string) => {
    try {
      const videos = await api.getVideos()
      const video = videos.find(v => v.video_id === vid)
      if (video) {
        setCurrentVideo(video)
      }
    } catch (error) {
      console.error('加载视频信息失败:', error)
    }
  }

  const loadClusters = async (vid: string) => {
    try {
      setLoading(true)
      const data = await api.getClusters(vid)
      setClusters(data.clusters || [])
    } catch (error) {
      console.error('加载聚类数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleClusterFaces = async () => {
    if (!currentVideo) return
    try {
      setClusteringInProgress(true)
      await api.clusterFaces(currentVideo.video_id, 2)
      // 重新加载聚类数据
      await loadClusters(currentVideo.video_id)
    } catch (error) {
      console.error('聚类失败:', error)
    } finally {
      setClusteringInProgress(false)
    }
  }

  const handleNameSave = () => {
    if (editingCluster) {
      setClusters((prev) =>
        prev.map((c) =>
          c.cluster_id === editingCluster.cluster_id ? { ...c, name: editingCluster.name } : c
        )
      )
      setEditingCluster(null)
    }
  }

  const handleRemoveSample = (clusterId: number, sampleId: string) => {
    setClusters((prev) =>
      prev.map((c) =>
        c.cluster_id === clusterId
          ? { ...c, samples: c.samples.filter((s) => s.sample_id !== sampleId) }
          : c
      )
    )
  }

  // 根据搜索查询过滤簇
  const filteredClusters = clusters.filter(cluster =>
    searchQuery === '' ||
    cluster.cluster_id.toString().includes(searchQuery) ||
    (cluster.name && cluster.name.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="mx-auto h-8 w-8 text-primary-400 animate-spin mb-4" />
          <p className="text-gray-400">加载聚类数据中...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">聚类标注</h1>
          <p className="text-gray-400">
            {currentVideo ? `当前视频: ${currentVideo.filename}` : '审核自动聚类结果，为角色命名'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentVideo && clusters.length === 0 && (
            <Button
              variant="primary"
              onClick={handleClusterFaces}
              disabled={clusteringInProgress}
              className="gap-2"
            >
              <RefreshCw size={18} className={clusteringInProgress ? 'animate-spin' : ''} />
              {clusteringInProgress ? '聚类中...' : '开始聚类'}
            </Button>
          )}
          {mergeMode ? (
            <>
              <Button variant="ghost" onClick={() => { setMergeMode(false); setMergeSource(null) }}>
                取消
              </Button>
              <Button variant="accent" disabled={mergeSource === null}>
                合并选定簇
              </Button>
            </>
          ) : (
            <Button variant="primary" onClick={() => setMergeMode(true)}>
              合并簇模式
            </Button>
          )}
        </div>
      </div>

      {/* Search */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <Input
            type="search"
            placeholder="搜索簇..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Filter size={16} />
          <span>{filteredClusters.length} 个簇</span>
        </div>
      </div>

      {/* No Data State */}
      {clusters.length === 0 && currentVideo && (
        <CardLG>
          <CardContent className="p-12 text-center">
            <AlertCircle className="mx-auto h-16 w-16 text-gray-600 mb-4" />
            <p className="text-lg font-medium text-gray-300 mb-2">暂无聚类数据</p>
            <p className="text-sm text-gray-500 mb-4">
              该视频尚未进行人脸聚类，点击上方按钮开始聚类
            </p>
            <Button variant="primary" onClick={handleClusterFaces} disabled={clusteringInProgress}>
              开始聚类
            </Button>
          </CardContent>
        </CardLG>
      )}

      {/* Main Content */}
      {clusters.length > 0 && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Cluster List */}
          <div className="lg:col-span-1">
            <CardLG>
              <CardHeader>
                <CardTitle>簇列表</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {filteredClusters.map((cluster) => (
                    <div
                      key={cluster.cluster_id}
                      onClick={() => mergeMode ? setMergeSource(cluster.cluster_id) : setSelectedCluster(cluster)}
                      className={cn(
                        'p-3 rounded-lg border cursor-pointer transition-all',
                        mergeMode && mergeSource === cluster.cluster_id
                          ? 'border-accent bg-accent/20'
                          : mergeMode
                          ? 'border-background-500 hover:border-background-400 hover:bg-background-300'
                          : 'border-background-500 hover:border-primary-500/50 hover:bg-background-300',
                        selectedCluster?.cluster_id === cluster.cluster_id && !mergeMode && 'border-primary-500 bg-primary-600/10'
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-background-300 overflow-hidden">
                            {cluster.samples && cluster.samples.length > 0 ? (
                              <img
                                src={cluster.samples[0].image_path}
                                alt=""
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <Users size={18} className="text-gray-400" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium text-gray-100">
                              {cluster.name || `簇 ${cluster.cluster_id + 1}`}
                            </p>
                            <p className="text-xs text-gray-500">
                              {cluster.sample_count} 样本 · {cluster.avg_quality?.toFixed(2) || '0.00'} 平均质量
                            </p>
                          </div>
                        </div>
                        {mergeMode && (
                          <div
                            className={cn(
                              'h-5 w-5 rounded-full border-2',
                              mergeSource === cluster.cluster_id
                                ? 'border-accent bg-accent/20'
                                : 'border-background-500'
                            )}
                          />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </CardLG>
          </div>

          {/* Cluster Detail */}
          <div className="lg:col-span-2">
            {selectedCluster ? (
              <CardLG>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    {editingCluster?.cluster_id === selectedCluster.cluster_id ? (
                      <div className="flex items-center gap-2 flex-1">
                        <Input
                          value={editingCluster.name || `簇 ${selectedCluster.cluster_id + 1}`}
                          onChange={(e) =>
                            setEditingCluster({ ...editingCluster, name: e.target.value })
                          }
                          className="flex-1"
                        />
                        <Button variant="ghost" size="sm" onClick={() => setEditingCluster(null)}>
                          ✕
                        </Button>
                        <Button variant="primary" size="sm" onClick={handleNameSave}>
                          ✓
                        </Button>
                      </div>
                    ) : (
                      <>
                        <div>
                          <CardTitle>{selectedCluster.name || `簇 ${selectedCluster.cluster_id + 1}`}</CardTitle>
                          <p className="text-sm text-gray-500 mt-1">
                            {selectedCluster.first_appearance?.toFixed(1) || '0'}s - {selectedCluster.last_appearance?.toFixed(1) || '0'}s
                          </p>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => setEditingCluster(selectedCluster)}>
                          编辑名称
                        </Button>
                      </>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-gray-100">{selectedCluster.sample_count}</p>
                      <p className="text-sm text-gray-500">样本数</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-gray-100">
                        {selectedCluster.avg_quality?.toFixed(2) || '0.00'}
                      </p>
                      <p className="text-sm text-gray-500">平均质量</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-gray-100">
                        {((selectedCluster.last_appearance || 0) - (selectedCluster.first_appearance || 0)).toFixed(1)}s
                      </p>
                      <p className="text-sm text-gray-500">出现时长</p>
                    </div>
                  </div>

                  {/* Face Grid */}
                  {selectedCluster.samples && selectedCluster.samples.length > 0 ? (
                    <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-3">
                      {selectedCluster.samples.map((sample) => (
                        <div
                          key={sample.sample_id}
                          className="group relative aspect-square rounded-lg overflow-hidden bg-background-100 border border-background-400"
                        >
                          <img
                            src={sample.image_path}
                            alt={sample.sample_id}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect fill="%23374151" width="100" height="100"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF"%3EN/A%3C/text%3E%3C/svg%3E'
                            }}
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                            <div className="absolute bottom-0 left-0 right-0 p-2">
                              <p className="text-xs text-white">质量: {sample.quality_score?.toFixed(2) || '0.00'}</p>
                              <p className="text-xs text-gray-300">{sample.timestamp?.toFixed(1) || '0'}s</p>
                            </div>
                          </div>
                          <button
                            onClick={() => handleRemoveSample(selectedCluster.cluster_id, sample.sample_id)}
                            className="absolute top-1 right-1 p-1 rounded bg-error/80 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-error"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      暂无样本数据
                    </div>
                  )}

                  {/* Actions */}
                  <div className="mt-6 flex items-center gap-2">
                    <Button variant="outline" size="sm" className="gap-2">
                      <Split size={16} />
                      拆分簇
                    </Button>
                    <Button variant="outline" size="sm" className="gap-2">
                      <Trash2 size={16} />
                      删除簇
                    </Button>
                  </div>
                </CardContent>
              </CardLG>
            ) : (
              <CardLG>
                <CardContent className="p-12 text-center">
                  <Users className="mx-auto h-16 w-16 text-gray-600 mb-4" />
                  <p className="text-lg font-medium text-gray-300 mb-2">未选择簇</p>
                  <p className="text-sm text-gray-500">从列表中选择一个簇查看详情</p>
                </CardContent>
              </CardLG>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
