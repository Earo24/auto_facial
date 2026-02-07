import { useState, useCallback, useEffect } from 'react'
import { Upload, FileVideo, X, Check, AlertCircle, Clapperboard } from 'lucide-react'
import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Progress } from '@/components/ui/Progress'
import { Badge } from '@/components/ui/Badge'
import { api, ProcessingStatus, Series } from '@/services/api'

interface UploadFile {
  id: string
  name: string
  size: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  error?: string
  videoId?: string
}

const processingSteps = [
  { key: 'upload', label: '上传视频' },
  { key: 'extract', label: '提取帧' },
  { key: 'detect', label: '检测人脸' },
  { key: 'embed', label: '提取特征' },
  { key: 'cluster', label: '聚类分析' },
  { key: 'complete', label: '完成' },
]

export default function Processing() {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedSeries, setSelectedSeries] = useState<string>('')
  const [series, setSeries] = useState<Series[]>([])
  const [loadingSeries, setLoadingSeries] = useState(false)

  // 加载剧集列表
  useEffect(() => {
    loadSeries()
  }, [])

  const loadSeries = async () => {
    setLoadingSeries(true)
    try {
      const data = await api.getSeries()
      setSeries(data.series || [])
    } catch (error) {
      console.error('加载剧集列表失败:', error)
    } finally {
      setLoadingSeries(false)
    }
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files)
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files)
    }
  }

  const handleFiles = async (fileList: FileList) => {
    for (const file of Array.from(fileList)) {
      if (file.type.startsWith('video/')) {
        const newFile: UploadFile = {
          id: Math.random().toString(36).substring(7),
          name: file.name,
          size: file.size,
          status: 'uploading',
          progress: 0,
        }
        setFiles((prev) => [...prev, newFile])

        try {
          const result = await api.uploadVideo(file, selectedSeries || undefined)
          newFile.videoId = result.video_id
          newFile.status = 'processing'

          // 轮询状态
          pollStatus(result.video_id, newFile.id)
        } catch (error) {
          newFile.status = 'error'
          newFile.error = '上传失败'
          setFiles((prev) => [...prev])
        }
      }
    }
  }

  const pollStatus = async (videoId: string, fileId: string) => {
    const interval = setInterval(async () => {
      try {
        const status: ProcessingStatus = await api.getVideoStatus(videoId)

        setFiles((prev) => prev.map((f) =>
          f.id === fileId ? { ...f, progress: status.progress, status: status.status, message: status.message } : f
        ))

        if (status.status === 'completed') {
          clearInterval(interval)
        } else if (status.status === 'error') {
          clearInterval(interval)
        }
      } catch (error) {
        clearInterval(interval)
      }
    }, 2000)
  }

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-100">视频处理</h1>
        <p className="text-gray-400">上传视频进行自动人脸检测和聚类分析</p>
      </div>

      <div className="max-w-5xl space-y-6">
        {/* 剧集选择 */}
        <Card>
          <CardContent className="p-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <Clapperboard size={16} className="text-primary-400" />
                选择剧集（可选）
              </label>
              <select
                value={selectedSeries}
                onChange={(e) => setSelectedSeries(e.target.value)}
                className="w-full bg-background-500 border border-background-500 rounded-lg px-4 py-3 text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">不关联剧集</option>
                {loadingSeries ? (
                  <option disabled>加载中...</option>
                ) : series.length === 0 ? (
                  <option disabled>暂无剧集</option>
                ) : (
                  series.map((s) => (
                    <option key={s.series_id} value={s.series_id}>
                      {s.name} {s.year ? `(${s.year})` : ''}
                    </option>
                  ))
                )}
              </select>
              <p className="text-xs text-gray-500">
                {selectedSeries
                  ? `已选择: ${series.find(s => s.series_id === selectedSeries)?.name}`
                  : '可在"剧集管理"页面创建新剧集'}
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Upload Area */}
        <CardLG>
          <CardContent className="p-6">
            <div
              className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
                dragActive
                  ? 'border-primary-500 bg-primary-600/10'
                  : 'border-background-500 hover:border-primary-500/50 hover:bg-background-300'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="video-upload"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                accept="video/*"
                multiple
                onChange={handleChange}
              />
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-background-300 p-4">
                  <Upload className="text-primary-400" size={32} />
                </div>
                <div>
                  <p className="text-lg font-medium text-gray-100">
                    拖拽视频到此处，或点击上传
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    支持 MP4、AVI、MOV、MKV 格式（最大2GB）
                  </p>
                </div>
                <Button variant="primary" size="md">
                  浏览文件
                </Button>
              </div>
            </div>
          </CardContent>
        </CardLG>

        {/* Processing Steps */}
        {files.some((f) => f.status === 'processing') && (
          <Card>
            <CardContent className="p-6">
              <h3 className="text-sm font-medium text-gray-300 mb-4">处理步骤</h3>
              <div className="flex items-center justify-between">
                {processingSteps.map((step, index) => (
                  <div key={step.key} className="flex items-center flex-1">
                    <div className="flex flex-col items-center flex-1">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                          index <= currentStep
                            ? 'bg-primary-600 text-white'
                            : 'bg-background-300 text-gray-500'
                        }`}
                      >
                        {index < currentStep ? <Check size={16} /> : index + 1}
                      </div>
                      <p
                        className={`text-xs mt-2 text-center ${
                          index <= currentStep ? 'text-gray-300' : 'text-gray-600'
                        }`}
                      >
                        {step.label}
                      </p>
                    </div>
                    {index < processingSteps.length - 1 && (
                      <div
                        className={`h-0.5 flex-1 mx-2 ${
                          index < currentStep ? 'bg-primary-600' : 'bg-background-500'
                        }`}
                      />
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* File List */}
        {files.length > 0 && (
          <CardLG>
            <CardHeader>
              <CardTitle>上传队列</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="flex items-center gap-4 rounded-lg bg-background-100 p-4 border border-background-400"
                  >
                    {/* Icon */}
                    <div className="rounded-lg bg-background-300 p-3">
                      <FileVideo className="text-gray-400" size={24} />
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="font-medium text-gray-100 truncate">{file.name}</p>
                        {file.status === 'uploading' && <Badge variant="warning">上传中</Badge>}
                        {file.status === 'processing' && <Badge variant="primary">处理中</Badge>}
                        {file.status === 'completed' && <Badge variant="success">已完成</Badge>}
                        {file.status === 'error' && <Badge variant="error">错误</Badge>}
                      </div>
                      <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>

                      {/* Progress */}
                      {(file.status === 'uploading' || file.status === 'processing') && (
                        <div className="mt-2">
                          <Progress value={file.progress} showLabel={false} />
                          <p className="text-xs text-gray-500 mt-1">{file.progress.toFixed(0)}% - {file.message || '处理中...'}</p>
                        </div>
                      )}

                      {file.error && (
                        <p className="text-sm text-error mt-1 flex items-center gap-1">
                          <AlertCircle size={14} />
                          {file.error}
                        </p>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                      {file.status === 'completed' && file.videoId && (
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => window.location.href = `/clustering?video=${file.videoId}`}
                        >
                          查看结果
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(file.id)}
                      >
                        <X size={18} />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </CardLG>
        )}

        {/* Configuration */}
        <CardLG>
          <CardHeader>
            <CardTitle>处理配置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">采样帧率 (fps)</label>
                <select className="input" defaultValue="1">
                  <option value="0.5">0.5 fps（快速，精度较低）</option>
                  <option value="1">1 fps（平衡）</option>
                  <option value="2">2 fps（慢速，精度较高）</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">检测置信度</label>
                <select className="input" defaultValue="0.9">
                  <option value="0.8">80%（严格）</option>
                  <option value="0.9">90%（推荐）</option>
                  <option value="0.95">95%（非常严格）</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">最小人脸尺寸（像素）</label>
                <select className="input" defaultValue="40">
                  <option value="30">30px（检测更多人脸）</option>
                  <option value="40">40px（平衡）</option>
                  <option value="50">50px（更少误检）</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">质量阈值</label>
                <select className="input" defaultValue="0.6">
                  <option value="0.5">0.5（包含低质量）</option>
                  <option value="0.6">0.6（平衡）</option>
                  <option value="0.7">0.7（仅高质量）</option>
                </select>
              </div>
            </div>
          </CardContent>
        </CardLG>
      </div>
    </div>
  )
}
