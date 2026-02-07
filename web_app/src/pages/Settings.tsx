import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'

export default function Settings() {
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">系统设置</h1>
        <p className="text-gray-400">配置系统参数</p>
      </div>

      <CardLG>
        <CardHeader>
          <CardTitle>基本设置</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">默认采样帧率</label>
            <Select defaultValue="1">
              <option value="0.5">0.5 fps</option>
              <option value="1">1 fps</option>
              <option value="2">2 fps</option>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">视频导出格式</label>
            <Select defaultValue="mp4">
              <option value="mp4">MP4</option>
              <option value="webm">WebM</option>
              <option value="avi">AVI</option>
            </Select>
          </div>
        </CardContent>
      </CardLG>

      <CardLG>
        <CardHeader>
          <CardTitle>人脸检测</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">检测置信度</label>
            <Input type="number" step="0.05" min="0" max="1" defaultValue="0.9" />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">最小人脸尺寸（像素）</label>
            <Input type="number" min="20" max="200" defaultValue="40" />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">质量阈值</label>
            <Input type="number" step="0.1" min="0" max="1" defaultValue="0.6" />
          </div>
        </CardContent>
      </CardLG>

      <CardLG>
        <CardHeader>
          <CardTitle>聚类算法</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">DBSCAN 距离阈值</label>
            <Input type="number" step="0.1" min="0.1" max="2" defaultValue="0.5" />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">最小簇大小</label>
            <Input type="number" min="1" max="50" defaultValue="5" />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">合并相似度阈值</label>
            <Input type="number" step="0.05" min="0" max="1" defaultValue="0.85" />
          </div>
        </CardContent>
      </CardLG>

      <div className="flex justify-end gap-3">
        <Button variant="ghost">恢复默认</Button>
        <Button variant="primary">保存设置</Button>
      </div>
    </div>
  )
}
