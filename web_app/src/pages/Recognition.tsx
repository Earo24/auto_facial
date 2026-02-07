import { useState } from 'react'
import { Search, Filter, Download, Eye, Clock, CheckCircle } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent, CardLG } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import { cn } from '@/lib/utils'

export default function Recognition() {
  const [selectedResult, setSelectedResult] = useState<any>(null)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">识别结果</h1>
          <p className="text-gray-400">查看和验证批量识别结果</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-background-300 p-2">
                <Eye className="text-gray-400" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">150</p>
                <p className="text-xs text-gray-500">总检测</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-green-600/20 p-2">
                <CheckCircle className="text-success" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">142</p>
                <p className="text-xs text-gray-500">已识别</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-yellow-600/20 p-2">
                <Clock className="text-warning" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">128</p>
                <p className="text-xs text-gray-500">高置信度</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-blue-600/20 p-2">
                <Download className="text-blue-400" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">导出</p>
                <p className="text-xs text-gray-500">结果数据</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Content */}
      <CardLG>
        <CardHeader>
          <CardTitle>识别结果列表</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center py-8 text-gray-500">
            请先在"聚类标注"页面完成角色命名，然后运行批量识别
          </p>
        </CardContent>
      </CardLG>
    </div>
  )
}
