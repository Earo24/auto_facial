import { Card, CardLG, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Users, Clock, TrendingUp, Award } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'

const characterData = [
  { name: '角色A', samples: 156, screenTime: 245.5, avgQuality: 0.92, color: '#3b82f6' },
  { name: '角色B', samples: 124, screenTime: 198.3, avgQuality: 0.88, color: '#10b981' },
  { name: '角色C', samples: 98, screenTime: 156.7, avgQuality: 0.85, color: '#f59e0b' },
  { name: '角色D', samples: 87, screenTime: 134.2, avgQuality: 0.90, color: '#ef4444' },
  { name: '角色E', samples: 45, screenTime: 67.8, avgQuality: 0.82, color: '#8b5cf6' },
]

export default function Analysis() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">分析报告</h1>
        <p className="text-gray-400">角色统计和出镜分析</p>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-blue-600/20 p-2">
                <Users className="text-blue-400" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">{characterData.length}</p>
                <p className="text-xs text-gray-500">角色总数</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-green-600/20 p-2">
                <Award className="text-success" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">510</p>
                <p className="text-xs text-gray-500">总样本数</p>
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
                <p className="text-2xl font-bold text-gray-100">14.2分</p>
                <p className="text-xs text-gray-500">总出镜时长</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-purple-600/20 p-2">
                <TrendingUp className="text-purple-400" size={20} />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-100">0.87</p>
                <p className="text-xs text-gray-500">平均质量</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Character Details */}
      <CardLG>
        <CardHeader>
          <CardTitle>角色详情</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {characterData.map((char) => (
              <div
                key={char.name}
                className="flex items-center justify-between p-4 rounded-lg bg-background-100 border border-background-400"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="h-10 w-10 rounded-full flex items-center justify-center text-white font-semibold"
                    style={{ backgroundColor: char.color }}
                  >
                    {char.name[0]}
                  </div>
                  <div>
                    <p className="font-medium text-gray-100">{char.name}</p>
                    <p className="text-xs text-gray-500">{char.samples} 个样本</p>
                  </div>
                </div>

                <div className="text-right">
                  <p className="font-medium text-gray-100">{char.screenTime.toFixed(1)}秒</p>
                  <Badge variant={char.avgQuality >= 0.85 ? 'success' : 'warning'} className="text-xs">
                    {(char.avgQuality * 100).toFixed(0)}% 质量
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </CardLG>
    </div>
  )
}
