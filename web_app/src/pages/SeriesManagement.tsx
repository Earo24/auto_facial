import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Users, Film } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';

interface Series {
  series_id: string;
  name: string;
  year?: number;
  description?: string;
  poster_path?: string;
  created_at: string;
}

interface SeriesActor {
  id: number;
  series_id: string;
  actor_id: string;
  character_name: string;
  actor_name?: string;
  photo_path?: string;
  role_order: number;
  is_main_character: boolean;
}

interface SeriesVideo {
  video_id: string;
  filename: string;
  duration: number;
  detected_faces: number;
  processed_frames: number;
}

export default function SeriesManagement() {
  const [series, setSeries] = useState<Series[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<Series | null>(null);
  const [actors, setActors] = useState<SeriesActor[]>([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showActorDialog, setShowActorDialog] = useState(false);
  const [loading, setLoading] = useState(false);
  const [seriesVideos, setSeriesVideos] = useState<Record<string, SeriesVideo[]>>({});

  // 新增电视剧表单
  const [newSeries, setNewSeries] = useState({
    name: '',
    year: new Date().getFullYear(),
    description: ''
  });

  // 新增演员表单
  const [newActor, setNewActor] = useState({
    actor_name: '',
    character_name: '',
    photo_file: null as File | null
  });

  useEffect(() => {
    loadSeries();
  }, []);

  const loadSeries = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/series');
      const data = await res.json();
      setSeries(data.series || []);

      // 加载所有视频
      const videosRes = await fetch('http://localhost:8000/api/videos');
      const videosData = await videosRes.json();

      // 按剧集分组视频
      const videosBySeries: Record<string, SeriesVideo[]> = {};
      for (const video of videosData) {
        if (video.series_id) {
          if (!videosBySeries[video.series_id]) {
            videosBySeries[video.series_id] = [];
          }
          videosBySeries[video.series_id].push({
            video_id: video.video_id,
            filename: video.filename,
            duration: video.duration,
            detected_faces: video.detected_faces,
            processed_frames: video.processed_frames,
          });
        }
      }
      setSeriesVideos(videosBySeries);
    } catch (error) {
      console.error('加载电视剧失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadActors = async (seriesId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/series/${seriesId}/actors`);
      const data = await res.json();
      setActors(data.actors || []);
    } catch (error) {
      console.error('加载演员失败:', error);
    }
  };

  const handleCreateSeries = async () => {
    if (!newSeries.name.trim()) return;

    try {
      const res = await fetch('http://localhost:8000/api/series', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSeries)
      });
      if (res.ok) {
        setShowAddDialog(false);
        setNewSeries({ name: '', year: new Date().getFullYear(), description: '' });
        loadSeries();
      }
    } catch (error) {
      console.error('创建电视剧失败:', error);
    }
  };

  const handleCreateActor = async () => {
    if (!newActor.actor_name.trim() || !newActor.character_name.trim()) return;

    try {
      // 先创建演员
      const actorRes = await fetch('http://localhost:8000/api/actors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newActor.actor_name })
      });
      const actorData = await actorRes.json();
      if (!actorData.success) return;

      // 添加到电视剧
      const addRes = await fetch(`http://localhost:8000/api/series/${selectedSeries?.series_id}/actors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actor_id: actorData.actor_id,
          character_name: newActor.character_name
        })
      });

      // 上传照片
      if (newActor.photo_file && addRes.ok) {
        const formData = new FormData();
        formData.append('file', newActor.photo_file);
        await fetch(`http://localhost:8000/api/actors/${actorData.actor_id}/photo`, {
          method: 'POST',
          body: formData
        });
      }

      setShowActorDialog(false);
      setNewActor({ actor_name: '', character_name: '', photo_file: null });
      if (selectedSeries) {
        loadActors(selectedSeries.series_id);
      }
    } catch (error) {
      console.error('添加演员失败:', error);
    }
  };

  const handleSelectSeries = (s: Series) => {
    setSelectedSeries(s);
    loadActors(s.series_id);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] p-6">
      {/* 页面标题 */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">剧集管理</h1>
          <p className="text-gray-400 mt-1">配置电视剧和演员角色信息</p>
        </div>
        <Button
          onClick={() => setShowAddDialog(true)}
          className="bg-primary-600 hover:bg-primary-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors"
        >
          + 新增剧集
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 剧集列表 */}
        <div className="lg:col-span-1">
          <h2 className="text-lg font-semibold text-gray-100 mb-4">剧集列表</h2>
          <div className="space-y-3">
            {loading ? (
              <div className="text-center py-8 text-gray-500">加载中...</div>
            ) : series.length === 0 ? (
              <Card className="p-8 text-center text-gray-500">
                暂无剧集，点击上方"新增剧集"开始配置
              </Card>
            ) : (
              series.map(s => {
                const videos = seriesVideos[s.series_id] || [];
                return (
                  <Card
                    key={s.series_id}
                    onClick={() => handleSelectSeries(s)}
                    className={`p-4 cursor-pointer transition-all duration-200 hover:bg-background-300 hover:shadow-lg ${
                      selectedSeries?.series_id === s.series_id
                        ? 'bg-primary-600/20 border-2 border-primary-500'
                        : 'border border-background-500'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className={`font-semibold ${selectedSeries?.series_id === s.series_id ? 'text-white' : 'text-gray-100'}`}>
                          {s.name}
                        </h3>
                        <p className={`text-sm mt-1 ${selectedSeries?.series_id === s.series_id ? 'text-purple-200' : 'text-gray-400'}`}>
                          {s.year}年
                        </p>
                        {s.description && (
                          <p className={`text-sm mt-2 line-clamp-2 ${selectedSeries?.series_id === s.series_id ? 'text-purple-200' : 'text-gray-500'}`}>
                            {s.description}
                          </p>
                        )}
                      </div>
                      {videos.length > 0 && (
                        <div className={`flex items-center gap-1 px-2 py-1 rounded-lg ${selectedSeries?.series_id === s.series_id ? 'bg-primary-600/30' : 'bg-background-300'}`}>
                          <Film size={14} className={selectedSeries?.series_id === s.series_id ? 'text-primary-300' : 'text-gray-400'} />
                          <span className={`text-sm font-medium ${selectedSeries?.series_id === s.series_id ? 'text-primary-300' : 'text-gray-400'}`}>
                            {videos.length}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* 显示关联的视频 */}
                    {videos.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-background-500/50">
                        <p className={`text-xs mb-2 ${selectedSeries?.series_id === s.series_id ? 'text-purple-200' : 'text-gray-500'}`}>关联视频:</p>
                        <div className="space-y-1.5">
                          {videos.slice(0, 3).map(video => (
                            <div key={video.video_id} className="flex items-center gap-2 text-xs">
                              <Film size={12} className={selectedSeries?.series_id === s.series_id ? 'text-purple-300' : 'text-gray-500'} />
                              <span className={selectedSeries?.series_id === s.series_id ? 'text-purple-100' : 'text-gray-400'}>
                                {video.filename}
                              </span>
                              {video.detected_faces > 0 && (
                                <span className={`ml-auto px-1.5 py-0.5 rounded text-xs ${
                                  selectedSeries?.series_id === s.series_id
                                    ? 'bg-green-600/30 text-green-300'
                                    : 'bg-green-600/20 text-green-400'
                                }`}>
                                  {video.detected_faces}人脸
                                </span>
                              )}
                            </div>
                          ))}
                          {videos.length > 3 && (
                            <p className={`text-xs ${selectedSeries?.series_id === s.series_id ? 'text-purple-200' : 'text-gray-500'}`}>
                              还有 {videos.length - 3} 个视频...
                            </p>
                          )}
                        </div>
                      </div>
                    )}

                    <div className={`mt-3 pt-3 border-t border-background-500/50 flex gap-2 ${videos.length > 0 ? '' : 'border-t-0'}`}>
                      <Link to={`/series/${s.series_id}/characters`}>
                        <Button variant="ghost" size="sm" className="gap-1">
                          <Users size={16} />
                          查看角色
                        </Button>
                      </Link>
                    </div>
                  </Card>
                );
              })
            )}
          </div>
        </div>

        {/* 演员配置 */}
        <div className="lg:col-span-2">
          {selectedSeries ? (
            <Card className="p-6">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-xl font-bold text-gray-100">{selectedSeries.name}</h2>
                  <p className="text-sm text-gray-400 mt-1">演员配置</p>
                </div>
                <Button
                  onClick={() => setShowActorDialog(true)}
                  className="bg-primary-600 hover:bg-primary-700 text-white px-5 py-2 rounded-lg font-medium transition-colors"
                >
                  + 添加演员
                </Button>
              </div>

              {actors.length === 0 ? (
                <div className="text-center py-16 text-gray-500">
                  <p className="text-lg mb-2">暂无演员配置</p>
                  <p className="text-sm">点击上方"添加演员"按钮开始配置</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {actors.map(actor => (
                    <Card key={actor.id} className="p-4 border border-background-500 hover:border-primary-500/50 hover:shadow-md transition-all duration-200">
                      <div className="flex items-center space-x-4">
                        {actor.photo_path ? (
                          <img
                            src={`http://localhost:8000/api/actor_photos/${actor.actor_id}.jpg`}
                            alt={actor.actor_name}
                            className="w-16 h-16 rounded-full object-cover border-2 border-background-500"
                          />
                        ) : (
                          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-600/20 to-primary-400/20 flex items-center justify-center">
                            <span className="text-primary-400 font-semibold text-lg">
                              {actor.actor_name?.charAt(0) || '?'}
                            </span>
                          </div>
                        )}
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-100">{actor.actor_name}</h4>
                          <p className="text-sm text-gray-400">饰演: {actor.character_name}</p>
                          <span className={`inline-block mt-2 px-2 py-1 text-xs font-medium rounded-full ${
                            actor.is_main_character
                              ? 'bg-primary-600/20 text-primary-400'
                              : 'bg-background-300 text-gray-400'
                          }`}>
                            {actor.is_main_character ? '主演' : '配角'}
                          </span>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </Card>
          ) : (
            <Card className="p-16 text-center">
              <div className="max-w-md mx-auto">
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary-600/20 to-primary-400/20 flex items-center justify-center">
                  <svg className="w-10 h-10 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m1 0l4-4m1 0l4 4M4 12h8" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-100 mb-2">选择剧集查看演员配置</h3>
                <p className="text-gray-400">从左侧列表中选择一个剧集，查看和管理演员角色信息</p>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* 新增剧集对话框 */}
      {showAddDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md p-6">
            <h3 className="text-xl font-bold text-gray-100 mb-6">新增剧集</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  剧名 <span className="text-red-400">*</span>
                </label>
                <Input
                  value={newSeries.name}
                  onChange={e => setNewSeries({...newSeries, name: e.target.value})}
                  placeholder="例如: 太平年"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  年份
                </label>
                <Input
                  type="number"
                  value={newSeries.year}
                  onChange={e => setNewSeries({...newSeries, year: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  简介
                </label>
                <textarea
                  className="w-full bg-background-500 border border-background-500 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  rows={3}
                  value={newSeries.description}
                  onChange={e => setNewSeries({...newSeries, description: e.target.value})}
                  placeholder="剧集简介..."
                />
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <Button
                onClick={() => setShowAddDialog(false)}
                className="px-5 py-2.5 border border-background-500 text-gray-300 rounded-lg hover:bg-background-500 transition-colors"
              >
                取消
              </Button>
              <Button
                onClick={handleCreateSeries}
                disabled={!newSeries.name.trim()}
                className="px-5 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                创建
              </Button>
            </div>
          </Card>
        </div>
      )}

      {/* 添加演员对话框 */}
      {showActorDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md p-6">
            <h3 className="text-xl font-bold text-gray-100 mb-6">添加演员</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  演员名 <span className="text-red-400">*</span>
                </label>
                <Input
                  value={newActor.actor_name}
                  onChange={e => setNewActor({...newActor, actor_name: e.target.value})}
                  placeholder="例如: 刘钧"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  角色名 <span className="text-red-400">*</span>
                </label>
                <Input
                  value={newActor.character_name}
                  onChange={e => setNewActor({...newActor, character_name: e.target.value})}
                  placeholder="例如: 乾隆"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  演员照片
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={e => setNewActor({...newActor, photo_file: e.target.files?.[0] || null})}
                  className="w-full bg-background-500 border border-background-500 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary-500 focus:border-transparent file:mr-4 file:rounded-lg"
                />
                {newActor.photo_file && (
                  <p className="text-sm text-gray-400 mt-1">已选择: {newActor.photo_file.name}</p>
                )}
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <Button
                onClick={() => setShowActorDialog(false)}
                className="px-5 py-2.5 border border-background-500 text-gray-300 rounded-lg hover:bg-background-500 transition-colors"
              >
                取消
              </Button>
              <Button
                onClick={handleCreateActor}
                disabled={!newActor.actor_name.trim() || !newActor.character_name.trim()}
                className="px-5 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                添加
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
