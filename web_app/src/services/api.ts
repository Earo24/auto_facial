/**
 * API 服务 - 与 Python 后端通信
 */

const API_BASE_URL = 'http://localhost:8000/api';

export interface VideoInfo {
  video_id: string;
  filename: string;
  duration: number;
  processed_frames: number;
  detected_faces: number;
  characters_found: number;
  series_id?: string;
  series_name?: string;
  series_year?: number;
  created_at: string;
}

export interface ProcessingStatus {
  video_id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  message: string;
  result?: {
    processed_frames: number;
    detected_faces: number;
  };
}

export interface FaceSample {
  sample_id: string;
  frame_id: string;
  timestamp: number;
  quality_score: number;
  image_path: string;
  cluster_id: number | null;
  character_id: string | null;
}

export interface Cluster {
  cluster_id: number;
  sample_count: number;
  avg_quality: number;
  first_appearance: number;
  last_appearance: number;
  samples: FaceSample[];
}

export interface RecognitionResult {
  sample_id: string;
  frame_id: string;
  timestamp: number;
  character_id: string | null;
  character_name: string | null;
  confidence: number;
  bbox: string;
}

export interface Character {
  character_id: string;
  name: string;
  video_id: string;
  prototypes: Array<{
    embedding: number[];
    image_path: string;
    quality_score: number;
    timestamp: number;
  }>;
}

export interface Series {
  series_id: string;
  name: string;
  year?: number;
  description?: string;
  poster_path?: string;
  created_at: string;
}

// API 函数
export const api = {
  // 健康检查
  async healthCheck(): Promise<{ status: string; models_loaded: boolean }> {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.json();
  },

  // 视频相关
  async getVideos(): Promise<VideoInfo[]> {
    const res = await fetch(`${API_BASE_URL}/videos`);
    const data = await res.json();
    // API直接返回数组
    return data || [];
  },

  async uploadVideo(file: File, seriesId?: string): Promise<{ video_id: string; status: string }> {
    const formData = new FormData();
    formData.append('file', file);
    if (seriesId) {
      formData.append('series_id', seriesId);
    }

    const res = await fetch(`${API_BASE_URL}/videos/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data;
  },

  async getVideoStatus(videoId: string): Promise<ProcessingStatus> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/status`);
    return res.json();
  },

  // 聚类相关
  async clusterFaces(videoId: string, minClusterSize: number = 5): Promise<{ clusters: Cluster[]; total: number }> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/cluster?min_cluster_size=${minClusterSize}`, {
      method: 'POST',
    });
    return res.json();
  },

  async getClusters(videoId: string): Promise<{ clusters: Cluster[] }> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/clusters`);
    return res.json();
  },

  async nameCluster(clusterId: number, videoId: string, name: string): Promise<{ success: boolean; character_id: string; name: string }> {
    const res = await fetch(`${API_BASE_URL}/clusters/${clusterId}/name`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, name }),
    });
    return res.json();
  },

  async mergeClusters(videoId: string, sourceClusterId: number, targetClusterId: number): Promise<{
    success: boolean;
    message?: string;
    merged_count?: number;
    target_cluster_id?: number;
    error?: string;
  }> {
    const res = await fetch(`${API_BASE_URL}/clusters/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: videoId,
        source_cluster_id: sourceClusterId,
        target_cluster_id: targetClusterId,
      }),
    });
    return res.json();
  },

  // 识别相关
  async recognizeVideo(videoId: string, useTemporalSmoothing: boolean = true): Promise<{
    video_id: string;
    total_samples: number;
    recognized: number;
    results: RecognitionResult[];
  }> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/recognize?use_temporal_smoothing=${useTemporalSmoothing}`, {
      method: 'POST',
    });
    return res.json();
  },

  async getRecognitionResults(videoId: string): Promise<RecognitionResult[]> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/recognition`);
    return res.json();
  },

  // 分析相关
  async getAnalysis(videoId: string): Promise<{
    video_id: string;
    characters: any[];
    recognition_stats: any[];
  }> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/analysis`);
    return res.json();
  },

  // 样本相关
  async getSamples(videoId: string, clusterId?: number, characterId?: string): Promise<{ samples: FaceSample[]; total: number }> {
    let url = `${API_BASE_URL}/videos/${videoId}/samples`;
    const params = new URLSearchParams();
    if (clusterId !== undefined) params.append('cluster_id', clusterId.toString());
    if (characterId) params.append('character_id', characterId);
    if (params.toString()) url += '?' + params.toString();

    const res = await fetch(url);
    return res.json();
  },

  async removeSample(videoId: string, sampleId: string): Promise<{
    success: boolean;
    message?: string;
    error?: string;
  }> {
    const res = await fetch(`${API_BASE_URL}/samples/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, sample_id: sampleId }),
    });
    return res.json();
  },

  // 角色相关
  async getCharacters(videoId: string): Promise<{ characters: Character[] }> {
    const res = await fetch(`${API_BASE_URL}/videos/${videoId}/characters`);
    return res.json();
  },

  // 剧集相关
  async getSeries(): Promise<{ series: Series[] }> {
    const res = await fetch(`${API_BASE_URL}/series`);
    return res.json();
  },

  async matchActors(seriesId: string, videoId: string): Promise<{
    success: boolean;
    matched_clusters: number;
    total_clusters: number;
    results: Array<{
      cluster_id: number;
      sample_count: number;
      actor_id: string | null;
      actor_name: string | null;
      character_name: string | null;
      similarity: number;
    }>;
    error?: string;
  }> {
    const res = await fetch(`${API_BASE_URL}/series/${seriesId}/recluster`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId }),
    });
    return res.json();
  },
};
