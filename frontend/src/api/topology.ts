/**
 * 拓扑 API 模块
 *
 * 拓扑生成使用本地生成器，配置存储使用后端 API
 */

import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ConnectionConfig,
  GlobalSwitchConfig,
  ManualConnectionConfig,
} from '../types';

import {
  topologyGenerator,
  TopologyGenerateRequest,
} from '../utils/topologyGenerator';

import {
  getChipTypes as storageGetChipTypes,
  getRackDimensions as storageGetRackDimensions,
  getLevelConnectionDefaults as storageGetLevelConnectionDefaults,
  SavedConfig,
  NetworkConfig,
  SavedChipConfig,
} from '../utils/storage';

// ============================================
// 拓扑生成 API（本地生成）
// ============================================

/**
 * 获取完整拓扑数据
 */
export async function getTopology(): Promise<HierarchicalTopology> {
  return topologyGenerator.getCachedTopology();
}

/**
 * 生成新的拓扑
 */
export async function generateTopology(config: {
  pod_count?: number;
  racks_per_pod?: number;
  rack_config?: {
    total_u: number;
    boards: Array<{
      id: string;
      name: string;
      u_height: number;
      count: number;
      chips: Array<{ name: string; count: number }>;
    }>;
  };
  switch_config?: GlobalSwitchConfig;
  manual_connections?: ManualConnectionConfig;
  interconnect_config?: {
    c2c?: { bandwidth_gbps: number; latency_us: number };
    b2b?: { bandwidth_gbps: number; latency_us: number };
    r2r?: { bandwidth_gbps: number; latency_us: number };
    p2p?: { bandwidth_gbps: number; latency_us: number };
  };
}): Promise<HierarchicalTopology> {
  return topologyGenerator.generate(config as TopologyGenerateRequest);
}

/**
 * 获取指定Pod
 */
export async function getPod(podId: string): Promise<PodConfig> {
  const pod = topologyGenerator.getPod(podId);
  if (!pod) {
    throw new Error(`Pod '${podId}' not found`);
  }
  return pod;
}

/**
 * 获取指定Rack
 */
export async function getRack(rackId: string): Promise<RackConfig> {
  const rack = topologyGenerator.getRack(rackId);
  if (!rack) {
    throw new Error(`Rack '${rackId}' not found`);
  }
  return rack;
}

/**
 * 获取指定Board
 */
export async function getBoard(boardId: string): Promise<BoardConfig> {
  const board = topologyGenerator.getBoard(boardId);
  if (!board) {
    throw new Error(`Board '${boardId}' not found`);
  }
  return board;
}

/**
 * 获取连接数据
 */
export async function getConnections(
  level?: string,
  parentId?: string
): Promise<ConnectionConfig[]> {
  if (level) {
    return topologyGenerator.getConnectionsForLevel(level, parentId);
  }
  return topologyGenerator.getCachedTopology().connections;
}

// ============================================
// 静态配置接口
// ============================================

/**
 * 获取Chip类型配置
 */
export async function getChipTypes(): Promise<{
  types: { id: string; name: string; color: string }[];
}> {
  return storageGetChipTypes();
}

/**
 * 获取Rack尺寸配置
 */
export async function getRackDimensions(): Promise<{
  width: number;
  depth: number;
  u_height: number;
  total_u: number;
  full_height: number;
}> {
  return storageGetRackDimensions();
}

/**
 * 获取各层级连接的默认带宽和延迟配置
 */
export async function getLevelConnectionDefaults(): Promise<{
  datacenter: { bandwidth: number; latency: number };
  pod: { bandwidth: number; latency: number };
  rack: { bandwidth: number; latency: number };
  board: { bandwidth: number; latency: number };
}> {
  return storageGetLevelConnectionDefaults();
}

// ============================================
// 拓扑配置保存/加载 API（后端存储）
// ============================================

export type { SavedConfig, NetworkConfig, SavedChipConfig };

/**
 * 获取所有保存的拓扑配置列表
 */
export async function listConfigs(): Promise<SavedConfig[]> {
  try {
    const response = await fetch('/api/topologies');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.topologies || [];
  } catch (error) {
    console.error('获取拓扑配置列表失败:', error);
    return [];
  }
}

/**
 * 获取指定名称的拓扑配置
 */
export async function getConfig(name: string): Promise<SavedConfig> {
  const response = await fetch(`/api/topologies/${encodeURIComponent(name)}`);
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`配置 '${name}' 不存在`);
    }
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}

/**
 * 保存拓扑配置（创建或更新）
 */
export async function saveConfig(config: SavedConfig): Promise<SavedConfig> {
  // 先尝试 PUT（更新），如果返回 404 则 POST（创建）
  let response = await fetch(`/api/topologies/${encodeURIComponent(config.name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

  if (response.status === 404) {
    // 配置不存在，创建新的
    response = await fetch('/api/topologies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '未知错误' }));
    throw new Error(error.detail || `保存失败: ${response.status}`);
  }

  return config;
}

/**
 * 删除拓扑配置
 */
export async function deleteConfig(name: string): Promise<void> {
  const response = await fetch(`/api/topologies/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`配置 '${name}' 不存在`);
    }
    throw new Error(`HTTP error! status: ${response.status}`);
  }
}

// ============================================
// 后端预设 API
// ============================================

/** 后端芯片预设 */
export interface BackendChipPreset {
  id: string;
  name: string;
  flops_dtype: string;
  compute_tflops: number;
  num_cores: number;
  sram_size_mb: number;
  dram_bandwidth_gbps: number;
  intra_bw_gbps: number;
  inter_bw_gbps: number;
}

/**
 * 从后端获取芯片预设列表
 */
export async function getBackendChipPresets(): Promise<BackendChipPreset[]> {
  try {
    const response = await fetch('/api/presets/chips');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    // tier6 返回 { presets: [...] } 格式
    return data.presets?.map((p: any) => ({
      id: p.name,
      name: p.config?.name || p.name,
      flops_dtype: 'BF16',
      compute_tflops: p.config?.compute_units?.cube?.mac_per_lane?.BF16 || 0,
      num_cores: p.config?.cores?.count || 0,
      sram_size_mb: (p.config?.memory?.lmem?.capacity_mb || 0) / 1024,
      dram_bandwidth_gbps: p.config?.memory?.gmem?.bandwidth_gbps || 0,
      intra_bw_gbps: 448,
      inter_bw_gbps: 448,
    })) || [];
  } catch (error) {
    console.error('获取后端芯片预设失败:', error);
    return [];
  }
}

// ============================================
// Benchmark 管理 API
// ============================================

/** Benchmark 配置类型 */
export interface BenchmarkConfig {
  id: string;
  name: string;
  model: Record<string, unknown>;
  inference: Record<string, unknown>;
}

/**
 * 获取所有自定义 Benchmark 列表（含完整配置）
 */
export async function listBenchmarks(): Promise<BenchmarkConfig[]> {
  try {
    const response = await fetch('/api/benchmarks');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    const summaries = data.benchmarks || [];

    // tier6 列表只返回摘要，需要获取每个 benchmark 的详情
    const benchmarks = await Promise.all(
      summaries.map(async (summary: any) => {
        // 使用 filename 获取详情（tier6 用 filename 作为查询 key）
        const id = summary.filename || summary.id;
        const detail = await getBenchmark(id);
        return detail;
      })
    );

    return benchmarks.filter((b): b is BenchmarkConfig => b !== null);
  } catch (error) {
    console.error('获取 Benchmark 列表失败:', error);
    return [];
  }
}

/**
 * 转换 Tier6 Benchmark 格式为前端期望格式
 */
function convertTier6Benchmark(data: any): BenchmarkConfig {
  const model = data.model || {};
  return {
    id: data.id,
    name: data.name,
    model: {
      ...model,
      // tier6 用 name，前端用 model_name
      model_name: model.model_name || model.name || data.name,
      // 根据是否有 moe 配置判断模型类型
      model_type: model.model_type || (model.moe ? 'moe' : 'dense'),
      // 兼容字段映射
      hidden_size: model.hidden_size,
      num_layers: model.num_layers || model.num_dense_layers,
      num_attention_heads: model.num_attention_heads || model.num_heads,
      num_key_value_heads: model.num_key_value_heads || model.num_heads,
      intermediate_size: model.intermediate_size || model.ffn?.intermediate_size,
      vocab_size: model.vocab_size,
      // MoE 配置转换
      moe_config: model.moe ? {
        num_experts: model.moe.num_routed_experts,
        num_shared_experts: model.moe.num_shared_experts,
        experts_per_token: model.moe.num_activated_experts,
        expert_intermediate_size: model.moe.intermediate_size,
      } : undefined,
      // MLA 配置转换
      mla_config: model.mla ? {
        q_lora_rank: model.mla.q_lora_rank,
        kv_lora_rank: model.mla.kv_lora_rank,
        qk_nope_head_dim: model.mla.qk_nope_head_dim,
        qk_rope_head_dim: model.mla.qk_rope_head_dim,
        v_head_dim: model.mla.v_head_dim,
      } : undefined,
    },
    inference: data.inference || {},
  };
}

/**
 * 获取单个 Benchmark 配置
 */
export async function getBenchmark(id: string): Promise<BenchmarkConfig | null> {
  try {
    const response = await fetch(`/api/benchmarks/${id}`);
    if (!response.ok) {
      if (response.status === 404) return null;
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return convertTier6Benchmark(data);
  } catch (error) {
    console.error('获取 Benchmark 失败:', error);
    return null;
  }
}

/**
 * 创建新的 Benchmark 配置
 */
export async function createBenchmark(benchmark: BenchmarkConfig): Promise<boolean> {
  try {
    const response = await fetch('/api/benchmarks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(benchmark),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return true;
  } catch (error) {
    console.error('创建 Benchmark 失败:', error);
    return false;
  }
}

/**
 * 更新 Benchmark 配置
 */
export async function updateBenchmark(id: string, benchmark: BenchmarkConfig): Promise<boolean> {
  try {
    const response = await fetch(`/api/benchmarks/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(benchmark),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return true;
  } catch (error) {
    console.error('更新 Benchmark 失败:', error);
    return false;
  }
}

/**
 * 删除 Benchmark 配置
 */
export async function deleteBenchmark(id: string): Promise<boolean> {
  try {
    const response = await fetch(`/api/benchmarks/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return true;
  } catch (error) {
    console.error('删除 Benchmark 失败:', error);
    return false;
  }
}
