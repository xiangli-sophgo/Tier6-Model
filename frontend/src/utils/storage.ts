/**
 * 存储工具模块
 *
 * 提供类型定义和静态配置函数
 * 配置数据现已迁移到后端 API 存储
 */

import { ManualConnectionConfig, GlobalSwitchConfig, HierarchicalTopology } from '../types';
import { ChipHardwareConfig } from './llmDeployment/types';
import { ChipPreset } from '../types/math_model';

// ============================================
// 类型定义
// ============================================

/**
 * 列配置方案
 */
export interface ColumnPreset {
  name: string;
  experiment_id: number;
  visible_columns: string[];
  column_order: string[];
  fixed_columns: string[];
  created_at: string;
}

/**
 * 配置文件结构
 */
export interface PresetsFile {
  version: number;
  presets: ColumnPreset[];
}

/**
 * 网络配置 - 存储带宽和延迟参数
 */
export interface NetworkConfig {
  /** Board 内互联带宽 (GB/s) - 如 NVLink, C2C */
  intra_board_bandwidth_gbps: number;
  /** Board 间互联带宽 (GB/s) - 如 InfiniBand, B2B */
  inter_board_bandwidth_gbps: number;
  /** Board 内延迟 (us) */
  intra_board_latency_us: number;
  /** Board 间延迟 (us) */
  inter_board_latency_us: number;
}

/**
 * 芯片配置项 - 包含预设ID和完整硬件参数
 */
export interface SavedChipConfig {
  /** 预设ID (如 'h100-sxm') */
  preset_id?: string;
  /** 芯片硬件配置 */
  hardware: ChipHardwareConfig;
  /** 芯片总数 */
  total_count: number;
  /** 每个 Board 的芯片数量 */
  chips_per_board: number;
}

/**
 * 保存的拓扑配置接口
 */
export interface SavedConfig {
  name: string;
  description?: string;
  pod_count: number;
  racks_per_pod: number;
  rack_config?: {
    total_u: number;
    boards: Array<{
      id: string;
      name: string;
      u_height: number;
      count: number;
      chips: Array<{
        name: string;
        count: number;
        preset_id?: string;
        // 芯片详细配置现在存储在顶层 chips 字典中（使用 ChipPreset 格式）
      }>;
    }>;
  };
  switch_config?: GlobalSwitchConfig;
  manual_connections?: ManualConnectionConfig;
  created_at?: string;
  updated_at?: string;

  // ============================================
  // 扩展字段 - 用于部署分析
  // ============================================

  /** 生成的完整拓扑数据 */
  generated_topology?: HierarchicalTopology;
  /** 芯片硬件配置列表 (已解析的完整硬件参数) */
  chip_configs?: SavedChipConfig[];
  /** 网络配置 (带宽/延迟参数) */
  network_config?: NetworkConfig;

  /** 芯片配置字典 - 使用 ChipPreset 格式 */
  chips?: Record<string, ChipPreset>;

  /** 互联配置 (层级链路 + 通信参数) */
  interconnect?: {
    /** 层级互联链路 */
    links: {
      c2c: { bandwidth_gbps: number; latency_us: number };
      b2b: { bandwidth_gbps: number; latency_us: number };
      r2r: { bandwidth_gbps: number; latency_us: number };
      p2p: { bandwidth_gbps: number; latency_us: number };
    };
    /** 通信参数 (统一所有延迟相关参数) */
    comm_params?: {
      // 协议相关
      bandwidth_utilization: number;
      sync_latency_us: number;
      // 网络基础设施
      switch_latency_us: number;
      cable_latency_us: number;
      // 芯片延迟
      memory_read_latency_us: number;
      memory_write_latency_us: number;
      noc_latency_us: number;
      die_to_die_latency_us: number;
    };
  };
}

// ============================================
// 静态配置函数
// ============================================

/**
 * 获取支持的Chip类型
 */
export function getChipTypes(): { types: { id: string; name: string; color: string }[] } {
  return {
    types: [
      { id: 'npu', name: 'NPU', color: '#eb2f96' },
      { id: 'cpu', name: 'CPU', color: '#1890ff' },
    ],
  };
}

/**
 * 获取Rack物理尺寸配置
 */
export function getRackDimensions(): {
  width: number;
  depth: number;
  u_height: number;
  total_u: number;
  full_height: number;
} {
  return {
    width: 0.6,
    depth: 1.0,
    u_height: 0.0445,
    total_u: 42,
    full_height: 42 * 0.0445,
  };
}

/**
 * 获取各层级连接的默认带宽和延迟配置
 *
 * 单位说明:
 * - bandwidth: GB/s (字节/秒，非 Gbps)
 * - latency: us (微秒)
 *
 * 参考值来源 (presets.ts):
 * - NVLink 4.0 (H100): 900 GB/s, 1 us
 * - NVLink 3.0 (A100): 600 GB/s, 1 us
 * - InfiniBand NDR: 50 GB/s (400 Gbps), 2 us
 * - PCIe 4.0: 64 GB/s, 5 us
 */
export function getLevelConnectionDefaults(): {
  datacenter: { bandwidth: number; latency: number };
  pod: { bandwidth: number; latency: number };
  rack: { bandwidth: number; latency: number };
  board: { bandwidth: number; latency: number };
} {
  return {
    // Pod 间: 跨 Pod 通信，通常走数据中心网络
    datacenter: { bandwidth: 50.0, latency: 50.0 },
    // Rack 间: 同 Pod 内不同 Rack，通常走 InfiniBand
    pod: { bandwidth: 50.0, latency: 5.0 },
    // Board 间: 同 Rack 内不同节点/Board，走 InfiniBand
    rack: { bandwidth: 50.0, latency: 2.0 },
    // Chip 间: 同 Board 内芯片互联，走 NVLink
    board: { bandwidth: 448.0, latency: 0.2 },
  };
}

// ============================================
// 列配置方案管理 API
// ============================================

/**
 * 获取所有列配置方案
 */
export function getColumnPresets(): PresetsFile {
  try {
    const data = localStorage.getItem('tier6_column_presets');
    if (!data) {
      return { version: 1, presets: [] };
    }
    return JSON.parse(data) as PresetsFile;
  } catch (error) {
    console.error('Failed to load column presets:', error);
    return { version: 1, presets: [] };
  }
}

/**
 * 获取指定实验的列配置方案
 */
export function getColumnPresetsByExperiment(experimentId: number): { presets: ColumnPreset[] } {
  const presetsFile = getColumnPresets();
  const filtered = presetsFile.presets.filter(p => p.experiment_id === experimentId);
  return { presets: filtered };
}

/**
 * 保存所有列配置方案（完全覆盖）
 */
export function saveColumnPresets(presetsFile: PresetsFile): { message: string; count: number } {
  try {
    localStorage.setItem('tier6_column_presets', JSON.stringify(presetsFile));
    return { message: '配置已保存', count: presetsFile.presets.length };
  } catch (error) {
    console.error('Failed to save column presets:', error);
    throw new Error('保存配置失败');
  }
}

/**
 * 添加或更新单个列配置方案
 */
export function addColumnPreset(preset: ColumnPreset): { message: string; preset: ColumnPreset } {
  const presetsFile = getColumnPresets();

  // 检查是否存在同名同实验ID的配置
  const existingIndex = presetsFile.presets.findIndex(
    p => p.name === preset.name && p.experiment_id === preset.experiment_id
  );

  if (existingIndex !== -1) {
    // 更新现有配置
    presetsFile.presets[existingIndex] = preset;
  } else {
    // 添加新配置
    presetsFile.presets.push(preset);
  }

  saveColumnPresets(presetsFile);

  const message = existingIndex !== -1
    ? `配置方案「${preset.name}」已更新`
    : `配置方案「${preset.name}」已保存`;

  return { message, preset };
}

/**
 * 删除列配置方案
 */
export function deleteColumnPreset(experimentId: number, name: string): { message: string } {
  const presetsFile = getColumnPresets();

  const originalCount = presetsFile.presets.length;
  presetsFile.presets = presetsFile.presets.filter(
    p => !(p.name === name && p.experiment_id === experimentId)
  );

  if (presetsFile.presets.length === originalCount) {
    throw new Error(`配置方案「${name}」不存在`);
  }

  saveColumnPresets(presetsFile);
  return { message: `配置方案「${name}」已删除` };
}

// ============================================
// 缓存清理 API
// ============================================

/**
 * 清除本地缓存数据（localStorage）
 */
export async function clearAllCache(): Promise<void> {
  // 清除 localStorage 中的相关缓存
  const keysToRemove = [
    'tier6_topology_config_cache',
    'tier6_sider_width_cache',
    'tier6_column_presets',
  ];
  keysToRemove.forEach(key => localStorage.removeItem(key));
}
