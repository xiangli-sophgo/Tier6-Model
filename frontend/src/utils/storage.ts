/**
 * 存储工具模块
 *
 * 提供类型定义和静态配置函数
 * 配置数据现已迁移到后端 API 存储
 */

import { ManualConnectionConfig, GlobalSwitchConfig, HierarchicalTopology } from '../types';
import { ChipHardwareConfig } from './llmDeployment/types';

// ============================================
// 类型定义
// ============================================

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
        compute_tflops_fp16?: number;
        memory_gb?: number;
        memory_bandwidth_gbps?: number;
        memory_bandwidth_utilization?: number;
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

  /** 通信延迟配置 (统一所有延迟相关参数) */
  comm_latency_config?: {
    // 协议相关
    rtt_tp_us: number;
    rtt_ep_us: number;
    bandwidth_utilization: number;
    sync_latency_us: number;
    // 网络基础设施
    switch_delay_us: number;
    cable_delay_us: number;
    // 芯片延迟
    memory_read_latency_us: number;
    memory_write_latency_us: number;
    noc_latency_us: number;
    die_to_die_latency_us: number;
  };

  /** 硬件参数配置 (芯片参数 + 互联参数) */
  hardware_params?: {
    // 多芯片独立配置，key = chip.name
    chips: Record<string, {
      name: string;
      num_cores: number;
      compute_tflops_fp8: number;
      compute_tflops_bf16: number;
      memory_capacity_gb: number;
      memory_bandwidth_gbps: number;
      memory_bandwidth_utilization: number;
      lmem_capacity_mb: number;
      lmem_bandwidth_gbps: number;
      cost_per_hour?: number;
      cube_m?: number;
      cube_k?: number;
      cube_n?: number;
      sram_size_kb?: number;
      sram_utilization?: number;
      lane_num?: number;
      align_bytes?: number;
      compute_dma_overlap_rate?: number;
    }>;
    interconnect: {
      c2c: { bandwidth_gbps: number; latency_us: number };
      b2b: { bandwidth_gbps: number; latency_us: number };
      r2r: { bandwidth_gbps: number; latency_us: number };
      p2p: { bandwidth_gbps: number; latency_us: number };
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
    board: { bandwidth: 900.0, latency: 1.0 },
  };
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
  ];
  keysToRemove.forEach(key => localStorage.removeItem(key));
}
