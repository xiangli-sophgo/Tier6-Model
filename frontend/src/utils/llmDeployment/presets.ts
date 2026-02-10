/**
 * LLM 部署分析系统 - 预设配置
 *
 * 包含常见模型和硬件的预设配置
 */

import {
  LLMModelConfig,
  ChipHardwareConfig,
  BoardConfig,
  RackConfig,
  PodConfig,
  HardwareConfig,
} from './types';
import { modelPresetToLLMConfig } from './configAdapters';
import { getChipPresets as backendGetChipPresets } from '../../api/math_model';
import type { ModelPreset } from '../../types/math_model';

// ============================================
// 模型预设（全部从后端获取）
// ============================================

/** 获取模型列表 */
export function getModelList(): Array<{ id: string; name: string; params: string }> {
  const models = Object.entries(backendModelPresetsCache).map(([id, preset]) => {
    const hasParamsInName = /\d+\.?\d*[BMK]/.test(preset.name);
    return {
      id,
      name: preset.name,
      params: hasParamsInName ? '' : estimateModelParamsFromPreset(preset),
    };
  });
  return models;
}

/** 从 ModelPreset 估算模型参数量 */
function estimateModelParamsFromPreset(preset: ModelPreset): string {
  const H = preset.hidden_size;
  const L = preset.num_layers;
  const V = preset.vocab_size;
  const I = preset.intermediate_size;

  const embedding = V * H;
  const attention = 4 * H * H * L;
  const layerNorm = 2 * H * L;

  let total: number;

  if (preset.MoE) {
    const E = preset.MoE.num_routed_experts;
    const S = preset.MoE.num_shared_experts || 0;
    const expertI = preset.MoE.intermediate_size;
    const ffn = 3 * H * expertI * L * (E + S);
    const router = E * H * L;
    total = embedding + attention + ffn + layerNorm + router;
  } else {
    const ffn = 3 * H * I * L;
    total = embedding + attention + ffn + layerNorm;
  }

  const billions = total / 1e9;
  if (billions >= 1) {
    return `${billions.toFixed(1)}B`;
  } else {
    return `${(total / 1e6).toFixed(0)}M`;
  }
}

// ============================================
// 芯片预设（全部从后端获取）
// ============================================

/** 默认芯片 ID */
export const DEFAULT_CHIP_ID = 'sg2260e';

// ============================================
// 后端模型预设管理
// ============================================

/** 后端模型预设缓存 (ModelPreset 格式) */
let backendModelPresetsCache: Record<string, ModelPreset> = {};
let backendModelPresetsLoaded = false;

/** 从后端加载模型预设 (math_model API 格式) */
export async function loadBackendModelPresets(): Promise<void> {
  if (backendModelPresetsLoaded) return;

  try {
    const response = await fetch('/api/presets/models');
    if (!response.ok) {
      console.warn('Backend model presets load failed');
      return;
    }
    const data = await response.json();
    // math_model 返回: { presets: [{ name: string, config: ModelPreset }] }
    const presets = data.presets || [];

    backendModelPresetsCache = {};
    for (const preset of presets) {
      backendModelPresetsCache[preset.name] = preset.config;
    }
    backendModelPresetsLoaded = true;
  } catch (error) {
    console.warn('Backend model presets load failed:', error);
  }
}

/** 强制重新加载模型预设 */
export async function reloadBackendModelPresets(): Promise<void> {
  backendModelPresetsLoaded = false;
  await loadBackendModelPresets();
}

/** 获取后端模型预设 (LLMModelConfig 格式, 向后兼容) */
export function getBackendModelPresets(): Record<string, LLMModelConfig> {
  const result: Record<string, LLMModelConfig> = {};
  for (const [id, preset] of Object.entries(backendModelPresetsCache)) {
    result[id] = modelPresetToLLMConfig(preset);
  }
  return result;
}

/** 获取后端模型预设 (ModelPreset 原始格式) */
export function getBackendModelPresetsRaw(): Record<string, ModelPreset> {
  return backendModelPresetsCache;
}

// ============================================
// 后端芯片预设管理
// ============================================

/** 后端芯片预设缓存 */
let backendChipPresetsCache: Record<string, ChipHardwareConfig> = {};
let backendPresetsLoaded = false;

/** 后端芯片互联配置缓存 */
let backendChipInterconnectCache: Record<string, ChipInterconnectConfig> = {};

/** 从后端加载芯片预设 */
export async function loadBackendChipPresets(): Promise<void> {
  if (backendPresetsLoaded) return;

  try {
    const data = await backendGetChipPresets();
    const presets = data.presets || [];

    // 直接使用 ChipPreset 格式
    backendChipPresetsCache = {};
    backendChipInterconnectCache = {};
    for (const preset of presets) {
      // 芯片配置直接使用后端格式
      backendChipPresetsCache[preset.name] = {
        ...preset.config,
        name: preset.name,
      };
      // 互联配置暂时使用默认值（c2c 在拓扑层配置）
      backendChipInterconnectCache[preset.name] = {
        interconnect_type: preset.name,
        intra_board_bandwidth_gbps: 448,  // 默认 c2c 带宽
        intra_board_latency_us: 0.2,  // 默认延迟
        recommended_chips_per_board: 8,
      };
    }
    backendPresetsLoaded = true;
  } catch (error) {
    console.warn('后端芯片预设加载失败:', error);
  }
}

/** 获取后端芯片预设（同步，需先调用 loadBackendChipPresets） */
export function getBackendChipPresets(): Record<string, ChipHardwareConfig> {
  return backendChipPresetsCache;
}

/** 从 ChipPreset 计算 BF16 TFLOPS */
function computeTflopsBf16(config: ChipHardwareConfig): number {
  if (!config.cores?.count) {
    throw new Error(`芯片 '${config.name}' 缺少 'cores.count' 字段`);
  }
  if (!config.cores?.lanes_per_core) {
    throw new Error(`芯片 '${config.name}' 缺少 'cores.lanes_per_core' 字段`);
  }
  if (!config.compute_units?.cube?.mac_per_lane?.BF16) {
    throw new Error(`芯片 '${config.name}' 缺少 'compute_units.cube.mac_per_lane.BF16' 字段`);
  }
  if (!config.frequency_ghz) {
    throw new Error(`芯片 '${config.name}' 缺少 'frequency_ghz' 字段`);
  }
  const cores = config.cores.count;
  const lanes = config.cores.lanes_per_core;
  const macBf16 = config.compute_units.cube.mac_per_lane.BF16;
  const freq = config.frequency_ghz;
  // TFLOPS = cores * lanes * mac * 2 * freq / 1000
  return (cores * lanes * macBf16 * 2 * freq) / 1000;
}

/** 从 ChipPreset 获取 GMEM 容量 */
function getGmemCapacityGb(config: ChipHardwareConfig): number {
  if (!config.memory?.gmem?.capacity_gb) {
    throw new Error(`芯片 '${config.name}' 缺少 'memory.gmem.capacity_gb' 字段`);
  }
  return config.memory.gmem.capacity_gb;
}

/** 获取芯片列表（优先使用后端预设） */
export function getChipList(): Array<{ id: string; name: string; memory: string; compute: string; flops_dtype?: string; isCustom?: boolean; isBackend?: boolean }> {
  // 后端预设优先
  const backend = Object.entries(backendChipPresetsCache).map(([id, config]) => ({
    id,
    name: config.name,
    memory: `${getGmemCapacityGb(config)}GB`,
    compute: `${computeTflopsBf16(config).toFixed(0)} BF16 TFLOPs`,
    flops_dtype: 'BF16',
    isCustom: false,
    isBackend: true,
  }));

  // 后端数据 + 自定义
  const custom = Object.entries(getCustomChipPresets()).map(([id, config]) => ({
    id,
    name: config.name,
    memory: `${getGmemCapacityGb(config)}GB`,
    compute: `${computeTflopsBf16(config).toFixed(0)} BF16 TFLOPs`,
    flops_dtype: 'BF16',
    isCustom: true,
    isBackend: false,
  }));
  return [...backend, ...custom];
}

// ============================================
// 自定义芯片预设管理
// ============================================

const CUSTOM_CHIP_PRESETS_KEY = 'llm_custom_chip_presets';

/** 获取自定义芯片预设 */
export function getCustomChipPresets(): Record<string, ChipHardwareConfig> {
  try {
    const data = localStorage.getItem(CUSTOM_CHIP_PRESETS_KEY);
    return data ? JSON.parse(data) : {};
  } catch {
    return {};
  }
}

/** 保存自定义芯片预设（调用后端 API，保存到 YAML）*/
export async function saveCustomChipPreset(config: ChipHardwareConfig): Promise<void> {
  try {
    // 调用 API 保存芯片预设
    const response = await fetch('/api/presets/chips', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: '未知错误' }));
      throw new Error(error.detail || `保存失败: ${response.statusText}`);
    }

    // 保存成功后重置缓存，下次会重新加载
    backendPresetsLoaded = false;
    await loadBackendChipPresets();
  } catch (error) {
    console.error('保存芯片预设失败:', error);
    throw error;
  }
}

/** 删除自定义芯片预设 */
export function deleteCustomChipPreset(id: string): void {
  const presets = getCustomChipPresets();
  delete presets[id];
  localStorage.setItem(CUSTOM_CHIP_PRESETS_KEY, JSON.stringify(presets));
}

/** 获取芯片配置（后端预设 + 自定义），支持通过 ID 或 name 查找 */
export function getChipConfig(idOrName: string): ChipHardwareConfig | null {
  // 1. 优先使用 ID 精确匹配（向后兼容）
  if (backendChipPresetsCache[idOrName]) {
    return backendChipPresetsCache[idOrName];
  }
  const custom = getCustomChipPresets();
  if (custom[idOrName]) {
    return custom[idOrName];
  }

  // 2. 通过 name 查找
  // 后端预设
  for (const config of Object.values(backendChipPresetsCache)) {
    if (config.name === idOrName) {
      return config;
    }
  }
  // 自定义预设
  for (const config of Object.values(custom)) {
    if (config.name === idOrName) {
      return config;
    }
  }

  return null;
}

// ============================================
// 芯片互联配置映射（从后端获取）
// ============================================

/** 芯片互联配置 */
export interface ChipInterconnectConfig {
  /** 互联类型名称 (如 NVLink 4.0, PCIe 4.0) */
  interconnect_type: string;
  /** Board 内带宽 (GB/s) */
  intra_board_bandwidth_gbps: number;
  /** Board 内延迟 (us) */
  intra_board_latency_us: number;
  /** 推荐的芯片数量/Board */
  recommended_chips_per_board: number;
}

/** 获取芯片互联配置（从后端获取） */
export function getChipInterconnectConfig(chipId: string): ChipInterconnectConfig | null {
  return backendChipInterconnectCache[chipId] || null;
}

// ============================================
// 预设 Board 配置
// ============================================

/** DGX H100 Board (8x H100 NVLink) */
export const DGX_H100_BOARD: BoardConfig = {
  chips_per_board: 8,
  b2b_bandwidth_gbps: 900,  // NVLink 4.0
  b2b_latency_us: 1,
};

/** DGX A100 Board (8x A100 NVLink) */
export const DGX_A100_BOARD: BoardConfig = {
  chips_per_board: 8,
  b2b_bandwidth_gbps: 600,  // NVLink 3.0
  b2b_latency_us: 1,
};

/** 通用 PCIe Board (8x GPU PCIe) */
export const PCIE_8GPU_BOARD: BoardConfig = {
  chips_per_board: 8,
  b2b_bandwidth_gbps: 64,   // PCIe 4.0 x16
  b2b_latency_us: 5,
};

/** 所有预设 Board */
export const BOARD_PRESETS: Record<string, BoardConfig> = {
  'dgx-h100': DGX_H100_BOARD,
  'dgx-a100': DGX_A100_BOARD,
  'pcie-8gpu': PCIE_8GPU_BOARD,
};

// ============================================
// 预设 Rack 和 Pod 配置
// ============================================

/** InfiniBand NDR Rack 配置 */
export const IB_NDR_RACK: RackConfig = {
  boards_per_rack: 4,
  r2r_bandwidth_gbps: 400,  // NDR 400G
  r2r_latency_us: 2,
};

/** InfiniBand HDR Rack 配置 */
export const IB_HDR_RACK: RackConfig = {
  boards_per_rack: 4,
  r2r_bandwidth_gbps: 200,  // HDR 200G
  r2r_latency_us: 2,
};

/** RoCE Rack 配置 */
export const ROCE_RACK: RackConfig = {
  boards_per_rack: 4,
  r2r_bandwidth_gbps: 100,  // 100GbE
  r2r_latency_us: 5,
};

/** 所有预设 Rack */
export const RACK_PRESETS: Record<string, RackConfig> = {
  'ib-ndr': IB_NDR_RACK,
  'ib-hdr': IB_HDR_RACK,
  'roce': ROCE_RACK,
};

/** InfiniBand NDR Pod 配置 */
export const IB_NDR_POD: PodConfig = {
  racks_per_pod: 4,
  p2p_bandwidth_gbps: 400,  // NDR 400G
  p2p_latency_us: 2,
};

/** InfiniBand HDR Pod 配置 */
export const IB_HDR_POD: PodConfig = {
  racks_per_pod: 4,
  p2p_bandwidth_gbps: 200,  // HDR 200G
  p2p_latency_us: 2,
};

/** RoCE Pod 配置 */
export const ROCE_POD: PodConfig = {
  racks_per_pod: 4,
  p2p_bandwidth_gbps: 100,  // 100GbE
  p2p_latency_us: 5,
};

/** 所有预设 Pod */
export const POD_PRESETS: Record<string, PodConfig> = {
  'ib-ndr': IB_NDR_POD,
  'ib-hdr': IB_HDR_POD,
  'roce': ROCE_POD,
};

// ============================================
// 预设完整硬件配置（动态创建，使用后端芯片配置）
// ============================================

/** 所有预设硬件配置（空，使用 createHardwareConfig 动态创建） */
export const HARDWARE_PRESETS: Record<string, HardwareConfig> = {};

// ============================================
// 推理配置和 Benchmark 预设（已移除，从后端配置文件加载）
// ============================================

// ============================================
// 辅助函数
// ============================================

/** 获取模型预设 (LLMModelConfig 格式, 向后兼容) */
export function getModelPreset(modelId: string): LLMModelConfig {
  const preset = backendModelPresetsCache[modelId];
  if (!preset) {
    throw new Error(`Model preset not found: ${modelId}`);
  }
  return modelPresetToLLMConfig(preset);
}

/** 获取模型预设 (ModelPreset 原始格式) */
export function getModelPresetRaw(modelId: string): ModelPreset {
  const preset = backendModelPresetsCache[modelId];
  if (!preset) {
    throw new Error(`Model preset not found: ${modelId}`);
  }
  return { ...preset };
}

/** 获取芯片预设 */
export function getChipPreset(chipId: string): ChipHardwareConfig {
  const preset = getChipConfig(chipId);
  if (!preset) {
    throw new Error(`未找到芯片预设: ${chipId}`);
  }
  return { ...preset };
}

/** 获取硬件预设 */
export function getHardwarePreset(hardwareId: string): HardwareConfig {
  const preset = HARDWARE_PRESETS[hardwareId];
  if (!preset) {
    throw new Error(`未找到硬件预设: ${hardwareId}`);
  }

  // 从预设中提取互联参数
  const board = (preset as any).board as BoardConfig | undefined;
  const rack = (preset as any).rack as RackConfig | undefined;
  const pod = (preset as any).pod as PodConfig | undefined;
  const chip = (preset as any).chip as ChipHardwareConfig | undefined;

  // 构建互联配置 - 从预设中提取，如果找不到则抛出错误
  if (!board) {
    throw new Error(`硬件预设 '${hardwareId}' 缺少 board 配置`);
  }
  if (!rack) {
    throw new Error(`硬件预设 '${hardwareId}' 缺少 rack 配置`);
  }
  if (!pod) {
    throw new Error(`硬件预设 '${hardwareId}' 缺少 pod 配置`);
  }

  // c2c: 从芯片名称查找互联配置，如果找不到则抛出错误
  let c2cBandwidth: number;
  let c2cLatency: number;
  if (chip?.name) {
    const chipInterconnect = backendChipInterconnectCache[chip.name];
    if (!chipInterconnect) {
      throw new Error(`硬件预设 '${hardwareId}' 的芯片 '${chip.name}' 缺少互联配置`);
    }
    c2cBandwidth = chipInterconnect.intra_board_bandwidth_gbps;
    c2cLatency = chipInterconnect.intra_board_latency_us;
  } else {
    throw new Error(`硬件预设 '${hardwareId}' 缺少芯片配置或芯片名称`);
  }

  return {
    chips: {},
    interconnect: {
      links: {
        c2c: { bandwidth_gbps: c2cBandwidth, latency_us: c2cLatency },
        b2b: { bandwidth_gbps: board.b2b_bandwidth_gbps, latency_us: board.b2b_latency_us },
        r2r: { bandwidth_gbps: rack.r2r_bandwidth_gbps, latency_us: rack.r2r_latency_us },
        p2p: { bandwidth_gbps: pod.p2p_bandwidth_gbps, latency_us: pod.p2p_latency_us },
      }
    },
    chip: { ...chip },
    board: { ...board },
    rack: { ...rack },
    pod: { ...pod },
  };
}


/** 创建自定义硬件配置 */
export function createHardwareConfig(
  chipId: string,
  boardId: string,
  rackId: string,
  podId: string
): HardwareConfig {
  const chip = getChipPreset(chipId);

  // 禁止使用默认值 - 如果预设不存在则抛出错误
  const board = BOARD_PRESETS[boardId];
  if (!board) {
    throw new Error(`未找到 Board 预设: ${boardId}`);
  }

  const rack = RACK_PRESETS[rackId];
  if (!rack) {
    throw new Error(`未找到 Rack 预设: ${rackId}`);
  }

  const pod = POD_PRESETS[podId];
  if (!pod) {
    throw new Error(`未找到 Pod 预设: ${podId}`);
  }

  // c2c: 从芯片互联配置获取，如果找不到则抛出错误
  const chipInterconnect = backendChipInterconnectCache[chipId] || (chip.name ? backendChipInterconnectCache[chip.name] : null);
  if (!chipInterconnect) {
    throw new Error(`芯片 '${chipId}' 缺少互联配置 (c2c)。请确保已加载芯片预设。`);
  }

  return {
    chips: {},
    interconnect: {
      links: {
        c2c: {
          bandwidth_gbps: chipInterconnect.intra_board_bandwidth_gbps,
          latency_us: chipInterconnect.intra_board_latency_us
        },
        b2b: {
          bandwidth_gbps: board.b2b_bandwidth_gbps,
          latency_us: board.b2b_latency_us
        },
        r2r: {
          bandwidth_gbps: rack.r2r_bandwidth_gbps,
          latency_us: rack.r2r_latency_us
        },
        p2p: {
          bandwidth_gbps: pod.p2p_bandwidth_gbps,
          latency_us: pod.p2p_latency_us
        },
      }
    },
    chip,
    board: { ...board },
    rack: { ...rack },
    pod: { ...pod },
  };
}

// ============================================
// 后端预设加载辅助函数
// ============================================

import {
  getBenchmarks,
  getBenchmark,
  getTopologies,
  getTopology,
} from '@/api/math_model';

import type {
  BenchmarkConfig,
  TopologyConfig,
} from '@/types/math_model';

/**
 * 加载 Benchmark 预设列表
 */
export async function loadBenchmarkPresets(): Promise<BenchmarkConfig[]> {
  const { benchmarks } = await getBenchmarks();
  // 逐个加载完整配置
  const fullConfigs = await Promise.all(
    benchmarks.map(b => getBenchmark(b.id))
  );
  return fullConfigs;
}

/**
 * 加载 Topology 预设列表
 */
export async function loadTopologyPresets(): Promise<TopologyConfig[]> {
  const { topologies } = await getTopologies();
  // 逐个加载完整配置
  const fullConfigs = await Promise.all(
    topologies.map(t => getTopology(t.name))
  );
  return fullConfigs;
}
