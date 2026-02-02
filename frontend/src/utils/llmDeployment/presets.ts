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

// ============================================
// 模型预设（全部从后端获取）
// ============================================

/** 获取模型列表（优先使用后端预设） */
export function getModelList(): Array<{ id: string; name: string; params: string }> {
  // 后端预设优先
  const models = Object.entries(backendModelPresetsCache).map(([id, config]) => {
    // 如果模型名称中已经包含参数量（如 671B, 70B），就不再显示
    const hasParamsInName = /\d+\.?\d*[BMK]/.test(config.model_name);
    return {
      id,
      name: config.model_name,
      params: hasParamsInName ? '' : estimateModelParams(config),
    };
  });

  return models;
}

/** 估算模型参数量 */
function estimateModelParams(config: LLMModelConfig): string {
  const H = config.hidden_size;
  const L = config.num_layers;
  const V = config.vocab_size;
  const I = config.intermediate_size;

  // 各部分参数量
  const embedding = V * H;
  const attention = 4 * H * H * L;  // Q, K, V, O 投影
  const layerNorm = 2 * H * L;

  let total: number;

  if (config.model_type === 'moe' && config.moe_config) {
    const E = config.moe_config.num_experts;
    const S = config.moe_config.num_shared_experts || 0;
    // 使用 expert_intermediate_size（如有），否则使用 intermediate_size
    const expertI = config.moe_config.expert_intermediate_size || I;
    const ffn = 3 * H * expertI * L * (E + S);
    const router = E * H * L;
    total = embedding + attention + ffn + layerNorm + router;
  } else {
    const ffn = 3 * H * I * L;  // gate, up, down (SwiGLU)
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

/** 后端模型预设缓存 */
let backendModelPresetsCache: Record<string, LLMModelConfig> = {};
let backendModelPresetsLoaded = false;

/** 从后端加载模型预设 */
export async function loadBackendModelPresets(): Promise<void> {
  if (backendModelPresetsLoaded) return;

  try {
    const response = await fetch('/api/presets/models');
    if (!response.ok) {
      console.warn('后端模型预设加载失败，使用本地预设');
      return;
    }
    const data = await response.json();
    const models = data.models || [];

    // 存储到缓存
    backendModelPresetsCache = {};
    for (const model of models) {
      backendModelPresetsCache[model.id] = {
        model_name: model.model_name,
        model_type: model.model_type,
        hidden_size: model.hidden_size,
        num_layers: model.num_layers,
        num_attention_heads: model.num_attention_heads,
        num_kv_heads: model.num_kv_heads,
        intermediate_size: model.intermediate_size,
        vocab_size: model.vocab_size,
        weight_dtype: model.weight_dtype,
        activation_dtype: model.activation_dtype,
        max_seq_length: model.max_seq_length,
        norm_type: model.norm_type || 'rmsnorm',
        attention_type: model.attention_type,
        moe_config: model.moe_config,
        mla_config: model.mla_config,
      };
    }
    backendModelPresetsLoaded = true;
  } catch (error) {
    console.warn('后端模型预设加载失败:', error);
  }
}

/** 获取后端模型预设（同步，需先调用 loadBackendModelPresets） */
export function getBackendModelPresets(): Record<string, LLMModelConfig> {
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
    const response = await fetch('/api/presets/chips');
    if (!response.ok) {
      console.warn('后端芯片预设加载失败，使用本地预设');
      return;
    }
    const data = await response.json();
    const chips = data.chips || [];

    // 转换为 ChipHardwareConfig 格式
    backendChipPresetsCache = {};
    backendChipInterconnectCache = {};
    for (const chip of chips) {
      backendChipPresetsCache[chip.id] = {
        name: chip.name,
        num_cores: chip.num_cores,
        compute_tflops_fp8: chip.compute_tflops * 2,  // FP8 = 2 × BF16/FP16
        compute_tflops_bf16: chip.compute_tflops,  // BF16/FP16
        memory_capacity_gb: chip.name.includes('SG2262') ? 128 : 64,  // SG2262: 128GB, 其他默认 64GB
        memory_bandwidth_gbps: chip.dram_bandwidth_gbps,
        memory_bandwidth_utilization: 0.85,
        lmem_capacity_mb: chip.sram_size_mb || 2,  // 从后端获取 SRAM 大小
        lmem_bandwidth_gbps: 512,  // 默认 512 GB/s LMEM 带宽
        // 微架构参数（SG2260E 默认值，用户可在前端修改）
        cube_m: 16,
        cube_k: 32,
        cube_n: 8,
        sram_size_kb: 2048,
        sram_utilization: 0.45,
        lane_num: 16,
        align_bytes: 32,
        compute_dma_overlap_rate: 0.8,
      };
      // 保存互联配置
      backendChipInterconnectCache[chip.id] = {
        interconnect_type: chip.name,  // 使用芯片名作为互联类型标识
        intra_board_bandwidth_gbps: chip.c2c_bw_unidirectional_gbps,
        intra_board_latency_us: chip.intra_latency_us,
        recommended_chips_per_board: 8,  // 默认值
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

/** 获取芯片列表（优先使用后端预设） */
export function getChipList(): Array<{ id: string; name: string; memory: string; compute: string; flops_dtype?: string; isCustom?: boolean; isBackend?: boolean }> {
  // 后端预设优先
  const backend = Object.entries(backendChipPresetsCache).map(([id, config]) => ({
    id,
    name: config.name,
    memory: `${config.memory_capacity_gb}GB`,
    compute: `${config.compute_tflops_bf16.toFixed(0)} BF16 TFLOPs`,
    flops_dtype: 'BF16',
    isCustom: false,
    isBackend: true,
  }));

  // 后端数据 + 自定义
  const custom = Object.entries(getCustomChipPresets()).map(([id, config]) => ({
    id,
    name: config.name,
    memory: `${config.memory_capacity_gb}GB`,
    compute: `${config.compute_tflops_bf16} BF16 TFLOPs`,
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
export async function saveCustomChipPreset(config: any): Promise<void> {
  try {
    const response = await fetch('/api/chip-presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`保存失败: ${response.statusText}`);
    }

    // 保存成功后重新加载预设
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

/** 获取模型预设（从后端预设获取） */
export function getModelPreset(modelId: string): LLMModelConfig {
  const preset = backendModelPresetsCache[modelId];
  if (!preset) {
    throw new Error(`未找到模型预设: ${modelId}`);
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
  return {
    hardware_params: { chips: {}, interconnect: { c2c: { bandwidth_gbps: 0, latency_us: 0 }, b2b: { bandwidth_gbps: 0, latency_us: 0 }, r2r: { bandwidth_gbps: 0, latency_us: 0 }, p2p: { bandwidth_gbps: 0, latency_us: 0 } } },
    chip: { ...(preset as any).chip },
    board: { ...(preset as any).board },
    rack: { ...(preset as any).rack },
    pod: { ...(preset as any).pod },
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
  const board = BOARD_PRESETS[boardId] ?? DGX_H100_BOARD;
  const rack = RACK_PRESETS[rackId] ?? IB_NDR_RACK;
  const pod = POD_PRESETS[podId] ?? IB_NDR_POD;

  return {
    hardware_params: { chips: {}, interconnect: { c2c: { bandwidth_gbps: 0, latency_us: 0 }, b2b: { bandwidth_gbps: 0, latency_us: 0 }, r2r: { bandwidth_gbps: 0, latency_us: 0 }, p2p: { bandwidth_gbps: 0, latency_us: 0 } } },
    chip,
    board: { ...board },
    rack: { ...rack },
    pod: { ...pod },
  };
}
