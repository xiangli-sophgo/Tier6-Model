/**
 * LLM 部署分析系统 - 预设配置
 *
 * 包含常见模型和硬件的预设配置
 */

import {
  LLMModelConfig,
  MLAConfig,
  ChipHardwareConfig,
  BoardConfig,
  RackConfig,
  PodConfig,
  HardwareConfig,
  InferenceConfig,
  BenchmarkScenario,
  BenchmarkPreset,
  FlopsDtype,
} from './types';

// ============================================
// 预设模型配置 (只保留 DeepSeek 和 Qwen 最新版本)
// ============================================

/** DeepSeek-V3 MLA 配置 (官方参数) */
const DEEPSEEK_V3_MLA: MLAConfig = {
  kv_lora_rank: 512,       // KV 压缩后的隐维度
  q_lora_rank: 1536,       // Q 的 LoRA rank
  qk_nope_head_dim: 128,   // 非 RoPE 头维度
  qk_rope_head_dim: 64,    // RoPE 头维度
  v_head_dim: 128,         // V 的头维度
};

/** DeepSeek-V3-671B (MoE + MLA) */
export const DEEPSEEK_V3: LLMModelConfig = {
  model_name: 'DeepSeek-V3-671B',
  model_type: 'moe',
  hidden_size: 7168,
  num_layers: 61,
  num_attention_heads: 128,  // MLA
  num_kv_heads: 128,         // MLA (实际使用 kv_lora_rank=512 压缩)
  intermediate_size: 18432,  // Dense层FFN维度 (前3层使用)
  vocab_size: 129280,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'mla',
  moe_config: {
    num_experts: 256,
    num_experts_per_tok: 8,
    expert_capacity_factor: 1.0,
    num_shared_experts: 1,
    expert_intermediate_size: 2048,  // 每个专家的FFN维度
    first_k_dense_replace: 3,        // 前3层使用Dense FFN
    moe_tp: 1,
    ep_tp_strategy: 'scatter_gather',
  },
  mla_config: DEEPSEEK_V3_MLA,
};

/** DeepSeek-R1-671B (MoE + MLA，与 V3 同架构) */
export const DEEPSEEK_R1: LLMModelConfig = {
  model_name: 'DeepSeek-R1-671B',
  model_type: 'moe',
  hidden_size: 7168,
  num_layers: 61,
  num_attention_heads: 128,  // MLA
  num_kv_heads: 128,         // MLA (实际使用 kv_lora_rank=512 压缩)
  intermediate_size: 18432,  // Dense层FFN维度 (前3层使用)
  vocab_size: 129280,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'mla',
  moe_config: {
    num_experts: 256,
    num_experts_per_tok: 8,
    expert_capacity_factor: 1.0,
    num_shared_experts: 1,
    expert_intermediate_size: 2048,  // 每个专家的FFN维度
    first_k_dense_replace: 3,        // 前3层使用Dense FFN
    moe_tp: 1,
    ep_tp_strategy: 'scatter_gather',
  },
  mla_config: DEEPSEEK_V3_MLA,  // 与 V3 相同的 MLA 配置
};

// ============================================
// Qwen 最新模型
// ============================================

/** Qwen2.5-72B (官方配置) */
export const QWEN2_5_72B: LLMModelConfig = {
  model_name: 'Qwen2.5-72B',
  model_type: 'dense',
  hidden_size: 8192,
  num_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,   // GQA
  intermediate_size: 29568,
  vocab_size: 152064,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'gqa',
};

/** Qwen2.5-32B */
export const QWEN2_5_32B: LLMModelConfig = {
  model_name: 'Qwen2.5-32B',
  model_type: 'dense',
  hidden_size: 5120,
  num_layers: 64,
  num_attention_heads: 40,
  num_kv_heads: 8,   // GQA
  intermediate_size: 27648,
  vocab_size: 152064,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'gqa',
};

/** Qwen3-32B */
export const QWEN3_32B: LLMModelConfig = {
  model_name: 'Qwen3-32B',
  model_type: 'dense',
  hidden_size: 5120,
  num_layers: 64,
  num_attention_heads: 64,
  num_kv_heads: 8,   // GQA
  intermediate_size: 25600,  // 估算: ~5H
  vocab_size: 151936,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'gqa',
};

/** Qwen3-235B-A22B (MoE 旗舰)
 * 注意: expert_intermediate_size 已调整为等效值以匹配官方 235B 参数量
 */
export const QWEN3_235B: LLMModelConfig = {
  model_name: 'Qwen3-235B-A22B',
  model_type: 'moe',
  hidden_size: 8192,
  num_layers: 94,
  num_attention_heads: 64,
  num_kv_heads: 8,   // GQA
  intermediate_size: 29568,  // Dense FFN
  vocab_size: 151936,
  weight_dtype: 'bf16',
  activation_dtype: 'bf16',
  max_seq_length: 131072,
  norm_type: 'rmsnorm',
  attention_type: 'gqa',
  moe_config: {
    num_experts: 128,
    num_experts_per_tok: 8,
    expert_capacity_factor: 1.0,
    expert_intermediate_size: 768,  // 等效值，匹配官方 235B
  },
};

/** 所有预设模型 (只保留 DeepSeek 和 Qwen 最新版本) */
export const MODEL_PRESETS: Record<string, LLMModelConfig> = {
  // DeepSeek 最新
  'deepseek-v3': DEEPSEEK_V3,
  'deepseek-r1': DEEPSEEK_R1,
  // Qwen 最新
  'qwen3-235b': QWEN3_235B,
  'qwen3-32b': QWEN3_32B,
  'qwen2.5-72b': QWEN2_5_72B,
  'qwen2.5-32b': QWEN2_5_32B,
};

/** 获取模型列表 */
export function getModelList(): Array<{ id: string; name: string; params: string }> {
  return Object.entries(MODEL_PRESETS).map(([id, config]) => {
    // 如果模型名称中已经包含参数量（如 671B, 70B），就不再显示
    const hasParamsInName = /\d+\.?\d*[BMK]/.test(config.model_name);
    return {
      id,
      name: config.model_name,
      params: hasParamsInName ? '' : estimateModelParams(config),
    };
  });
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
        chip_type: chip.name,
        num_cores: chip.num_cores,
        compute_tflops_fp8: chip.compute_tflops * 2,  // FP8 = 2 × BF16/FP16
        compute_tflops_bf16: chip.compute_tflops,  // BF16/FP16
        memory_capacity_gb: chip.name.includes('SG2262') ? 128 : 64,  // SG2262: 128GB, 其他默认 64GB
        memory_bandwidth_gbps: chip.dram_bandwidth_gbps,
        memory_bandwidth_utilization: 0.85,
        lmem_capacity_mb: chip.sram_size_mb || 2,  // 从后端获取 SRAM 大小
        lmem_bandwidth_gbps: 512,  // 默认 512 GB/s LMEM 带宽
        c2c_bandwidth_gbps: chip.c2c_bw_unidirectional_gbps,  // C2C 带宽
        c2c_latency_us: chip.intra_latency_us,  // C2C 延迟
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
    name: config.chip_type,
    memory: `${config.memory_capacity_gb}GB`,
    compute: `${config.compute_tflops_bf16.toFixed(0)} BF16 TFLOPs`,
    flops_dtype: 'BF16',
    isCustom: false,
    isBackend: true,
  }));

  // 后端数据 + 自定义
  const custom = Object.entries(getCustomChipPresets()).map(([id, config]) => ({
    id,
    name: config.chip_type,
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

/** 保存自定义芯片预设 */
export function saveCustomChipPreset(id: string, config: ChipHardwareConfig): void {
  const presets = getCustomChipPresets();
  presets[id] = config;
  localStorage.setItem(CUSTOM_CHIP_PRESETS_KEY, JSON.stringify(presets));
}

/** 删除自定义芯片预设 */
export function deleteCustomChipPreset(id: string): void {
  const presets = getCustomChipPresets();
  delete presets[id];
  localStorage.setItem(CUSTOM_CHIP_PRESETS_KEY, JSON.stringify(presets));
}

/** 获取芯片配置（后端预设 + 自定义） */
export function getChipConfig(id: string): ChipHardwareConfig | null {
  // 优先使用后端预设
  if (backendChipPresetsCache[id]) {
    return backendChipPresetsCache[id];
  }
  // 检查自定义
  const custom = getCustomChipPresets();
  return custom[id] || null;
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
// 预设推理配置
// ============================================

/** 低延迟交互场景 */
export const INFERENCE_LOW_LATENCY: InferenceConfig = {
  batch_size: 1,
  input_seq_length: 128,
  output_seq_length: 128,
  max_seq_length: 256,
  num_micro_batches: 1,
};

/** 标准推理场景 */
export const INFERENCE_STANDARD: InferenceConfig = {
  batch_size: 8,
  input_seq_length: 512,
  output_seq_length: 256,
  max_seq_length: 768,
  num_micro_batches: 4,
};

/** 高吞吐批处理场景 */
export const INFERENCE_HIGH_THROUGHPUT: InferenceConfig = {
  batch_size: 32,
  input_seq_length: 512,
  output_seq_length: 256,
  max_seq_length: 768,
  num_micro_batches: 8,
};

/** 长上下文场景 */
export const INFERENCE_LONG_CONTEXT: InferenceConfig = {
  batch_size: 1,
  input_seq_length: 32768,
  output_seq_length: 1024,
  max_seq_length: 33792,
  num_micro_batches: 1,
};

/** 代码生成场景 */
export const INFERENCE_CODE_GEN: InferenceConfig = {
  batch_size: 4,
  input_seq_length: 2048,
  output_seq_length: 2048,
  max_seq_length: 4096,
  num_micro_batches: 2,
};

/** 所有预设推理配置 */
export const INFERENCE_PRESETS: Record<string, InferenceConfig> = {
  'low-latency': INFERENCE_LOW_LATENCY,
  'standard': INFERENCE_STANDARD,
  'high-throughput': INFERENCE_HIGH_THROUGHPUT,
  'long-context': INFERENCE_LONG_CONTEXT,
  'code-gen': INFERENCE_CODE_GEN,
};

// ============================================
// Benchmark 预设
// ============================================

/** 标准推理 Benchmark 场景 */
export const BENCHMARK_SCENARIOS: BenchmarkScenario[] = [
  {
    name: '低延迟交互',
    description: '单请求低延迟场景，优化首token延迟',
    inference_config: INFERENCE_LOW_LATENCY,
    optimization_target: 'latency',
    success_criteria: {
      max_ttft_ms: 100,
      max_tpot_ms: 50,
    },
  },
  {
    name: '高吞吐批处理',
    description: '批量请求高吞吐场景，最大化token生成速度',
    inference_config: INFERENCE_HIGH_THROUGHPUT,
    optimization_target: 'throughput',
    success_criteria: {
      min_throughput: 1000,
    },
  },
  {
    name: '长上下文处理',
    description: '32K上下文场景，验证显存效率',
    inference_config: INFERENCE_LONG_CONTEXT,
    optimization_target: 'balanced',
    success_criteria: {
      max_ttft_ms: 5000,
    },
  },
  {
    name: '代码生成',
    description: '中等负载代码生成场景，平衡延迟和输出长度',
    inference_config: INFERENCE_CODE_GEN,
    optimization_target: 'balanced',
    success_criteria: {
      max_ttft_ms: 500,
    },
  },
];

/** Benchmark 预设 */
export const BENCHMARK_PRESETS: Record<string, BenchmarkPreset> = {
  'inference-standard': {
    name: '标准推理Benchmark',
    models: ['qwen2.5-32b', 'qwen2.5-72b'],
    batch_sizes: [1, 8, 32],
    seq_lengths: [512, 2048],
    metrics: ['TTFT', 'TPOT', 'throughput', 'memory'],
  },
  'long-context': {
    name: '长上下文Benchmark',
    models: ['deepseek-v3', 'qwen2.5-72b'],
    batch_sizes: [1, 4],
    seq_lengths: [8192, 16384, 32768],
    metrics: ['memory', 'TTFT', 'E2E'],
  },
  'moe-benchmark': {
    name: 'MoE模型Benchmark',
    models: ['deepseek-v3', 'qwen3-235b'],
    batch_sizes: [8, 32],
    seq_lengths: [2048],
    metrics: ['throughput', 'memory', 'communication'],
  },
};

// ============================================
// 辅助函数
// ============================================

/** 获取模型预设 */
export function getModelPreset(modelId: string): LLMModelConfig {
  const preset = MODEL_PRESETS[modelId];
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
    chip: { ...preset.chip },
    board: { ...preset.board },
    rack: { ...preset.rack },
    pod: { ...preset.pod },
  };
}

/** 获取推理预设 */
export function getInferencePreset(inferenceId: string): InferenceConfig {
  const preset = INFERENCE_PRESETS[inferenceId];
  if (!preset) {
    throw new Error(`未找到推理预设: ${inferenceId}`);
  }
  return { ...preset };
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
    chip,
    board: { ...board },
    rack: { ...rack },
    pod: { ...pod },
  };
}
