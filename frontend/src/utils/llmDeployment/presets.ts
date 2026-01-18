/**
 * LLM 部署分析系统 - 预设配置
 *
 * 包含常见模型和硬件的预设配置
 */

import {
  LLMModelConfig,
  MLAConfig,
  ChipHardwareConfig,
  NodeConfig,
  ClusterConfig,
  HardwareConfig,
  InferenceConfig,
  BenchmarkScenario,
  BenchmarkPreset,
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
// 预设硬件配置
// ============================================

/** 算能 SG2260E (默认芯片) */
export const SG2260E: ChipHardwareConfig = {
  chip_type: 'SG2260E',
  compute_tflops_fp16: 64,     // FP16 算力
  compute_tops_int8: 128,      // INT8 算力
  num_cores: 8,                // 计算核心数
  memory_gb: 64,               // DRAM 容量
  memory_bandwidth_gbps: 273,  // DRAM 理论带宽
  memory_bandwidth_utilization: 0.893,  // 带宽利用率
  l2_cache_mb: 16,             // L2M 容量
  l2_bandwidth_gbps: 512,      // L2M 单向带宽
};

/** NVIDIA H100 SXM */
export const H100_SXM: ChipHardwareConfig = {
  chip_type: 'H100-SXM',
  compute_tflops_fp16: 1979,   // FP16 Tensor Core
  compute_tops_int8: 3958,     // INT8 Tensor Core
  memory_gb: 80,
  memory_bandwidth_gbps: 3350,
  memory_bandwidth_utilization: 0.9,
};

/** NVIDIA H100 PCIe */
export const H100_PCIE: ChipHardwareConfig = {
  chip_type: 'H100-PCIe',
  compute_tflops_fp16: 1513,   // FP16 Tensor Core
  compute_tops_int8: 3026,     // INT8 Tensor Core
  memory_gb: 80,
  memory_bandwidth_gbps: 2000,
  memory_bandwidth_utilization: 0.9,
};

/** NVIDIA A100 SXM */
export const A100_SXM: ChipHardwareConfig = {
  chip_type: 'A100-SXM',
  compute_tflops_fp16: 312,
  compute_tops_int8: 624,
  memory_gb: 80,
  memory_bandwidth_gbps: 2039,
  memory_bandwidth_utilization: 0.85,
};

/** NVIDIA A100 PCIe */
export const A100_PCIE: ChipHardwareConfig = {
  chip_type: 'A100-PCIe',
  compute_tflops_fp16: 312,
  compute_tops_int8: 624,
  memory_gb: 80,
  memory_bandwidth_gbps: 1935,
  memory_bandwidth_utilization: 0.85,
};

/** NVIDIA A800 (中国版) */
export const A800: ChipHardwareConfig = {
  chip_type: 'A800',
  compute_tflops_fp16: 312,
  compute_tops_int8: 624,
  memory_gb: 80,
  memory_bandwidth_gbps: 2039,
  memory_bandwidth_utilization: 0.85,
};

/** NVIDIA L40S */
export const L40S: ChipHardwareConfig = {
  chip_type: 'L40S',
  compute_tflops_fp16: 362,
  compute_tops_int8: 724,
  memory_gb: 48,
  memory_bandwidth_gbps: 864,
  memory_bandwidth_utilization: 0.85,
};

/** NVIDIA RTX 4090 */
export const RTX_4090: ChipHardwareConfig = {
  chip_type: 'RTX-4090',
  compute_tflops_fp16: 165,
  compute_tops_int8: 330,
  memory_gb: 24,
  memory_bandwidth_gbps: 1008,
  memory_bandwidth_utilization: 0.85,
};

/** AMD MI300X */
export const MI300X: ChipHardwareConfig = {
  chip_type: 'MI300X',
  compute_tflops_fp16: 1307,
  compute_tops_int8: 2614,
  memory_gb: 192,
  memory_bandwidth_gbps: 5300,
  memory_bandwidth_utilization: 0.85,
};

/** 华为昇腾910B */
export const ASCEND_910B: ChipHardwareConfig = {
  chip_type: 'Ascend-910B',
  compute_tflops_fp16: 320,
  compute_tops_int8: 640,
  memory_gb: 64,
  memory_bandwidth_gbps: 1600,
  memory_bandwidth_utilization: 0.85,
};

/** 所有预设芯片 */
export const CHIP_PRESETS: Record<string, ChipHardwareConfig> = {
  'sg2260e': SG2260E,        // 默认芯片
  'h100-sxm': H100_SXM,
  'h100-pcie': H100_PCIE,
  'a100-sxm': A100_SXM,
  'a100-pcie': A100_PCIE,
  'a800': A800,
  'l40s': L40S,
  'rtx-4090': RTX_4090,
  'mi300x': MI300X,
  'ascend-910b': ASCEND_910B,
};

/** 默认芯片 ID */
export const DEFAULT_CHIP_ID = 'sg2260e';

/** 获取芯片列表 */
export function getChipList(): Array<{ id: string; name: string; memory: string; compute: string; isCustom?: boolean }> {
  const builtIn = Object.entries(CHIP_PRESETS).map(([id, config]) => ({
    id,
    name: config.chip_type,
    memory: `${config.memory_gb}GB`,
    compute: `${config.compute_tflops_fp16} TFLOPs`,
    isCustom: false,
  }));
  const custom = Object.entries(getCustomChipPresets()).map(([id, config]) => ({
    id,
    name: config.chip_type,
    memory: `${config.memory_gb}GB`,
    compute: `${config.compute_tflops_fp16} TFLOPs`,
    isCustom: true,
  }));
  return [...builtIn, ...custom];
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

/** 获取芯片配置（包含内置和自定义） */
export function getChipConfig(id: string): ChipHardwareConfig | null {
  if (CHIP_PRESETS[id]) {
    return CHIP_PRESETS[id];
  }
  const custom = getCustomChipPresets();
  return custom[id] || null;
}

// ============================================
// 芯片互联配置映射
// ============================================

/** 芯片互联配置 */
export interface ChipInterconnectConfig {
  /** 互联类型名称 (如 NVLink 4.0, PCIe 4.0) */
  interconnect_type: string;
  /** 节点内带宽 (GB/s) */
  intra_node_bandwidth_gbps: number;
  /** 节点内延迟 (us) */
  intra_node_latency_us: number;
  /** 推荐的芯片数量/节点 */
  recommended_chips_per_node: number;
}

/** SG2260E 互联配置 */
export const SG2260E_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'TP Group',
  intra_node_bandwidth_gbps: 64,   // TP Group 内带宽
  intra_node_latency_us: 1,
  recommended_chips_per_node: 8,
};

/** H100 SXM 互联配置 - NVLink 4.0 */
export const H100_SXM_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'NVLink 4.0',
  intra_node_bandwidth_gbps: 900,
  intra_node_latency_us: 1,
  recommended_chips_per_node: 8,
};

/** H100 PCIe 互联配置 - PCIe 5.0 */
export const H100_PCIE_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'PCIe 5.0',
  intra_node_bandwidth_gbps: 128,
  intra_node_latency_us: 15,  // PCIe实测 ~13-20us
  recommended_chips_per_node: 8,
};

/** A100 SXM 互联配置 - NVLink 3.0 */
export const A100_SXM_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'NVLink 3.0',
  intra_node_bandwidth_gbps: 600,
  intra_node_latency_us: 2,  // 实测 ~2us
  recommended_chips_per_node: 8,
};

/** A100/A800 PCIe 互联配置 - PCIe 4.0 */
export const PCIE_4_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'PCIe 4.0',
  intra_node_bandwidth_gbps: 64,
  intra_node_latency_us: 15,  // PCIe实测 ~13-20us
  recommended_chips_per_node: 8,
};

/** MI300X 互联配置 - Infinity Fabric 3.0 */
export const MI300X_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'Infinity Fabric 3.0',
  intra_node_bandwidth_gbps: 896,
  intra_node_latency_us: 1,
  recommended_chips_per_node: 8,
};

/** Ascend 910B 互联配置 - HCCS */
export const ASCEND_910B_INTERCONNECT: ChipInterconnectConfig = {
  interconnect_type: 'HCCS',
  intra_node_bandwidth_gbps: 392,  // 7链路 × 56 GB/s
  intra_node_latency_us: 2,
  recommended_chips_per_node: 8,
};

/** 芯片ID到互联配置的映射 */
export const CHIP_INTERCONNECT_PRESETS: Record<string, ChipInterconnectConfig> = {
  'sg2260e': SG2260E_INTERCONNECT,  // 默认芯片
  'h100-sxm': H100_SXM_INTERCONNECT,
  'h100-pcie': H100_PCIE_INTERCONNECT,
  'a100-sxm': A100_SXM_INTERCONNECT,
  'a100-pcie': PCIE_4_INTERCONNECT,
  'a800': A100_SXM_INTERCONNECT,  // A800 也支持 NVLink 3.0
  'l40s': PCIE_4_INTERCONNECT,
  'rtx-4090': PCIE_4_INTERCONNECT,
  'mi300x': MI300X_INTERCONNECT,
  'ascend-910b': ASCEND_910B_INTERCONNECT,
};

/** 获取芯片互联配置 */
export function getChipInterconnectConfig(chipId: string): ChipInterconnectConfig | null {
  return CHIP_INTERCONNECT_PRESETS[chipId] || null;
}

// ============================================
// 预设节点配置
// ============================================

/** DGX H100 节点 (8x H100 NVLink) */
export const DGX_H100_NODE: NodeConfig = {
  chips_per_node: 8,
  intra_node_bandwidth_gbps: 900,  // NVLink 4.0
  intra_node_latency_us: 1,
};

/** DGX A100 节点 (8x A100 NVLink) */
export const DGX_A100_NODE: NodeConfig = {
  chips_per_node: 8,
  intra_node_bandwidth_gbps: 600,  // NVLink 3.0
  intra_node_latency_us: 1,
};

/** 通用 PCIe 节点 (8x GPU PCIe) */
export const PCIE_8GPU_NODE: NodeConfig = {
  chips_per_node: 8,
  intra_node_bandwidth_gbps: 64,   // PCIe 4.0 x16
  intra_node_latency_us: 5,
};

/** 所有预设节点 */
export const NODE_PRESETS: Record<string, NodeConfig> = {
  'dgx-h100': DGX_H100_NODE,
  'dgx-a100': DGX_A100_NODE,
  'pcie-8gpu': PCIE_8GPU_NODE,
};

// ============================================
// 预设集群配置
// ============================================

/** InfiniBand NDR 集群 */
export const IB_NDR_CLUSTER: ClusterConfig = {
  num_nodes: 16,
  inter_node_bandwidth_gbps: 400,  // NDR 400G
  inter_node_latency_us: 2,
};

/** InfiniBand HDR 集群 */
export const IB_HDR_CLUSTER: ClusterConfig = {
  num_nodes: 16,
  inter_node_bandwidth_gbps: 200,  // HDR 200G
  inter_node_latency_us: 2,
};

/** RoCE 集群 */
export const ROCE_CLUSTER: ClusterConfig = {
  num_nodes: 8,
  inter_node_bandwidth_gbps: 100,  // 100GbE
  inter_node_latency_us: 5,
};

/** 所有预设集群 */
export const CLUSTER_PRESETS: Record<string, ClusterConfig> = {
  'ib-ndr': IB_NDR_CLUSTER,
  'ib-hdr': IB_HDR_CLUSTER,
  'roce': ROCE_CLUSTER,
};

// ============================================
// 预设完整硬件配置
// ============================================

/** 8x H100 DGX 节点 */
export const HARDWARE_8xH100: HardwareConfig = {
  chip: H100_SXM,
  node: DGX_H100_NODE,
  cluster: { num_nodes: 1, inter_node_bandwidth_gbps: 0, inter_node_latency_us: 0 },
};

/** 16x H100 (2节点) */
export const HARDWARE_16xH100: HardwareConfig = {
  chip: H100_SXM,
  node: DGX_H100_NODE,
  cluster: { ...IB_NDR_CLUSTER, num_nodes: 2 },
};

/** 64x H100 (8节点) */
export const HARDWARE_64xH100: HardwareConfig = {
  chip: H100_SXM,
  node: DGX_H100_NODE,
  cluster: { ...IB_NDR_CLUSTER, num_nodes: 8 },
};

/** 8x A100 DGX 节点 */
export const HARDWARE_8xA100: HardwareConfig = {
  chip: A100_SXM,
  node: DGX_A100_NODE,
  cluster: { num_nodes: 1, inter_node_bandwidth_gbps: 0, inter_node_latency_us: 0 },
};

/** 所有预设硬件配置 */
export const HARDWARE_PRESETS: Record<string, HardwareConfig> = {
  '8xh100': HARDWARE_8xH100,
  '16xh100': HARDWARE_16xH100,
  '64xh100': HARDWARE_64xH100,
  '8xa100': HARDWARE_8xA100,
};

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
  const preset = CHIP_PRESETS[chipId];
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
    node: { ...preset.node },
    cluster: { ...preset.cluster },
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
  nodeId: string,
  numNodes: number,
  interNodeBandwidth: number = 400
): HardwareConfig {
  const chip = getChipPreset(chipId);
  const node = NODE_PRESETS[nodeId] ?? DGX_H100_NODE;

  return {
    chip,
    node: { ...node },
    cluster: {
      num_nodes: numNodes,
      inter_node_bandwidth_gbps: numNodes > 1 ? interNodeBandwidth : 0,
      inter_node_latency_us: numNodes > 1 ? 2 : 0,
    },
  };
}
