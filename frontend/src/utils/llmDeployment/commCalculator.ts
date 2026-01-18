/**
 * LLM 部署分析系统 - 通信量计算器
 *
 * 计算各种并行策略的通信量
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  CommunicationAnalysis,
  HardwareConfig,
  getBytesPerElement,
} from './types';

// ============================================
// 张量并行 (TP) 通信量
// ============================================

/**
 * 计算 TP AllReduce 通信量 (Prefill 阶段)
 *
 * TP 在每层的 Attention 和 FFN 后需要 AllReduce
 * AllReduce 通信量 = 2 × (n-1)/n × message_size
 */
export function calculateTPCommVolumePrefill(
  model: LLMModelConfig,
  inference: InferenceConfig,
  tpSize: number
): number {
  if (tpSize <= 1) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const seq = inference.input_seq_length;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // 每层有 2 次 AllReduce: Attention 后 + FFN 后
  // 每次 AllReduce 的消息大小 = batch × seq × H × bytes
  const messageSize = batch * seq * H * bytesPerElement;

  // AllReduce 通信量 = 2 × (n-1)/n × message_size
  const allReduceFactor = 2 * (tpSize - 1) / tpSize;

  // 总通信量 = 2次/层 × 层数 × AllReduce通信量
  const totalBytes = 2 * L * messageSize * allReduceFactor;

  return totalBytes / 1e9; // GB
}

/**
 * 计算 TP AllReduce 通信量 (Decode 阶段，每 token)
 */
export function calculateTPCommVolumeDecode(
  model: LLMModelConfig,
  inference: InferenceConfig,
  tpSize: number
): number {
  if (tpSize <= 1) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // Decode 时 seq=1
  const messageSize = batch * 1 * H * bytesPerElement;
  const allReduceFactor = 2 * (tpSize - 1) / tpSize;
  const totalBytes = 2 * L * messageSize * allReduceFactor;

  return totalBytes / 1e9; // GB
}

// ============================================
// 流水线并行 (PP) 通信量
// ============================================

/**
 * 计算 PP P2P 通信量 (Prefill 阶段)
 *
 * PP 在每个 micro-batch 的 stage 边界需要 P2P 传输
 */
export function calculatePPCommVolumePrefill(
  model: LLMModelConfig,
  inference: InferenceConfig,
  ppSize: number,
  numMicroBatches: number
): number {
  if (ppSize <= 1) return 0;

  const H = model.hidden_size;
  const batch = inference.batch_size;
  const seq = inference.input_seq_length;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // micro-batch 大小
  const microBatchSize = batch / numMicroBatches;

  // 每个 micro-batch 在 (pp-1) 个 stage 边界需要传输
  // 消息大小 = micro_batch × seq × H × bytes
  const messageSize = microBatchSize * seq * H * bytesPerElement;

  // 总通信量 = num_micro_batches × (pp-1) × message_size
  const totalBytes = numMicroBatches * (ppSize - 1) * messageSize;

  return totalBytes / 1e9; // GB
}

/**
 * 计算 PP P2P 通信量 (Decode 阶段，每 token)
 */
export function calculatePPCommVolumeDecode(
  model: LLMModelConfig,
  inference: InferenceConfig,
  ppSize: number
): number {
  if (ppSize <= 1) return 0;

  const H = model.hidden_size;
  const batch = inference.batch_size;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // Decode 时 seq=1
  const messageSize = batch * 1 * H * bytesPerElement;

  // Decode 通常不用 micro-batch，直接传输
  const totalBytes = (ppSize - 1) * messageSize;

  return totalBytes / 1e9; // GB
}

// ============================================
// 专家并行 (EP) 通信量
// ============================================

/**
 * 计算 EP AllToAll 通信量 (Prefill 阶段)
 *
 * EP 在每个 MoE 层需要 2 次 AllToAll: 分发 + 收集
 */
export function calculateEPCommVolumePrefill(
  model: LLMModelConfig,
  inference: InferenceConfig,
  epSize: number
): number {
  if (epSize <= 1) return 0;
  if (model.model_type !== 'moe' || !model.moe_config) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const seq = inference.input_seq_length;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);
  const expertsPerTok = model.moe_config.num_experts_per_tok;

  // 每个 token 激活 expertsPerTok 个专家
  // AllToAll: 每个 token 发送到 expertsPerTok 个目标
  // 消息大小 = batch × seq × H × expertsPerTok × bytes
  const messageSize = batch * seq * H * expertsPerTok * bytesPerElement;

  // AllToAll 通信量因子 = 2 × (n-1)/n (分发 + 收集)
  const allToAllFactor = 2 * (epSize - 1) / epSize;

  // 总通信量 = 层数 × message_size × factor
  const totalBytes = L * messageSize * allToAllFactor;

  return totalBytes / 1e9; // GB
}

/**
 * 计算 EP AllToAll 通信量 (Decode 阶段，每 token)
 */
export function calculateEPCommVolumeDecode(
  model: LLMModelConfig,
  inference: InferenceConfig,
  epSize: number
): number {
  if (epSize <= 1) return 0;
  if (model.model_type !== 'moe' || !model.moe_config) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);
  const expertsPerTok = model.moe_config.num_experts_per_tok;

  // Decode 时 seq=1
  const messageSize = batch * 1 * H * expertsPerTok * bytesPerElement;
  const allToAllFactor = 2 * (epSize - 1) / epSize;
  const totalBytes = L * messageSize * allToAllFactor;

  return totalBytes / 1e9; // GB
}

// ============================================
// 序列并行 (SP) 通信量
// ============================================

/**
 * 计算 SP AllGather/ReduceScatter 通信量 (Prefill 阶段)
 *
 * SP 与 TP 结合使用，在 LayerNorm 和 Dropout 处需要通信
 */
export function calculateSPCommVolumePrefill(
  model: LLMModelConfig,
  inference: InferenceConfig,
  spSize: number
): number {
  if (spSize <= 1) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const seq = inference.input_seq_length;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // AllGather + ReduceScatter: 每次通信量 = (n-1)/n × message_size
  // 每层约 2 次
  const messageSize = batch * seq * H * bytesPerElement;
  const commFactor = (spSize - 1) / spSize;

  const totalBytes = 2 * L * messageSize * commFactor;

  return totalBytes / 1e9; // GB
}

/**
 * 计算 SP 通信量 (Decode 阶段)
 * 注意: Decode 阶段 seq=1，SP 效果有限
 */
export function calculateSPCommVolumeDecode(
  model: LLMModelConfig,
  inference: InferenceConfig,
  spSize: number
): number {
  if (spSize <= 1) return 0;

  const H = model.hidden_size;
  const L = model.num_layers;
  const batch = inference.batch_size;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  const messageSize = batch * 1 * H * bytesPerElement;
  const commFactor = (spSize - 1) / spSize;
  const totalBytes = 2 * L * messageSize * commFactor;

  return totalBytes / 1e9; // GB
}

// ============================================
// 综合通信分析
// ============================================

/**
 * 计算通信延迟
 */
function calculateCommLatency(
  commVolumeGB: number,
  bandwidthGBps: number,
  latencyUs: number
): number {
  if (commVolumeGB === 0) return 0;

  // 传输时间 + 启动延迟
  const transferTimeMs = (commVolumeGB / bandwidthGBps) * 1000;
  const startupLatencyMs = latencyUs / 1000;

  return transferTimeMs + startupLatencyMs;
}

/**
 * 完整通信分析
 */
export function analyzeCommunication(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): CommunicationAnalysis {
  const numMicroBatches = inference.num_micro_batches ?? Math.max(parallelism.pp, 4);

  // TP 通信量
  const tpCommPrefill = calculateTPCommVolumePrefill(model, inference, parallelism.tp);

  // PP 通信量
  const ppCommPrefill = calculatePPCommVolumePrefill(model, inference, parallelism.pp, numMicroBatches);

  // EP 通信量
  const epCommPrefill = calculateEPCommVolumePrefill(model, inference, parallelism.ep);

  // SP 通信量
  const spCommPrefill = calculateSPCommVolumePrefill(model, inference, parallelism.sp);

  // 总通信量 (Prefill)
  const totalCommPrefill = tpCommPrefill + ppCommPrefill + epCommPrefill + spCommPrefill;

  // 确定带宽
  // TP 通常在节点内 (NVLink)
  const tpBandwidth = hardware.node.intra_node_bandwidth_gbps;
  const tpLatency = hardware.node.intra_node_latency_us;

  // PP/EP 可能跨节点 (取决于配置)
  const totalChipsPerNode = hardware.node.chips_per_node;
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const numNodes = Math.ceil(totalChips / totalChipsPerNode);

  const ppBandwidth = numNodes > 1 ? hardware.cluster.inter_node_bandwidth_gbps : tpBandwidth;
  const ppLatency = numNodes > 1 ? hardware.cluster.inter_node_latency_us : tpLatency;

  // 计算延迟
  const tpLatencyMs = calculateCommLatency(tpCommPrefill, tpBandwidth, tpLatency);
  const ppLatencyMs = calculateCommLatency(ppCommPrefill, ppBandwidth, ppLatency);

  // 分析瓶颈
  let bottleneckDescription: string | undefined;
  if (tpLatencyMs > ppLatencyMs && tpLatencyMs > 10) {
    bottleneckDescription = `TP AllReduce 占主导 (${tpLatencyMs.toFixed(1)}ms)，考虑减小 TP 或使用更快互联`;
  } else if (ppLatencyMs > 10) {
    bottleneckDescription = `PP 通信占主导 (${ppLatencyMs.toFixed(1)}ms)，考虑增加 micro-batch 数量`;
  }

  return {
    tp_comm_volume_gb: tpCommPrefill,
    pp_comm_volume_gb: ppCommPrefill,
    ep_comm_volume_gb: epCommPrefill,
    sp_comm_volume_gb: spCommPrefill,
    total_comm_volume_gb: totalCommPrefill,
    tp_comm_latency_ms: tpLatencyMs,
    pp_comm_latency_ms: ppLatencyMs,
    bottleneck_description: bottleneckDescription,
  };
}

// ============================================
// 辅助函数
// ============================================

/**
 * 估算 DP 梯度同步通信量 (仅训练时需要)
 */
export function calculateDPGradientSyncVolume(
  model: LLMModelConfig,
  dpSize: number
): number {
  if (dpSize <= 1) return 0;

  const totalParams = calculateModelParamsForComm(model);
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // AllReduce 梯度: 2 × (n-1)/n × 参数量 × bytes
  const allReduceFactor = 2 * (dpSize - 1) / dpSize;
  const totalBytes = totalParams * bytesPerElement * allReduceFactor;

  return totalBytes / 1e9; // GB
}

/**
 * 辅助函数：计算模型参数量 (用于通信计算)
 */
function calculateModelParamsForComm(model: LLMModelConfig): number {
  const H = model.hidden_size;
  const L = model.num_layers;
  const V = model.vocab_size;
  const I = model.intermediate_size;

  let params = V * H + 4 * H * H * L + 3 * H * I * L + 2 * H * L;

  if (model.model_type === 'moe' && model.moe_config) {
    const numExperts = model.moe_config.num_experts;
    params = V * H + 4 * H * H * L + 3 * H * I * L * numExperts + 2 * H * L;
  }

  return params;
}
