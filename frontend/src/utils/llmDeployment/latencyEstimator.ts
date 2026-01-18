/**
 * LLM 部署分析系统 - 延迟估算器
 *
 * 估算 Prefill/Decode 延迟，识别瓶颈
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  HardwareConfig,
  LatencyAnalysis,
  LatencyPercentiles,
  BottleneckType,
  CostAnalysis,
  BottleneckAnalysis,
  PhaseBottleneckAnalysis,
  getBytesPerElement,
} from './types';
import {
  calculatePrefillFlops,
  calculateDecodeFlopsPerToken,
  calculateModelMemory,
  calculateKVCacheMemory,
} from './modelCalculator';
import {
  calculateTPCommVolumePrefill,
  calculateTPCommVolumeDecode,
  calculatePPCommVolumePrefill,
  calculatePPCommVolumeDecode,
  calculateEPCommVolumePrefill,
  calculateEPCommVolumeDecode,
  calculateSPCommVolumePrefill,
  calculateSPCommVolumeDecode,
} from './commCalculator';

// ============================================
// 常量定义
// ============================================

/** HBM 效率因子 (实际带宽 / 峰值带宽) */
const HBM_EFFICIENCY = 0.85;

/** 计算效率因子 - 不同操作的实际 GPU 利用率 */
const COMPUTE_EFFICIENCY: Record<string, number> = {
  matmul_large: 0.70,    // 大矩阵乘法 (Prefill QKV, Dense FFN)
  matmul_small: 0.50,    // 小矩阵乘法 (Decode QKV)
  attention: 0.60,       // Attention Score/Output
  elementwise: 0.30,     // 逐元素操作 (LayerNorm, Softmax)
  moe_sparse: 0.20,      // MoE 稀疏计算 (小batch、sparse routing、expert load imbalance)
};

/** 随机访问延迟惩罚因子 */
const RANDOM_ACCESS_PENALTY = 1.5;

/** GB 到 bytes 转换 (使用 1024^3 与后端保持一致) */
const GB_TO_BYTES = 1024 * 1024 * 1024;

// ============================================
// 动态 MFU 估算
// ============================================

/**
 * 基于 Roofline 模型估算可达 MFU
 *
 * Roofline 模型原理:
 * - 算术强度 (AI) = FLOPs / Bytes
 * - 峰点 (Ridge Point) = 峰值算力 / 峰值带宽
 * - 实际算力利用率 = min(1, AI / Ridge Point)
 *
 * 参考:
 * - NVIDIA: https://developer.nvidia.com/blog/achieving-optimal-performance-with-roofline-analysis/
 */
export function estimateAchievableMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  phase: 'prefill' | 'decode',
  contextLength?: number
): number {
  // 计算每 token 的 FLOPs
  const avgContext = contextLength ?? (inference.input_seq_length + inference.output_seq_length / 2);
  const flopsPerToken = phase === 'prefill'
    ? calculatePrefillFlops(model, inference) / inference.input_seq_length
    : calculateDecodeFlopsPerToken(model, inference, avgContext);

  // 计算每 token 需要读取的数据量 (bytes)
  // 模型权重 (每 token 都需要读取)
  const modelMemoryGB = calculateModelMemory(model, parallelism);
  const modelMemoryBytes = modelMemoryGB * 1e9;

  // KV Cache (仅 decode 阶段)
  let kvCacheBytes = 0;
  if (phase === 'decode') {
    const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);
    const kvCacheRatio = avgContext / inference.max_seq_length;
    kvCacheBytes = kvCacheGB * kvCacheRatio * 1e9;
  }

  const bytesPerToken = modelMemoryBytes + kvCacheBytes;

  // 算术强度 (FLOPs / Bytes)
  const arithmeticIntensity = flopsPerToken / bytesPerToken;

  // Ridge Point = 峰值算力 (FLOPs/s) / 峰值带宽 (Bytes/s)
  const peakFlops = hardware.chip.compute_tflops_fp16 * 1e12;
  const peakBandwidthBytesPerS = hardware.chip.memory_bandwidth_gbps * GB_TO_BYTES * HBM_EFFICIENCY;
  const ridgePoint = peakFlops / peakBandwidthBytesPerS;

  // 基于 Roofline 的理论 MFU
  const theoreticalMFU = Math.min(1.0, arithmeticIntensity / ridgePoint);

  // 实际 MFU 通常更低，考虑以下因素:
  // - 并行效率损失 (TP/PP 通信开销)
  // - 启动开销 (kernel launch)
  // - 负载不均衡
  const parallelismOverhead = 1 - (parallelism.tp - 1) * 0.02 - (parallelism.pp - 1) * 0.03;
  const practicalFactor = 0.8; // 实际因素

  // 最终 MFU
  const achievableMFU = theoreticalMFU * parallelismOverhead * practicalFactor;

  // 放宽限制范围，避免过度约束
  // Prefill: 通常 20-65%
  // Decode: 通常 5-40%
  if (phase === 'prefill') {
    return Math.max(0.15, Math.min(0.65, achievableMFU));
  } else {
    return Math.max(0.05, Math.min(0.40, achievableMFU));
  }
}

// ============================================
// 逐操作延迟估算 (Per-Operation Roofline)
// ============================================

/**
 * 操作类型 - 用于确定计算效率
 */
type OpType = 'matmul_large' | 'matmul_small' | 'attention' | 'elementwise' | 'memory_only' | 'moe_sparse';

/**
 * 计算单个操作的延迟 (ms)
 * 使用 Roofline 模型: max(compute_time, memory_time)
 *
 * @param flops - 浮点运算数
 * @param memoryBytes - 内存访问字节数
 * @param peakFlops - 峰值算力 (FLOPs)
 * @param memoryBandwidthGBps - 内存带宽 (GB/s)
 * @param opType - 操作类型，决定计算效率
 * @param isRandomAccess - 是否随机访问（如 KV Cache 读取）
 */
function calcOpLatency(
  flops: number,
  memoryBytes: number,
  peakFlops: number,
  memoryBandwidthGBps: number,
  opType: OpType = 'matmul_large',
  isRandomAccess: boolean = false
): { latencyMs: number; computeMs: number; memoryMs: number } {
  // 计算效率
  const computeEfficiency = opType === 'memory_only' ? 1.0 : COMPUTE_EFFICIENCY[opType];

  // 计算时间 (考虑效率)
  const effectiveFlops = peakFlops * computeEfficiency;
  const computeTimeS = flops > 0 ? flops / effectiveFlops : 0;
  const computeTimeMs = computeTimeS * 1000;

  // 内存时间 (转换 bytes 到 GB，使用 1024^3)
  const memoryGB = memoryBytes / GB_TO_BYTES;
  const effectiveBandwidthGBps = memoryBandwidthGBps * HBM_EFFICIENCY;
  let memoryTimeMs = (memoryGB / effectiveBandwidthGBps) * 1000;

  // 随机访问惩罚
  if (isRandomAccess) {
    memoryTimeMs *= RANDOM_ACCESS_PENALTY;
  }

  return {
    latencyMs: Math.max(computeTimeMs, memoryTimeMs),
    computeMs: computeTimeMs,
    memoryMs: memoryTimeMs,
  };
}

/**
 * 逐操作 Prefill 延迟估算 (Per-Operation Roofline)
 *
 * 原理: 对每个子操作应用 Roofline 模型，然后求和
 * 单层包含: LayerNorm1 + QKV + Score + Softmax + Output + TP_AllReduce1
 *         + LayerNorm2 + FFN_gate + FFN_up + FFN_down + TP_AllReduce2 + KV_Cache_Write
 *
 * Σ max(compute_i, memory_i) >> max(Σ compute, Σ memory)
 */
export function estimatePrefillLatencyPerLayer(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): { totalMs: number; computeMs: number; memoryMs: number; commMs: number } {
  const numLayers = model.num_layers;
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  // 硬件参数 (带宽使用 GB/s，不再转换为 bytes/s)
  const peakFlops = hardware.chip.compute_tflops_fp16 * 1e12;
  const memoryBandwidthGBps = hardware.chip.memory_bandwidth_gbps;
  const tpBandwidthGBps = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;

  // 模型参数
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;
  const kvDim = headDim * numKVHeads;
  const B = inference.batch_size;
  const S = inference.input_seq_length;
  const weightBytesPerElement = getBytesPerElement(model.weight_dtype);
  const actBytesPerElement = getBytesPerElement(model.activation_dtype);

  // TP 分片后的参数
  const headsPerTP = Math.ceil(numHeads / parallelism.tp);
  const kvHeadsPerTP = Math.ceil(numKVHeads / parallelism.tp);

  let totalComputeMs = 0;
  let totalMemoryMs = 0;
  let totalCommMs = 0;
  let totalLatencyMs = 0;

  // 逐层计算
  for (let layer = 0; layer < layersPerChip; layer++) {
    // === 1. LayerNorm 1 (RMSNorm) ===
    const ln1Flops = 3 * B * S * H;  // RMSNorm: square, mean, normalize
    const ln1Memory = B * S * H * actBytesPerElement * 2;  // 读+写
    const ln1 = calcOpLatency(ln1Flops, ln1Memory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 2. QKV Projection (大矩阵乘法) ===
    const qkvFlops = 2 * B * S * H * (H + 2 * kvDim) / parallelism.tp;
    const qkvWeightBytes = H * (H + 2 * kvDim) * weightBytesPerElement / parallelism.tp;
    const qkvIOBytes = B * S * (H + H + 2 * kvDim) * actBytesPerElement / parallelism.tp;
    const qkv = calcOpLatency(qkvFlops, qkvWeightBytes + qkvIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_large');

    // === 3. Attention Score (Q @ K^T) - Attention 操作 ===
    const scoreFlops = 2 * B * headsPerTP * S * S * headDim;
    const scoreQBytes = B * headsPerTP * S * headDim * actBytesPerElement;
    const scoreKBytes = B * kvHeadsPerTP * S * headDim * actBytesPerElement;
    const scoreOutBytes = B * headsPerTP * S * S * actBytesPerElement;
    const score = calcOpLatency(scoreFlops, scoreQBytes + scoreKBytes + scoreOutBytes, peakFlops, memoryBandwidthGBps, 'attention');

    // === 4. Softmax (逐元素操作) ===
    const softmaxFlops = 5 * B * headsPerTP * S * S;  // exp + sum + div + sub + max
    const softmaxMemory = B * headsPerTP * S * S * actBytesPerElement * 2;  // 读+写
    const softmax = calcOpLatency(softmaxFlops, softmaxMemory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 5. Attention Output (Softmax @ V + Output Projection) ===
    const svFlops = 2 * B * headsPerTP * S * S * headDim;
    const outProjFlops = 2 * B * S * H * H / parallelism.tp;
    const attnOutFlops = svFlops + outProjFlops;
    const svMemory = B * headsPerTP * S * S * actBytesPerElement + B * kvHeadsPerTP * S * headDim * actBytesPerElement;
    const outProjMemory = H * H * weightBytesPerElement / parallelism.tp + B * S * H * actBytesPerElement;
    const attnOut = calcOpLatency(attnOutFlops, svMemory + outProjMemory, peakFlops, memoryBandwidthGBps, 'attention');

    // === 6. TP AllReduce 1 (Attention 后) ===
    let tpComm1Ms = 0;
    if (parallelism.tp > 1) {
      const allReduceGB = 2 * B * S * H * actBytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
      tpComm1Ms = (allReduceGB / tpBandwidthGBps) * 1000 + tpLatencyUs / 1000;
    }

    // === 7. LayerNorm 2 (RMSNorm) ===
    const ln2 = calcOpLatency(ln1Flops, ln1Memory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 8-10. FFN Gate/Up/Down ===
    // 判断是否为 MoE 层 (简化: model_type='moe' 时假设大部分层是 MoE)
    const isMoELayer = model.model_type === 'moe' && model.moe_config;
    let gate: { latencyMs: number; computeMs: number; memoryMs: number };
    let up: { latencyMs: number; computeMs: number; memoryMs: number };
    let down: { latencyMs: number; computeMs: number; memoryMs: number };
    let epCommMs = 0;

    if (isMoELayer && model.moe_config) {
      // MoE FFN: 稀疏计算，按 EP 切分 (不按 TP!)
      const moeConfig = model.moe_config;
      const ffnI = moeConfig.expert_intermediate_size ?? I;
      const numActiveExperts = moeConfig.num_experts_per_tok;

      // MoE FLOPs: B * S * num_active_experts * 2 * H * expert_I
      const moeGateFlops = 2 * B * S * H * ffnI * numActiveExperts;
      // MoE 权重: 需要加载 num_active_experts 个专家的权重 (不除以 TP!)
      const expertWeightBytes = 3 * H * ffnI * weightBytesPerElement;
      const totalWeightBytes = expertWeightBytes * numActiveExperts;
      const moeIOBytes = B * S * (H + ffnI) * actBytesPerElement * numActiveExperts;

      // 使用 moe_sparse 效率 (20%) - MoE 是稀疏计算
      gate = calcOpLatency(moeGateFlops / 2, totalWeightBytes / 3 + moeIOBytes / 2, peakFlops, memoryBandwidthGBps, 'moe_sparse');
      up = calcOpLatency(moeGateFlops / 2, totalWeightBytes / 3 + moeIOBytes / 2, peakFlops, memoryBandwidthGBps, 'moe_sparse');
      const moeDownFlops = 2 * B * S * ffnI * H * numActiveExperts;
      down = calcOpLatency(moeDownFlops, totalWeightBytes / 3 + B * S * (ffnI + H) * actBytesPerElement * numActiveExperts, peakFlops, memoryBandwidthGBps, 'moe_sparse');

      // EP All-to-All 通信开销 (dispatch + combine)
      if (parallelism.ep > 1) {
        const allToAllBytes = B * S * H * actBytesPerElement * 2;  // dispatch + combine
        epCommMs = (allToAllBytes / GB_TO_BYTES / tpBandwidthGBps) * 1000 * 2;  // 双向
      }
    } else {
      // Dense FFN: 大矩阵乘法，按 TP 切分
      const ffnI = I;
      const gateFlops = 2 * B * S * H * ffnI / parallelism.tp;
      const gateWeightBytes = H * ffnI * weightBytesPerElement / parallelism.tp;
      const gateIOBytes = B * S * (H + ffnI) * actBytesPerElement / parallelism.tp;
      gate = calcOpLatency(gateFlops, gateWeightBytes + gateIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_large');
      up = calcOpLatency(gateFlops, gateWeightBytes + gateIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_large');

      const downFlops = 2 * B * S * ffnI * H / parallelism.tp;
      const downWeightBytes = ffnI * H * weightBytesPerElement / parallelism.tp;
      const downIOBytes = B * S * (ffnI + H) * actBytesPerElement / parallelism.tp;
      down = calcOpLatency(downFlops, downWeightBytes + downIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_large');
    }

    // === 11. TP AllReduce 2 (FFN 后) ===
    let tpComm2Ms = 0;
    if (parallelism.tp > 1) {
      const allReduceGB = 2 * B * S * H * actBytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
      tpComm2Ms = (allReduceGB / tpBandwidthGBps) * 1000 + tpLatencyUs / 1000;
    }

    // === 12. KV Cache Write (Prefill 阶段写入所有 token 的 KV) ===
    const kvWriteBytes = 2 * B * S * kvHeadsPerTP * headDim * actBytesPerElement;
    const kvWrite = calcOpLatency(0, kvWriteBytes, peakFlops, memoryBandwidthGBps, 'memory_only');

    // 单层总延迟 = 所有操作延迟之和 (含 EP 通信)
    const layerLatencyMs = ln1.latencyMs + qkv.latencyMs + score.latencyMs + softmax.latencyMs +
                           attnOut.latencyMs + tpComm1Ms + ln2.latencyMs + gate.latencyMs +
                           up.latencyMs + down.latencyMs + epCommMs + tpComm2Ms + kvWrite.latencyMs;

    const layerComputeMs = ln1.computeMs + qkv.computeMs + score.computeMs + softmax.computeMs +
                           attnOut.computeMs + ln2.computeMs + gate.computeMs + up.computeMs + down.computeMs;

    const layerMemoryMs = ln1.memoryMs + qkv.memoryMs + score.memoryMs + softmax.memoryMs +
                          attnOut.memoryMs + ln2.memoryMs + gate.memoryMs + up.memoryMs +
                          down.memoryMs + kvWrite.memoryMs;

    totalComputeMs += layerComputeMs;
    totalMemoryMs += layerMemoryMs;
    totalCommMs += tpComm1Ms + tpComm2Ms + epCommMs;
    totalLatencyMs += layerLatencyMs;
  }

  return {
    totalMs: totalLatencyMs,
    computeMs: totalComputeMs,
    memoryMs: totalMemoryMs,
    commMs: totalCommMs,
  };
}

/**
 * 逐操作 Decode 延迟估算 (Per-Operation Roofline)
 *
 * Decode 阶段每 token 的延迟计算
 * 与 Prefill 类似，但：
 * - num_tokens = 1
 * - 需要读取 KV Cache (context_length 个 token)
 * - 不需要写入 KV Cache (只写 1 个 token，很小)
 */
export function estimateDecodeLatencyPerOperation(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number
): { totalMs: number; computeMs: number; memoryMs: number; commMs: number } {
  const numLayers = model.num_layers;
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  // 硬件参数 (带宽使用 GB/s)
  const peakFlops = hardware.chip.compute_tflops_fp16 * 1e12;
  const memoryBandwidthGBps = hardware.chip.memory_bandwidth_gbps;
  const tpBandwidthGBps = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;

  // 模型参数
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;
  const kvDim = headDim * numKVHeads;
  const B = inference.batch_size;
  const S = 1;  // Decode: 每次只处理 1 个 token
  const C = contextLength;  // 需要与 context 个 token 做 attention
  const weightBytesPerElement = getBytesPerElement(model.weight_dtype);
  const actBytesPerElement = getBytesPerElement(model.activation_dtype);

  // TP 分片后的参数
  const headsPerTP = Math.ceil(numHeads / parallelism.tp);
  const kvHeadsPerTP = Math.ceil(numKVHeads / parallelism.tp);

  let totalComputeMs = 0;
  let totalMemoryMs = 0;
  let totalCommMs = 0;
  let totalLatencyMs = 0;

  // 逐层计算
  for (let layer = 0; layer < layersPerChip; layer++) {
    // === 1. LayerNorm 1 (RMSNorm) - 逐元素操作 ===
    const ln1Flops = 3 * B * S * H;
    const ln1Memory = B * S * H * actBytesPerElement * 2;
    const ln1 = calcOpLatency(ln1Flops, ln1Memory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 2. QKV Projection - Decode 用小矩阵乘法 (batch=1, seq=1) ===
    const qkvFlops = 2 * B * S * H * (H + 2 * kvDim) / parallelism.tp;
    const qkvWeightBytes = H * (H + 2 * kvDim) * weightBytesPerElement / parallelism.tp;
    const qkvIOBytes = B * S * (H + H + 2 * kvDim) * actBytesPerElement / parallelism.tp;
    const qkv = calcOpLatency(qkvFlops, qkvWeightBytes + qkvIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_small');

    // === 3. KV Cache Read (Decode 特有：读取历史 context 的 KV) - 随机访问 ===
    const kvReadBytes = 2 * B * C * kvHeadsPerTP * headDim * actBytesPerElement;
    const kvRead = calcOpLatency(0, kvReadBytes, peakFlops, memoryBandwidthGBps, 'memory_only', true);

    // === 4. Attention Score (Q @ K^T): 1 query vs C keys ===
    const scoreFlops = 2 * B * headsPerTP * S * C * headDim;
    const scoreQBytes = B * headsPerTP * S * headDim * actBytesPerElement;
    const scoreKBytes = B * kvHeadsPerTP * C * headDim * actBytesPerElement;
    const scoreOutBytes = B * headsPerTP * S * C * actBytesPerElement;
    const score = calcOpLatency(scoreFlops, scoreQBytes + scoreKBytes + scoreOutBytes, peakFlops, memoryBandwidthGBps, 'attention');

    // === 5. Softmax - 逐元素操作 ===
    const softmaxFlops = 5 * B * headsPerTP * S * C;
    const softmaxMemory = B * headsPerTP * S * C * actBytesPerElement * 2;
    const softmax = calcOpLatency(softmaxFlops, softmaxMemory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 6. Attention Output (Softmax @ V + Output Projection) ===
    const svFlops = 2 * B * headsPerTP * S * C * headDim;
    const outProjFlops = 2 * B * S * H * H / parallelism.tp;
    const attnOutFlops = svFlops + outProjFlops;
    const svMemory = B * headsPerTP * S * C * actBytesPerElement + B * kvHeadsPerTP * C * headDim * actBytesPerElement;
    const outProjMemory = H * H * weightBytesPerElement / parallelism.tp + B * S * H * actBytesPerElement;
    const attnOut = calcOpLatency(attnOutFlops, svMemory + outProjMemory, peakFlops, memoryBandwidthGBps, 'matmul_small');

    // === 7. TP AllReduce 1 ===
    let tpComm1Ms = 0;
    if (parallelism.tp > 1) {
      const allReduceGB = 2 * B * S * H * actBytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
      tpComm1Ms = (allReduceGB / tpBandwidthGBps) * 1000 + tpLatencyUs / 1000;
    }

    // === 8. LayerNorm 2 - 逐元素操作 ===
    const ln2 = calcOpLatency(ln1Flops, ln1Memory, peakFlops, memoryBandwidthGBps, 'elementwise');

    // === 9-11. FFN (Gate, Up, Down) ===
    // 判断是否为 MoE 层
    const isMoELayer = model.model_type === 'moe' && model.moe_config;
    let gate: { latencyMs: number; computeMs: number; memoryMs: number };
    let up: { latencyMs: number; computeMs: number; memoryMs: number };
    let down: { latencyMs: number; computeMs: number; memoryMs: number };
    let epCommMs = 0;

    if (isMoELayer && model.moe_config) {
      // MoE FFN: 需要加载 num_experts_per_tok 个专家的权重 (不除以 TP!)
      const moeConfig = model.moe_config;
      const ffnI = moeConfig.expert_intermediate_size ?? I;
      const numActiveExperts = moeConfig.num_experts_per_tok;

      // Decode MoE: 仍然是 memory-bound，需要加载专家权重
      const moeGateFlops = 2 * B * S * H * ffnI * numActiveExperts;
      const expertWeightBytes = 3 * H * ffnI * weightBytesPerElement;
      const totalWeightBytes = expertWeightBytes * numActiveExperts;
      const moeIOBytes = B * S * (H + ffnI) * actBytesPerElement * numActiveExperts;

      // Decode 用 moe_sparse 效率
      gate = calcOpLatency(moeGateFlops / 2, totalWeightBytes / 3 + moeIOBytes / 2, peakFlops, memoryBandwidthGBps, 'moe_sparse');
      up = calcOpLatency(moeGateFlops / 2, totalWeightBytes / 3 + moeIOBytes / 2, peakFlops, memoryBandwidthGBps, 'moe_sparse');
      const moeDownFlops = 2 * B * S * ffnI * H * numActiveExperts;
      down = calcOpLatency(moeDownFlops, totalWeightBytes / 3 + B * S * (ffnI + H) * actBytesPerElement * numActiveExperts, peakFlops, memoryBandwidthGBps, 'moe_sparse');

      // EP All-to-All 通信
      if (parallelism.ep > 1) {
        const allToAllBytes = B * S * H * actBytesPerElement * 2;
        epCommMs = (allToAllBytes / GB_TO_BYTES / tpBandwidthGBps) * 1000 * 2;
      }
    } else {
      // Dense FFN: 小矩阵乘法
      const ffnI = I;
      const gateFlops = 2 * B * S * H * ffnI / parallelism.tp;
      const gateWeightBytes = H * ffnI * weightBytesPerElement / parallelism.tp;
      const gateIOBytes = B * S * (H + ffnI) * actBytesPerElement / parallelism.tp;
      gate = calcOpLatency(gateFlops, gateWeightBytes + gateIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_small');
      up = calcOpLatency(gateFlops, gateWeightBytes + gateIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_small');

      const downFlops = 2 * B * S * ffnI * H / parallelism.tp;
      const downWeightBytes = ffnI * H * weightBytesPerElement / parallelism.tp;
      const downIOBytes = B * S * (ffnI + H) * actBytesPerElement / parallelism.tp;
      down = calcOpLatency(downFlops, downWeightBytes + downIOBytes, peakFlops, memoryBandwidthGBps, 'matmul_small');
    }

    // === 12. TP AllReduce 2 ===
    let tpComm2Ms = 0;
    if (parallelism.tp > 1) {
      const allReduceGB = 2 * B * S * H * actBytesPerElement * (parallelism.tp - 1) / parallelism.tp / GB_TO_BYTES;
      tpComm2Ms = (allReduceGB / tpBandwidthGBps) * 1000 + tpLatencyUs / 1000;
    }

    // === 13. KV Cache Write (写入 1 个 token，很小) ===
    const kvWriteBytes = 2 * B * S * kvHeadsPerTP * headDim * actBytesPerElement;
    const kvWrite = calcOpLatency(0, kvWriteBytes, peakFlops, memoryBandwidthGBps, 'memory_only');

    // 单层总延迟 (含 EP 通信)
    const layerLatencyMs = ln1.latencyMs + qkv.latencyMs + kvRead.latencyMs + score.latencyMs +
                           softmax.latencyMs + attnOut.latencyMs + tpComm1Ms + ln2.latencyMs +
                           gate.latencyMs + up.latencyMs + down.latencyMs + epCommMs + tpComm2Ms + kvWrite.latencyMs;

    const layerComputeMs = ln1.computeMs + qkv.computeMs + score.computeMs + softmax.computeMs +
                           attnOut.computeMs + ln2.computeMs + gate.computeMs + up.computeMs + down.computeMs;

    const layerMemoryMs = ln1.memoryMs + qkv.memoryMs + kvRead.memoryMs + score.memoryMs +
                          softmax.memoryMs + attnOut.memoryMs + ln2.memoryMs + gate.memoryMs +
                          up.memoryMs + down.memoryMs + kvWrite.memoryMs;

    totalComputeMs += layerComputeMs;
    totalMemoryMs += layerMemoryMs;
    totalCommMs += tpComm1Ms + tpComm2Ms + epCommMs;
    totalLatencyMs += layerLatencyMs;
  }

  return {
    totalMs: totalLatencyMs,
    computeMs: totalComputeMs,
    memoryMs: totalMemoryMs,
    commMs: totalCommMs,
  };
}

// ============================================
// 计算延迟估算
// ============================================

/**
 * 估算 Prefill 阶段计算延迟
 *
 * 使用逐操作 Roofline 模型：对每个子操作应用 max(compute, memory)，然后求和
 * 返回总延迟（计算 + 访存 + 通信）
 */
export function estimatePrefillComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  _mfuEstimate?: number // 保留参数兼容性，但逐操作计算不使用整体 MFU
): number {
  // 使用逐操作计算方法
  const perLayerResult = estimatePrefillLatencyPerLayer(model, inference, parallelism, hardware);

  // 返回总延迟 (逐操作 Roofline 的结果)
  // 注意: 这个值用于 MFU 计算和 analyzeLatency 中的 prefillCompute
  return perLayerResult.totalMs;
}

/**
 * 估算 Decode 阶段单 token 计算延迟
 *
 * Decode 阶段通常是 memory-bound，但仍需计算 compute time 作为参考
 */
export function estimateDecodeComputeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number,
  mfuEstimate?: number // 可选，不传则使用动态估算
): number {
  // 使用动态 MFU 估算（如果未指定）
  const mfu = mfuEstimate ?? estimateAchievableMFU(model, inference, parallelism, hardware, 'decode', contextLength);

  const tokenFlops = calculateDecodeFlopsPerToken(model, inference, contextLength);

  const chipTflops = hardware.chip.compute_tflops_fp16;
  const flopsPerSecond = chipTflops * 1e12;
  const effectiveParallelism = parallelism.tp * parallelism.pp;
  const flopsPerChip = tokenFlops / effectiveParallelism;
  const computeTimeS = flopsPerChip / (flopsPerSecond * mfu);

  return computeTimeS * 1000; // ms
}

// ============================================
// 访存延迟估算
// ============================================

/**
 * 估算访存延迟 (Memory Bandwidth Bound)
 *
 * Decode 阶段通常是 memory-bound
 * 延迟 = 需读取的数据量 / (显存带宽 × HBM效率)
 */
export function estimateMemoryLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 模型权重大小 (每芯片)
  const modelMemoryGB = calculateModelMemory(model, parallelism);

  // KV Cache 大小 (每芯片) - 每 token 需要读取
  const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);

  // Decode 时每 token 需读取的数据量
  // = 模型权重 + 当前 context 的 KV Cache
  const dataToReadGB = modelMemoryGB + kvCacheGB * (inference.input_seq_length / inference.max_seq_length);

  // 显存带宽 (考虑 HBM 效率)
  const effectiveBandwidthGBps = hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY;

  // 访存时间
  const memoryTimeS = dataToReadGB / effectiveBandwidthGBps;

  return memoryTimeS * 1000; // ms
}

/**
 * 估算 Decode 阶段每 token 的访存延迟
 */
export function estimateDecodeMemoryLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  contextLength: number
): number {
  // 模型权重 (每 token 都要读一遍)
  const modelMemoryGB = calculateModelMemory(model, parallelism);

  // KV Cache 按当前 context 长度比例
  const kvCacheGB = calculateKVCacheMemory(model, inference, parallelism);
  const kvCacheRatio = contextLength / inference.max_seq_length;
  const currentKVCacheGB = kvCacheGB * kvCacheRatio;

  const dataToReadGB = modelMemoryGB + currentKVCacheGB;
  // 考虑 HBM 效率
  const effectiveBandwidthGBps = hardware.chip.memory_bandwidth_gbps * HBM_EFFICIENCY;
  const memoryTimeS = dataToReadGB / effectiveBandwidthGBps;

  return memoryTimeS * 1000; // ms
}

// ============================================
// 通信延迟估算
// ============================================

/**
 * 估算通信延迟
 *
 * 延迟 = 通信量 / 带宽 + 启动延迟
 */
export function estimateCommLatency(
  commVolumeGB: number,
  bandwidthGBps: number,
  startupLatencyUs: number
): number {
  if (commVolumeGB === 0) return 0;

  const transferTimeMs = (commVolumeGB / bandwidthGBps) * 1000;
  const startupLatencyMs = startupLatencyUs / 1000;

  return transferTimeMs + startupLatencyMs;
}

/**
 * 估算 Prefill 阶段总通信延迟
 */
export function estimatePrefillCommLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const numMicroBatches = inference.num_micro_batches ?? Math.max(parallelism.pp, 4);

  // 各策略通信量
  const tpComm = calculateTPCommVolumePrefill(model, inference, parallelism.tp);
  const ppComm = calculatePPCommVolumePrefill(model, inference, parallelism.pp, numMicroBatches);
  const epComm = calculateEPCommVolumePrefill(model, inference, parallelism.ep);
  const spComm = calculateSPCommVolumePrefill(model, inference, parallelism.sp);

  // 判断是否跨节点
  const totalChipsPerNode = hardware.node.chips_per_node;
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const numNodes = Math.ceil(totalChips / totalChipsPerNode);

  // TP 通常在节点内
  const tpBandwidth = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;

  // PP/EP 可能跨节点
  const ppBandwidth = numNodes > 1 ? hardware.cluster.inter_node_bandwidth_gbps : tpBandwidth;
  const ppLatencyUs = numNodes > 1 ? hardware.cluster.inter_node_latency_us : tpLatencyUs;

  // 计算各通信延迟
  const tpLatency = estimateCommLatency(tpComm, tpBandwidth, tpLatencyUs);
  const ppLatency = estimateCommLatency(ppComm, ppBandwidth, ppLatencyUs);
  const epLatency = estimateCommLatency(epComm, ppBandwidth, ppLatencyUs);
  const spLatency = estimateCommLatency(spComm, tpBandwidth, tpLatencyUs);

  // 通信可以部分重叠，但 TP 是关键路径
  // 简化: 取最大值 + 其他的一定比例
  const maxLatency = Math.max(tpLatency, ppLatency, epLatency, spLatency);
  const otherLatency = (tpLatency + ppLatency + epLatency + spLatency - maxLatency) * 0.3;

  return maxLatency + otherLatency;
}

/**
 * 估算 Decode 阶段单 token 通信延迟
 */
export function estimateDecodeCommLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 各策略通信量
  const tpComm = calculateTPCommVolumeDecode(model, inference, parallelism.tp);
  const ppComm = calculatePPCommVolumeDecode(model, inference, parallelism.pp);
  const epComm = calculateEPCommVolumeDecode(model, inference, parallelism.ep);
  const spComm = calculateSPCommVolumeDecode(model, inference, parallelism.sp);

  // 带宽选择
  const totalChipsPerNode = hardware.node.chips_per_node;
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const numNodes = Math.ceil(totalChips / totalChipsPerNode);

  const tpBandwidth = hardware.node.intra_node_bandwidth_gbps;
  const tpLatencyUs = hardware.node.intra_node_latency_us;
  const ppBandwidth = numNodes > 1 ? hardware.cluster.inter_node_bandwidth_gbps : tpBandwidth;
  const ppLatencyUs = numNodes > 1 ? hardware.cluster.inter_node_latency_us : tpLatencyUs;

  const tpLatency = estimateCommLatency(tpComm, tpBandwidth, tpLatencyUs);
  const ppLatency = estimateCommLatency(ppComm, ppBandwidth, ppLatencyUs);
  const epLatency = estimateCommLatency(epComm, ppBandwidth, ppLatencyUs);
  const spLatency = estimateCommLatency(spComm, tpBandwidth, tpLatencyUs);

  // Decode 通信通常更少，直接求和
  return tpLatency + ppLatency + epLatency + spLatency;
}

// ============================================
// 流水线气泡
// ============================================

/**
 * 计算流水线气泡比
 *
 * 气泡比 = (PP - 1) / (num_micro_batches + PP - 1)
 */
export function calculatePPBubbleRatio(
  ppSize: number,
  numMicroBatches: number
): number {
  if (ppSize <= 1) return 0;

  // 标准流水线气泡公式
  const bubbleRatio = (ppSize - 1) / (numMicroBatches + ppSize - 1);

  return bubbleRatio;
}

/**
 * 计算流水线效率
 */
export function calculatePPEfficiency(
  ppSize: number,
  numMicroBatches: number
): number {
  const bubbleRatio = calculatePPBubbleRatio(ppSize, numMicroBatches);
  return 1 - bubbleRatio;
}

// ============================================
// 瓶颈识别 (Roofline 模型)
// ============================================

/**
 * 计算硬件临界点 (Ridge Point)
 * Ridge Point = Peak Compute / Peak Memory BW
 */
function calculateRidgePoint(hardware: HardwareConfig): number {
  // 使用 compute_tflops_fp16 作为峰值算力
  const peakTflops = hardware.chip.compute_tflops_fp16;
  const memBwGBps = hardware.chip.memory_bandwidth_gbps;

  // 防止除零和无效值
  if (!peakTflops || !memBwGBps || memBwGBps <= 0) {
    // 返回默认值：H100 的 ridge point 约为 312 ops/byte
    return 312;
  }

  // Ridge Point = TFLOPs / (GB/s) = (TFLOPs * 1000) / (TB/s) = ops/byte
  // 简化: peakTflops (TFLOPs) / (memBwGBps / 1000) (TB/s) = peakTflops * 1000 / memBwGBps
  return (peakTflops * 1000) / memBwGBps;
}

/**
 * 计算 Prefill 阶段算术强度
 * AI = FLOPs / Bytes
 * Prefill: FLOPs ≈ 2 * Params * SeqLen * Batch
 * Bytes ≈ Params * dtype_bytes (模型权重读取一次)
 */
function calculatePrefillArithmeticIntensity(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const weightBytes = getBytesPerElement(model.weight_dtype);
  const actBytes = getBytesPerElement(model.activation_dtype);
  const params = estimateModelParams(model);

  // FLOPs: 2 * Params * SeqLen * Batch / TP (TP切分后单卡计算量)
  const flops = (2 * params * inference.input_seq_length * inference.batch_size) / parallelism.tp;

  // Bytes: 模型权重 / TP + 激活值
  // 激活值: batch * seq * hidden * dtype_bytes * 2 (输入输出)
  const modelBytes = (params * weightBytes) / parallelism.tp;
  const activationBytes = inference.batch_size * inference.input_seq_length * model.hidden_size * actBytes * 2;
  const totalBytes = modelBytes + activationBytes;

  return flops / totalBytes;
}

/**
 * 计算 Decode 阶段算术强度
 * Decode: FLOPs ≈ 2 * Params (每token)
 * Bytes ≈ Params * dtype + KV_cache_per_token
 */
function calculateDecodeArithmeticIntensity(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  contextLen: number
): number {
  const weightBytes = getBytesPerElement(model.weight_dtype);
  const actBytes = getBytesPerElement(model.activation_dtype);
  const params = estimateModelParams(model);

  // FLOPs per token: 2 * Params * Batch / TP
  const flopsPerToken = (2 * params * inference.batch_size) / parallelism.tp;

  // Bytes per token: 模型权重 + KV cache
  const modelBytes = (params * weightBytes) / parallelism.tp;
  // KV cache per token: 2 * num_layers * hidden_size * context_len * batch * dtype / TP
  const kvBytesPerToken = (2 * model.num_layers * model.hidden_size * contextLen * inference.batch_size * actBytes) / parallelism.tp;
  const totalBytes = modelBytes + kvBytesPerToken;

  return flopsPerToken / totalBytes;
}

/**
 * 估算模型参数量
 */
function estimateModelParams(model: LLMModelConfig): number {
  // 简化估算: 12 * num_layers * hidden_size^2
  return 12 * model.num_layers * model.hidden_size * model.hidden_size;
}

/**
 * 分析单阶段瓶颈 (Roofline 模型)
 */
function analyzePhaseBottleneck(
  phase: 'prefill' | 'decode',
  arithmeticIntensity: number,
  ridgePoint: number,
  computeLatency: number,
  memoryLatency: number,
  commLatency: number,
  actualLatency: number,
  _hardware: HardwareConfig,
  utilization: number
): PhaseBottleneckAnalysis {
  // 判断瓶颈类型
  const aiRatio = arithmeticIntensity / ridgePoint;
  let boundType: 'compute' | 'memory' | 'balanced';

  if (aiRatio < 0.8) {
    boundType = 'memory';
  } else if (aiRatio > 1.2) {
    boundType = 'compute';
  } else {
    boundType = 'balanced';
  }

  // 计算延迟占比
  const totalLatencyComponents = computeLatency + memoryLatency + commLatency;
  const computeRatio = totalLatencyComponents > 0 ? computeLatency / totalLatencyComponents : 0;
  const memoryRatio = totalLatencyComponents > 0 ? memoryLatency / totalLatencyComponents : 0;
  const commRatio = totalLatencyComponents > 0 ? commLatency / totalLatencyComponents : 0;

  // 计算理论最优延迟
  const theoreticalLatency = Math.max(computeLatency, memoryLatency);

  // 效率损失原因
  const efficiencyLoss: string[] = [];

  if (commRatio > 0.2) {
    efficiencyLoss.push(`通信开销 ${(commRatio * 100).toFixed(0)}%`);
  }
  if (utilization < 0.5) {
    efficiencyLoss.push(`硬件利用率低 ${(utilization * 100).toFixed(0)}%`);
  }
  if (boundType === 'memory' && phase === 'prefill') {
    efficiencyLoss.push('Prefill 意外进入 memory-bound (batch 过小?)');
  }
  if (boundType === 'compute' && phase === 'decode') {
    efficiencyLoss.push('Decode 意外进入 compute-bound (batch 过大?)');
  }

  return {
    phase,
    arithmetic_intensity: arithmeticIntensity,
    hardware_ridge_point: ridgePoint,
    bound_type: boundType,
    compute_ratio: computeRatio,
    memory_ratio: memoryRatio,
    comm_ratio: commRatio,
    utilization,
    theoretical_latency_ms: theoreticalLatency,
    actual_latency_ms: actualLatency,
    efficiency_loss: efficiencyLoss,
  };
}

/**
 * 完整瓶颈分析 (Roofline 模型)
 */
export function analyzeBottleneckRoofline(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig,
  prefillCompute: number,
  prefillComm: number,
  prefillTotal: number,
  decodeCompute: number,
  decodeMemory: number,
  decodeComm: number,
  decodePerToken: number,
  mfu: number,
  mbu: number
): BottleneckAnalysis {
  const ridgePoint = calculateRidgePoint(hardware);
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;

  // Prefill 算术强度
  const prefillAI = calculatePrefillArithmeticIntensity(model, inference, parallelism);

  // Decode 算术强度
  const decodeAI = calculateDecodeArithmeticIntensity(model, inference, parallelism, avgContextLen);

  // 分析各阶段瓶颈
  const prefillAnalysis = analyzePhaseBottleneck(
    'prefill', prefillAI, ridgePoint,
    prefillCompute, 0, prefillComm, prefillTotal,
    hardware, mfu
  );

  const decodeAnalysis = analyzePhaseBottleneck(
    'decode', decodeAI, ridgePoint,
    decodeCompute, decodeMemory, decodeComm, decodePerToken,
    hardware, mbu
  );

  // 判断主导阶段
  const prefillTotalTime = prefillTotal;
  const decodeTotalTime = decodePerToken * inference.output_seq_length;
  const dominantPhase = prefillTotalTime > decodeTotalTime ? 'prefill' : 'decode';

  // 综合瓶颈类型
  const dominantAnalysis = dominantPhase === 'prefill' ? prefillAnalysis : decodeAnalysis;
  let overallBottleneck: BottleneckType;

  if (dominantAnalysis.comm_ratio > 0.4) {
    overallBottleneck = 'communication';
  } else if (dominantAnalysis.bound_type === 'memory') {
    overallBottleneck = 'memory';
  } else if (dominantAnalysis.bound_type === 'compute') {
    overallBottleneck = 'compute';
  } else {
    overallBottleneck = 'balanced';
  }

  // 瓶颈严重程度 (1 - 利用率)
  const severity = 1 - dominantAnalysis.utilization;

  // 优化潜力分析
  const weightBytes = getBytesPerElement(model.weight_dtype);
  const currentAI = dominantPhase === 'prefill' ? prefillAI : decodeAI;

  // Batch scaling: 增大 batch 提升算术强度
  const batchScalingPotential = decodeAI < ridgePoint ? Math.min(ridgePoint / decodeAI, 4) : 1;

  // 量化: INT8 可减少一半内存访问（针对权重）
  const quantizationPotential = weightBytes > 1 ? weightBytes / 1 : 1;

  // 减少 TP: 减少通信开销
  const reduceTPPotential = dominantAnalysis.comm_ratio > 0.1 ? 1 / (1 - dominantAnalysis.comm_ratio * 0.5) : 1;

  // 生成摘要
  const phaseLabel = dominantPhase === 'prefill' ? 'Prefill' : 'Decode';
  const boundLabel = dominantAnalysis.bound_type === 'memory' ? '访存瓶颈' :
                     dominantAnalysis.bound_type === 'compute' ? '算力瓶颈' : '均衡状态';
  const summary = `${phaseLabel}阶段主导 (${(dominantPhase === 'prefill' ? prefillTotalTime : decodeTotalTime).toFixed(1)}ms)，` +
                  `${boundLabel}，算术强度 ${currentAI.toFixed(1)} ops/byte (临界点 ${ridgePoint.toFixed(0)})，` +
                  `利用率 ${(dominantAnalysis.utilization * 100).toFixed(0)}%`;

  return {
    prefill: prefillAnalysis,
    decode: decodeAnalysis,
    dominant_phase: dominantPhase,
    overall_bottleneck: overallBottleneck,
    severity,
    optimization_potential: {
      batch_scaling: {
        current_ai: currentAI,
        target_ai: ridgePoint,
        potential_speedup: batchScalingPotential,
      },
      quantization: {
        current_bytes: weightBytes,
        target_bytes: 1,
        potential_speedup: quantizationPotential,
      },
      reduce_tp: {
        current_comm_ratio: dominantAnalysis.comm_ratio,
        potential_speedup: reduceTPPotential,
      },
    },
    summary,
  };
}

/**
 * 识别主要瓶颈 (兼容旧接口)
 */
export function identifyBottleneck(
  computeLatency: number,
  memoryLatency: number,
  commLatency: number,
  bubbleRatio: number
): { type: BottleneckType; details: string } {
  // 气泡影响: 如果气泡比 > 20%，认为是瓶颈
  const bubbleImpact = bubbleRatio > 0.2;

  // 计算瓶颈得分
  const scores = [
    { type: 'compute' as BottleneckType, score: computeLatency, details: `计算延迟 ${computeLatency.toFixed(2)}ms` },
    { type: 'memory' as BottleneckType, score: memoryLatency, details: `访存延迟 ${memoryLatency.toFixed(2)}ms` },
    { type: 'communication' as BottleneckType, score: commLatency, details: `通信延迟 ${commLatency.toFixed(2)}ms` },
  ];

  // 排序找最大
  scores.sort((a, b) => b.score - a.score);
  const primary = scores[0];

  // 如果气泡影响显著，也要报告
  if (bubbleImpact && primary.type !== 'pipeline_bubble') {
    return {
      type: primary.type,
      details: `${primary.details}，流水线气泡比 ${(bubbleRatio * 100).toFixed(1)}%`,
    };
  }

  // 如果气泡是主要问题
  if (bubbleRatio > 0.3) {
    return {
      type: 'pipeline_bubble',
      details: `流水线气泡比过高 ${(bubbleRatio * 100).toFixed(1)}%，建议增加 micro-batch 数量`,
    };
  }

  return primary;
}

// ============================================
// 分位数估算
// ============================================

/**
 * 估算延迟分位数
 *
 * 业界基准 (MLPerf Inference v5.0):
 * - TTFT P99 ≤ 450ms (Server scenario)
 * - TPOT P99 ≤ 40ms (Server scenario)
 *
 * 分位数倍率基于实际测量数据的经验值:
 * - P50 ≈ 基准延迟 × 1.0 (中位数接近理论值)
 * - P90 ≈ 基准延迟 × 1.3 (网络/调度抖动)
 * - P99 ≈ 基准延迟 × 1.8 (尾部延迟，含 GC/页面缺失等)
 *
 * 影响因素:
 * - 网络通信: TP/PP 越高，通信抖动越大
 * - 负载变化: 高负载时排队延迟增加
 * - 批次变化: continuous batching 带来请求级差异
 *
 * 参考来源:
 * - Meta LLaMa 3 Benchmark: https://ai.meta.com/blog/meta-llama-3/
 * - Databricks MosaicML: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
 */
export function estimateLatencyPercentiles(
  baseLatencyMs: number,
  parallelism: ParallelismStrategy,
  isDecodePhase: boolean = false
): LatencyPercentiles {
  // 基础倍率
  let p90Multiplier = 1.3;
  let p99Multiplier = 1.8;

  // TP/PP 通信引入额外抖动
  if (parallelism.tp > 1) {
    const tpFactor = 1 + (parallelism.tp - 1) * 0.02; // 每增加 1 TP 增加 2%
    p90Multiplier *= tpFactor;
    p99Multiplier *= tpFactor;
  }

  if (parallelism.pp > 1) {
    const ppFactor = 1 + (parallelism.pp - 1) * 0.03; // PP 同步更敏感
    p90Multiplier *= ppFactor;
    p99Multiplier *= ppFactor;
  }

  // Decode 阶段抖动更稳定 (批次更小，通信更少)
  if (isDecodePhase) {
    p90Multiplier *= 0.9;
    p99Multiplier *= 0.85;
  }

  return {
    p50: baseLatencyMs,
    p90: baseLatencyMs * p90Multiplier,
    p99: baseLatencyMs * p99Multiplier,
  };
}

// ============================================
// 综合延迟分析
// ============================================

/**
 * 完整延迟分析
 */
export function analyzeLatency(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): LatencyAnalysis {
  const numMicroBatches = inference.num_micro_batches ?? Math.max(parallelism.pp, 4);

  // ===== Prefill 阶段 (使用逐层计算) =====
  // 逐层 Roofline: Σ max(compute_i, memory_i) + Σ comm_i
  const perLayerResult = estimatePrefillLatencyPerLayer(model, inference, parallelism, hardware);

  // 为了向后兼容，仍然分别返回 compute 和 comm
  const prefillCompute = Math.max(perLayerResult.computeMs, perLayerResult.memoryMs);
  const prefillComm = perLayerResult.commMs;

  // 流水线气泡
  const bubbleRatio = calculatePPBubbleRatio(parallelism.pp, numMicroBatches);

  // Prefill 总延迟 (使用逐层结果，已包含计算+访存+通信)
  // 逐层计算: Σ [max(compute_i, memory_i) + comm_i]
  const prefillIdeal = perLayerResult.totalMs;
  const prefillTotal = prefillIdeal / (1 - bubbleRatio);

  // ===== Decode 阶段 (使用逐操作计算) =====
  // 使用平均 context 长度
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;

  // 逐操作 Roofline 计算
  const decodePerOpResult = estimateDecodeLatencyPerOperation(
    model, inference, parallelism, hardware, avgContextLen
  );

  // 为了向后兼容，仍然分别返回 compute 和 memory
  const decodeCompute = decodePerOpResult.computeMs;
  const decodeMemory = decodePerOpResult.memoryMs;
  const decodeComm = decodePerOpResult.commMs;

  // Decode 每 token 延迟 (使用逐操作结果)
  const decodePerToken = decodePerOpResult.totalMs;

  // ===== 端到端延迟 =====
  const e2eLatency = prefillTotal + decodePerToken * inference.output_seq_length;

  // ===== 瓶颈识别 =====
  // Prefill 瓶颈
  const prefillBottleneck = identifyBottleneck(prefillCompute, 0, prefillComm, bubbleRatio);

  // Decode 瓶颈
  const decodeBottleneck = identifyBottleneck(decodeCompute, decodeMemory, decodeComm, 0);

  // 综合瓶颈 (取更严重的)
  let bottleneckType: BottleneckType;
  let bottleneckDetails: string;

  if (prefillTotal > decodePerToken * inference.output_seq_length) {
    // Prefill 占主导
    bottleneckType = prefillBottleneck.type;
    bottleneckDetails = `Prefill 阶段: ${prefillBottleneck.details}`;
  } else {
    // Decode 占主导
    bottleneckType = decodeBottleneck.type;
    bottleneckDetails = `Decode 阶段: ${decodeBottleneck.details}`;
  }

  // ===== 分位数估算 =====
  const ttftPercentiles = estimateLatencyPercentiles(prefillTotal, parallelism, false);
  const tpotPercentiles = estimateLatencyPercentiles(decodePerToken, parallelism, true);

  // ===== Prefill FLOPs =====
  const prefillFlops = calculatePrefillFlops(model, inference);

  return {
    prefill_compute_latency_ms: prefillCompute,
    prefill_comm_latency_ms: prefillComm,
    prefill_total_latency_ms: prefillTotal,
    prefill_flops: prefillFlops,
    decode_compute_latency_ms: decodeCompute,
    decode_memory_latency_ms: decodeMemory,
    decode_comm_latency_ms: decodeComm,
    decode_per_token_latency_ms: decodePerToken,
    end_to_end_latency_ms: e2eLatency,
    pipeline_bubble_ratio: bubbleRatio,
    bottleneck_type: bottleneckType,
    bottleneck_details: bottleneckDetails,
    ttft_percentiles: ttftPercentiles,
    tpot_percentiles: tpotPercentiles,
  };
}

// ============================================
// 吞吐量估算
// ============================================

/**
 * 估算 TPS per Batch (用户体验指标)
 * 公式: TPS_batch = 1000 / T_decode(ms)
 * 其中 T_decode = TPOT = decode_per_token_latency_ms
 * SLO约束: ≥10 tokens/s
 */
export function estimateTpsPerBatch(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const tpot_ms = latency.decode_per_token_latency_ms;

  // TPS per Batch = 1000 / TPOT(ms)
  return tpot_ms > 0 ? 1000 / tpot_ms : 0;
}

/**
 * 估算 TPS per Chip (成本效益指标，优化目标)
 * 公式: TPS_chip = B × 1000 / T_decode(ms) = B × TPS_batch
 * 其中 B = batch_size, T_decode = TPOT
 */
export function estimateTpsPerChip(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tpsPerBatch = estimateTpsPerBatch(model, inference, parallelism, hardware);
  // TPS per Chip = batch_size × TPS per Batch
  return inference.batch_size * tpsPerBatch;
}

/**
 * 估算 token 吞吐量 (集群总吞吐)
 * 公式: Total TPS = TPS_chip × NumChips
 */
export function estimateTokenThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tpsPerChip = estimateTpsPerChip(model, inference, parallelism, hardware);
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;

  // Total TPS = TPS_chip × NumChips
  return tpsPerChip * totalChips;
}

/**
 * 估算请求吞吐量
 */
export function estimateRequestThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);
  return tokenThroughput / inference.output_seq_length;
}

/**
 * 估算 Decode 阶段 MFU (Model FLOPs Utilization)
 *
 * 注意: Decode 阶段是 memory-bound，MFU 通常很低 (5-15%)
 */
export function estimateMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);

  // 每 token 的理论 FLOPs
  const flopsPerToken = calculateDecodeFlopsPerToken(
    model,
    inference,
    inference.input_seq_length + inference.output_seq_length / 2
  );

  // 实际算力使用
  const actualFlopsPerSecond = tokenThroughput * flopsPerToken;

  // 理论峰值算力 (所有芯片)
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const theoreticalFlopsPerSecond = hardware.chip.compute_tflops_fp16 * 1e12 * totalChips;

  return actualFlopsPerSecond / theoreticalFlopsPerSecond;
}

/**
 * 估算 Prefill 阶段 MFU (Model FLOPs Utilization)
 *
 * Prefill 阶段是 compute-bound，MFU 通常较高 (30-50%)
 * 此函数用于与仿真结果对比 (仿真计算的是 Prefill MFU)
 */
export function estimatePrefillMFU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // Prefill 总 FLOPs
  const prefillFlops = calculatePrefillFlops(model, inference);

  // Prefill 时间
  const prefillTimeMs = estimatePrefillComputeLatency(model, inference, parallelism, hardware);
  const prefillTimeS = prefillTimeMs / 1000;

  // 实际算力 (TFLOPs)
  const achievedTflops = (prefillFlops / 1e12) / prefillTimeS;

  // 单 DP 副本的峰值算力 (tp * pp 个芯片)
  const chipsPerReplica = parallelism.tp * parallelism.pp;
  const peakTflops = hardware.chip.compute_tflops_fp16 * chipsPerReplica;

  return achievedTflops / peakTflops;
}

/**
 * 估算理论最大吞吐量
 */
export function estimateTheoreticalMaxThroughput(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  // 假设 100% MFU
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;
  const theoreticalFlopsPerSecond = hardware.chip.compute_tflops_fp16 * 1e12 * totalChips;

  const flopsPerToken = calculateDecodeFlopsPerToken(
    model,
    inference,
    inference.input_seq_length + inference.output_seq_length / 2
  );

  return theoreticalFlopsPerSecond / flopsPerToken;
}

/** Kernel Launch 开销常量 */
const KERNELS_PER_LAYER = 12;  // 每层约 12 个 kernel (LN, QKV, Attn, FFN 等)
const KERNEL_LAUNCH_US = 15;   // 每个 kernel 约 15μs launch 时间

/**
 * 估算 MBU (Memory Bandwidth Utilization)
 *
 * MBU = Achieved_Bandwidth / Peak_Bandwidth
 * 其中 Achieved_Bandwidth = (Model_Size + KV_Cache_Size) / TPOT
 *
 * 注意: TPOT 包含 kernel launch 开销，这会降低有效 MBU
 *
 * 业界标准公式来源:
 * - Databricks: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
 * - NVIDIA: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
 */
export function estimateMBU(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): number {
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const tpotComputeMs = latency.decode_per_token_latency_ms;

  if (tpotComputeMs <= 0) return 0;

  // Kernel launch 开销 (显著影响 Decode 性能)
  const layersPerChip = Math.ceil(model.num_layers / parallelism.pp);
  const kernelOverheadMs = KERNELS_PER_LAYER * KERNEL_LAUNCH_US / 1000 * layersPerChip;
  const tpotTotalMs = tpotComputeMs + kernelOverheadMs;
  const tpotSeconds = tpotTotalMs / 1000;

  // 计算 Decode 每 token 需读取的数据量 (正确处理 MoE)
  const weightBytesPerElement = getBytesPerElement(model.weight_dtype);
  const H = model.hidden_size;

  // 1. Attention 权重 (每层, 按 TP 切分)
  let attnWeightPerLayer: number;
  if (model.mla_config) {
    // MLA Attention: 5 个投影矩阵，与 modelCalculator.ts 保持一致
    const mla = model.mla_config;
    const Nh = model.num_attention_heads;

    // W_q_a: H → q_lora_rank (11.0M for DeepSeek-V3)
    const W_q_a = H * mla.q_lora_rank;
    // W_q_b: q_lora_rank → Nh × (qk_nope_head_dim + qk_rope_head_dim) (37.7M)
    const W_q_b = mla.q_lora_rank * Nh * (mla.qk_nope_head_dim + mla.qk_rope_head_dim);
    // W_kv_a: H → (kv_lora_rank + qk_rope_head_dim) (4.1M)
    const W_kv_a = H * (mla.kv_lora_rank + mla.qk_rope_head_dim);
    // W_kv_b: kv_lora_rank → Nh × (qk_nope_head_dim + v_head_dim) (16.8M)
    const W_kv_b = mla.kv_lora_rank * Nh * (mla.qk_nope_head_dim + mla.v_head_dim);
    // W_o: Nh × v_head_dim → H (117.4M)
    const W_o = Nh * mla.v_head_dim * H;

    attnWeightPerLayer = (W_q_a + W_q_b + W_kv_a + W_kv_b + W_o) * weightBytesPerElement / parallelism.tp;
  } else {
    // 标准 MHA
    attnWeightPerLayer = 4 * H * H * weightBytesPerElement / parallelism.tp;
  }

  // 2. FFN 权重 (需要区分 Dense 和 MoE)
  let modelWeightBytes: number;
  if (model.model_type === 'moe' && model.moe_config) {
    // MoE 模型: 前几层是 Dense，其余是 MoE
    const numDenseLayers = model.moe_config.first_k_dense_replace ?? 3;  // 使用配置的 first_k_dense_replace
    const numMoELayers = model.num_layers - numDenseLayers;

    // Dense FFN (按 TP 切分)
    const denseFFNWeightPerLayer = 3 * H * model.intermediate_size * weightBytesPerElement / parallelism.tp;
    // MoE FFN (每 token 激活 k 个 experts, 不按 TP 切分!)
    const moeFFNWeightPerLayer = 3 * H * (model.moe_config.expert_intermediate_size ?? model.intermediate_size) *
      weightBytesPerElement * model.moe_config.num_experts_per_tok;

    modelWeightBytes = (attnWeightPerLayer + denseFFNWeightPerLayer) * numDenseLayers +
                       (attnWeightPerLayer + moeFFNWeightPerLayer) * numMoELayers;
  } else {
    // Dense 模型
    const ffnWeightPerLayer = 3 * H * model.intermediate_size * weightBytesPerElement / parallelism.tp;
    modelWeightBytes = (attnWeightPerLayer + ffnWeightPerLayer) * model.num_layers;
  }

  // 3. KV Cache (使用 MLA 压缩后的维度，使用激活精度)
  const actBytesPerElement = getBytesPerElement(model.activation_dtype);
  const avgContextLen = inference.input_seq_length + inference.output_seq_length / 2;
  const kvDim = model.mla_config?.kv_lora_rank ??
    (model.hidden_size / model.num_attention_heads * model.num_kv_heads);
  const kvCacheBytes = 2 * inference.batch_size * avgContextLen * kvDim *
    actBytesPerElement / parallelism.tp * model.num_layers;

  // 4. 计算 MBU
  const dataReadPerTokenGB = (modelWeightBytes + kvCacheBytes) / GB_TO_BYTES;
  const achievedBandwidthGBps = dataReadPerTokenGB / tpotSeconds;
  const peakBandwidthGBps = hardware.chip.memory_bandwidth_gbps;

  const mbu = achievedBandwidthGBps / peakBandwidthGBps;

  // MBU 不应超过 1 (理论上)
  return Math.min(mbu, 1.0);
}

// ============================================
// 成本分析
// ============================================

/**
 * 默认芯片成本表 ($/hour)
 * 数据来源: 云服务商按需实例定价 (2025年)
 * - AWS: https://aws.amazon.com/ec2/instance-types/
 * - Azure: https://azure.microsoft.com/pricing/details/virtual-machines/
 * - GCP: https://cloud.google.com/compute/all-pricing
 */
const DEFAULT_CHIP_COSTS: Record<string, number> = {
  // NVIDIA
  'H100': 4.5,       // ~$32-35/h per 8-GPU node
  'H200': 6.0,       // 预估 (尚未普遍)
  'A100-80GB': 3.0,  // ~$24/h per 8-GPU node
  'A100-40GB': 2.5,
  'L40S': 1.5,
  'A10': 1.0,
  // AMD
  'MI300X': 4.0,     // 预估
  'MI250X': 2.5,
  // 国产
  'Ascend-910B': 2.0, // 预估
};

/**
 * 获取芯片成本
 */
function getChipCost(hardware: HardwareConfig): number {
  // 优先使用配置的成本
  if (hardware.chip.cost_per_hour !== undefined) {
    return hardware.chip.cost_per_hour;
  }

  // 查找默认成本
  const chipType = hardware.chip.chip_type;
  for (const [name, cost] of Object.entries(DEFAULT_CHIP_COSTS)) {
    if (chipType.toLowerCase().includes(name.toLowerCase())) {
      return cost;
    }
  }

  // 默认成本 (基于算力估算: ~$0.01 per TFLOP-hour)
  return hardware.chip.compute_tflops_fp16 * 0.01;
}

/**
 * 估算成本分析
 *
 * 业界成本计算标准:
 * - $/M tokens = (Hardware_Cost_Per_Hour × 1e6) / (Tokens_Per_Second × 3600)
 * - 输入/输出成本比例通常为 1:3 ~ 1:5 (因为 Prefill 计算密度高但快，Decode 慢)
 *
 * 参考:
 * - OpenAI Pricing: https://openai.com/pricing
 * - Anthropic Pricing: https://www.anthropic.com/pricing
 * - Together AI: https://www.together.ai/pricing
 */
export function estimateCost(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  hardware: HardwareConfig
): CostAnalysis {
  // 计算总芯片数
  const totalChips = parallelism.dp * parallelism.tp * parallelism.pp * parallelism.ep;

  // 每芯片成本
  const chipCostPerHour = getChipCost(hardware);

  // 总硬件成本
  const totalCostPerHour = chipCostPerHour * totalChips;

  // 获取吞吐量
  const tokenThroughput = estimateTokenThroughput(model, inference, parallelism, hardware);

  if (tokenThroughput <= 0) {
    return {
      hardware_cost_per_hour: chipCostPerHour,
      total_hardware_cost_per_hour: totalCostPerHour,
      cost_per_million_tokens: Infinity,
      input_cost_per_million_tokens: Infinity,
      output_cost_per_million_tokens: Infinity,
      tokens_per_dollar: 0,
    };
  }

  // 每百万 token 成本 (综合)
  // $/M tokens = ($/hour × 1e6) / (tokens/s × 3600s/hour)
  const tokensPerHour = tokenThroughput * 3600;
  const costPerMillionTokens = (totalCostPerHour * 1e6) / tokensPerHour;

  // 输入/输出成本分解
  // 通常输出 token 比输入 token 成本高 3-4 倍
  // 原因: Prefill (输入) 是 compute-bound 且批量处理
  //       Decode (输出) 是 memory-bound 且逐 token 生成
  const latency = analyzeLatency(model, inference, parallelism, hardware);
  const prefillTime = latency.prefill_total_latency_ms;
  const decodeTime = latency.decode_per_token_latency_ms * inference.output_seq_length;
  const totalTime = prefillTime + decodeTime;

  // 按时间比例分配成本
  const inputRatio = prefillTime / totalTime;
  const outputRatio = decodeTime / totalTime;

  // 输入成本 ($/M input tokens)
  const inputCostPerMillion = (totalCostPerHour * 1e6 * inputRatio) /
                               (inference.batch_size * inference.input_seq_length * 3600 / (totalTime / 1000));

  // 输出成本 ($/M output tokens)
  const outputCostPerMillion = (totalCostPerHour * 1e6 * outputRatio) /
                                (inference.batch_size * inference.output_seq_length * 3600 / (totalTime / 1000));

  // Token/美元效率
  const tokensPerDollar = tokensPerHour / totalCostPerHour;

  return {
    hardware_cost_per_hour: chipCostPerHour,
    total_hardware_cost_per_hour: totalCostPerHour,
    cost_per_million_tokens: costPerMillionTokens,
    input_cost_per_million_tokens: inputCostPerMillion,
    output_cost_per_million_tokens: outputCostPerMillion,
    tokens_per_dollar: tokensPerDollar,
  };
}
