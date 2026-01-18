/**
 * LLM 部署分析系统 - 模型计算器
 *
 * 计算模型参数量、显存需求、FLOPs
 */

import {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
  MemoryAnalysis,
  getBytesPerElement,
} from './types';

// ============================================
// 参数量计算
// ============================================

/**
 * 计算模型总参数量
 *
 * 参考模型参数量:
 * - LLaMA-7B: 6.74B (tie_word_embeddings=true)
 * - LLaMA-70B: 68.98B (tie_word_embeddings=true)
 * - Qwen-7B: 7.72B (tie_word_embeddings=false)
 */
export function calculateModelParams(model: LLMModelConfig): number {
  const H = model.hidden_size;
  const L = model.num_layers;
  const V = model.vocab_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;

  // 每个头的维度
  const headDim = H / numHeads;

  // Embedding 层: token embedding
  const embeddingParams = V * H;

  // LM Head: 如果共享 embedding 权重则不额外计算
  // 大多数现代模型 (LLaMA, Mistral, DeepSeek) 共享权重
  const tieWordEmbeddings = model.tie_word_embeddings ?? true;
  const lmHeadParams = tieWordEmbeddings ? 0 : H * V;

  // 每层 Transformer 参数:
  // Attention 参数量计算
  let attentionParams: number;

  if (model.attention_type === 'mla' && model.mla_config) {
    // MLA (Multi-head Latent Attention): 5 个投影矩阵
    // 参考文档《性能指标计算》: 单层 MLA 参数量 = 187M
    const mla = model.mla_config;
    const Nh = model.num_attention_heads;

    // q_a_proj: H → q_lora_rank
    const W_q_a = H * mla.q_lora_rank;
    // q_b_proj: q_lora_rank → Nh × (qk_nope_head_dim + qk_rope_head_dim)
    const W_q_b = mla.q_lora_rank * Nh * (mla.qk_nope_head_dim + mla.qk_rope_head_dim);
    // kv_a_proj: H → (kv_lora_rank + qk_rope_head_dim)
    const W_kv_a = H * (mla.kv_lora_rank + mla.qk_rope_head_dim);
    // kv_b_proj: kv_lora_rank → Nh × (qk_nope_head_dim + v_head_dim)
    const W_kv_b = mla.kv_lora_rank * Nh * (mla.qk_nope_head_dim + mla.v_head_dim);
    // o_proj: Nh × v_head_dim → H
    const W_o = Nh * mla.v_head_dim * H;

    attentionParams = W_q_a + W_q_b + W_kv_a + W_kv_b + W_o;
  } else {
    // 标准 GQA/MHA
    //   - Q: H * H
    //   - K: H * headDim * numKVHeads
    //   - V: H * headDim * numKVHeads
    //   - O: H * H
    const qParams = H * H;
    const kParams = H * headDim * numKVHeads;
    const vParams = H * headDim * numKVHeads;
    const oParams = H * H;
    attentionParams = qParams + kParams + vParams + oParams;
  }

  // FFN (SwiGLU):
  //   - gate: H * I
  //   - up: H * I
  //   - down: I * H
  let ffnParams = 3 * H * I;

  // MoE: 区分 Dense 层和 MoE 层
  if (model.model_type === 'moe' && model.moe_config) {
    const numExperts = model.moe_config.num_experts;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    const expertI = model.moe_config.expert_intermediate_size ?? I;
    const firstKDense = model.moe_config.first_k_dense_replace ?? 0;
    const moeLayerFreq = model.moe_config.moe_layer_freq ?? 1;

    // Dense 层 (前 firstKDense 层): 使用标准 FFN
    const numDenseLayers = Math.min(firstKDense, L);
    const denseFFNParams = 3 * H * I * numDenseLayers;

    // MoE 层: 考虑 moe_layer_freq (1=每层MoE, 2=隔层MoE)
    const remainingLayers = L - numDenseLayers;
    const numMoELayers = Math.floor(remainingLayers / moeLayerFreq);
    const numNonMoELayers = remainingLayers - numMoELayers;

    // MoE 层参数
    const moeFFNParams = (3 * H * expertI * (numExperts + numSharedExperts) + H * numExperts) * numMoELayers;
    // 非 MoE 的剩余层使用标准 FFN
    const nonMoeFFNParams = 3 * H * I * numNonMoELayers;

    // 覆盖 ffnParams（不再乘以 L，因为已经考虑了层数）
    ffnParams = (denseFFNParams + moeFFNParams + nonMoeFFNParams) / L;
  }

  // LayerNorm/RMSNorm: 每层 2 个 (attention前 + FFN前)
  // LayerNorm: 2H (gamma + beta), RMSNorm: H (仅 gamma)
  const normType = model.norm_type ?? 'rmsnorm'; // 现代模型默认 RMSNorm
  const layerNormParams = normType === 'rmsnorm' ? 2 * H : 4 * H;

  // 每层总参数
  const paramsPerLayer = attentionParams + ffnParams + layerNormParams;

  // Final LayerNorm (模型输出前的最后一个 RMSNorm/LayerNorm)
  const finalLayerNormParams = normType === 'rmsnorm' ? H : 2 * H;

  // 总参数
  const totalParams = embeddingParams + L * paramsPerLayer + lmHeadParams + finalLayerNormParams;

  return totalParams;
}

/**
 * 计算 MoE 模型的等效 expert_intermediate_size
 *
 * 用于将官方参数量反推为我们公式所需的 expert_intermediate_size，
 * 以处理混合 Dense/MoE 架构的模型（如 LLaMA 4, Qwen3 MoE）
 *
 * @param model - 模型配置（不含 expert_intermediate_size 或设为临时值）
 * @param targetParamsB - 官方参数量（单位: B，如 400 表示 400B）
 * @returns 等效的 expert_intermediate_size 值
 */
export function calculateEquivalentExpertSize(
  model: LLMModelConfig,
  targetParamsB: number
): number {
  if (model.model_type !== 'moe' || !model.moe_config) {
    throw new Error('此函数仅适用于 MoE 模型');
  }

  const H = model.hidden_size;
  const L = model.num_layers;
  const V = model.vocab_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;
  const E = model.moe_config.num_experts;
  const S = model.moe_config.num_shared_experts ?? 0;

  // 1. 计算非 FFN 部分参数
  const embedding = V * H;
  const tieWordEmbeddings = model.tie_word_embeddings ?? true;
  const lmHead = tieWordEmbeddings ? 0 : H * V;
  const attention = (H * H + 2 * H * headDim * numKVHeads + H * H) * L;
  const normType = model.norm_type ?? 'rmsnorm';
  const layerNorm = (normType === 'rmsnorm' ? 2 * H : 4 * H) * L;
  const finalNorm = normType === 'rmsnorm' ? H : 2 * H;

  const P_other = embedding + lmHead + attention + layerNorm + finalNorm;

  // 2. FFN 预算
  const P_target = targetParamsB * 1e9;
  const P_ffn = P_target - P_other;
  const P_ffn_per_layer = P_ffn / L;

  // 3. 反推 expert_intermediate_size
  // P_ffn_per_layer = 3 * H * expertI * (E + S) + H * E
  // expertI = (P_ffn_per_layer - H * E) / (3 * H * (E + S))
  const router = H * E;
  const expertI = (P_ffn_per_layer - router) / (3 * H * (E + S));

  return Math.round(expertI);
}

/**
 * 计算每层参数量
 */
export function calculateParamsPerLayer(model: LLMModelConfig): {
  attention: number;
  ffn: number;
  layerNorm: number;
  total: number;
} {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  const attention = H * H + 2 * H * headDim * numKVHeads + H * H;

  let ffn = 3 * H * I;
  if (model.model_type === 'moe' && model.moe_config) {
    const numExperts = model.moe_config.num_experts;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    const expertI = model.moe_config.expert_intermediate_size ?? I;
    const firstKDense = model.moe_config.first_k_dense_replace ?? 0;
    const moeLayerFreq = model.moe_config.moe_layer_freq ?? 1;
    const numLayers = model.num_layers;

    // 计算平均每层 FFN 参数（混合 Dense 和 MoE）
    const numDenseLayers = Math.min(firstKDense, numLayers);
    const remainingLayers = numLayers - numDenseLayers;
    const numMoELayers = Math.floor(remainingLayers / moeLayerFreq);
    const numNonMoELayers = remainingLayers - numMoELayers;

    const denseFFN = 3 * H * I;
    const moeFFN = 3 * H * expertI * (numExperts + numSharedExperts) + H * numExperts;

    // 加权平均
    ffn = numLayers > 0
      ? (denseFFN * (numDenseLayers + numNonMoELayers) + moeFFN * numMoELayers) / numLayers
      : moeFFN;
  }

  // LayerNorm/RMSNorm: 每层 2 个
  const normType = model.norm_type ?? 'rmsnorm';
  const layerNorm = normType === 'rmsnorm' ? 2 * H : 4 * H;

  return {
    attention,
    ffn,
    layerNorm,
    total: attention + ffn + layerNorm,
  };
}

// ============================================
// 显存计算
// ============================================

/**
 * 计算模型权重显存 (每芯片)
 */
export function calculateModelMemory(
  model: LLMModelConfig,
  parallelism: ParallelismStrategy
): number {
  const totalParams = calculateModelParams(model);
  const bytesPerParam = getBytesPerElement(model.weight_dtype);

  // 模型按 TP 和 PP 切分
  const paramsPerChip = totalParams / parallelism.tp / parallelism.pp;

  // MoE 模型按 EP 切分专家
  let effectiveParams = paramsPerChip;
  if (model.model_type === 'moe' && model.moe_config && parallelism.ep > 1) {
    // 专家参数按EP切分，非专家参数（attention等）不变
    const layerParams = calculateParamsPerLayer(model);
    const expertRatio = layerParams.ffn / layerParams.total;
    const nonExpertParams = paramsPerChip * (1 - expertRatio);
    const expertParams = paramsPerChip * expertRatio / parallelism.ep;
    effectiveParams = nonExpertParams + expertParams;
  }

  return effectiveParams * bytesPerParam / 1e9; // GB
}

/**
 * 计算 KV Cache 显存 (每芯片)
 */
export function calculateKVCacheMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = model.hidden_size / numHeads;
  const numLayers = model.num_layers;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // KV Cache 大小 = 2 (K+V) × batch × seq × kv_heads × head_dim × layers × bytes
  // 按 TP 切分 KV heads
  const kvHeadsPerChip = Math.ceil(numKVHeads / parallelism.tp);
  // 按 PP 切分 layers
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  const kvCacheBytes =
    2 *
    inference.batch_size *
    inference.max_seq_length *
    kvHeadsPerChip *
    headDim *
    layersPerChip *
    bytesPerElement;

  return kvCacheBytes / 1e9; // GB
}

/**
 * 计算激活值显存 (每芯片)
 *
 * 激活值包括中间结果，主要在前向传播时占用
 * 对于推理，激活值相对较小
 */
export function calculateActivationMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numLayers = model.num_layers;
  const bytesPerElement = getBytesPerElement(model.activation_dtype);

  // Prefill 阶段激活值 (最大)
  const seqLen = inference.input_seq_length;
  const batch = inference.batch_size;

  // 每层激活值估算:
  // - Attention 输入: batch × seq × H
  // - Q, K, V: 3 × batch × seq × H
  // - Attention 输出: batch × seq × H
  // - FFN 中间: batch × seq × I
  // 总计约: batch × seq × (5H + I)
  const activationPerLayer = batch * seqLen * (5 * H + I) * bytesPerElement;

  // 按 PP 切分 layers，但激活值需要保留用于 PP 通信
  // 简化：只考虑当前 PP stage 的激活值
  const layersPerChip = Math.ceil(numLayers / parallelism.pp);

  // TP 会减少每个芯片的激活值（H 维度切分）
  const activationBytes = activationPerLayer * layersPerChip / parallelism.tp;

  // 推理时只需要前向传播，激活值可以逐层释放
  // 实际只需保留 1-2 层的激活值
  const effectiveActivation = activationBytes * 2 / layersPerChip;

  return effectiveActivation / 1e9; // GB
}

/**
 * 计算其他显存开销
 * 包括：CUDA context、临时缓冲区、碎片等
 */
export function calculateOverheadMemory(
  _model: LLMModelConfig,
  _inference: InferenceConfig
): number {
  // 固定开销：CUDA context 约 1GB
  const cudaContext = 1.0;

  // 临时缓冲区：约 0.5GB
  const tempBuffers = 0.5;

  // 碎片：约 10% 的显存
  const fragmentation = 0.5;

  return cudaContext + tempBuffers + fragmentation;
}

/**
 * 完整显存分析
 */
export function analyzeMemory(
  model: LLMModelConfig,
  inference: InferenceConfig,
  parallelism: ParallelismStrategy,
  chipMemoryGB: number
): MemoryAnalysis {
  const modelMemory = calculateModelMemory(model, parallelism);
  const kvCacheMemory = calculateKVCacheMemory(model, inference, parallelism);
  const activationMemory = calculateActivationMemory(model, inference, parallelism);
  const overhead = calculateOverheadMemory(model, inference);

  const totalPerChip = modelMemory + kvCacheMemory + activationMemory + overhead;
  const utilization = totalPerChip / chipMemoryGB;
  const isSufficient = totalPerChip <= chipMemoryGB * 0.95; // 留 5% 余量

  return {
    model_memory_gb: modelMemory,
    kv_cache_memory_gb: kvCacheMemory,
    activation_memory_gb: activationMemory,
    overhead_gb: overhead,
    total_per_chip_gb: totalPerChip,
    is_memory_sufficient: isSufficient,
    memory_utilization: Math.min(utilization, 1.0),
  };
}

// ============================================
// FLOPs 计算
// ============================================

/**
 * 计算单层 Transformer 的 FLOPs (Prefill 阶段)
 */
export function calculateLayerFlopsPrefill(
  model: LLMModelConfig,
  batchSize: number,
  seqLen: number
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  // Attention FLOPs:
  // QKV projection: 3 × 2 × batch × seq × H × H (考虑 GQA 时 K,V 更小)
  const qProj = 2 * batchSize * seqLen * H * H;
  const kvProj = 2 * 2 * batchSize * seqLen * H * (headDim * numKVHeads);
  const oProj = 2 * batchSize * seqLen * H * H;

  // Attention score: batch × heads × seq × seq × head_dim × 2 (Q×K + softmax×V)
  const attnScore = 2 * batchSize * numHeads * seqLen * seqLen * headDim;
  const attnOutput = 2 * batchSize * numHeads * seqLen * seqLen * headDim;

  const attentionFlops = qProj + kvProj + oProj + attnScore + attnOutput;

  // FFN FLOPs (SwiGLU):
  // gate: 2 × batch × seq × H × I
  // up: 2 × batch × seq × H × I
  // down: 2 × batch × seq × I × H
  // SiLU: batch × seq × I
  let ffnFlops = 2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I;

  // MoE: 只有部分专家被激活
  if (model.model_type === 'moe' && model.moe_config) {
    const expertsPerTok = model.moe_config.num_experts_per_tok;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    // 使用 expert_intermediate_size，如果未设置则使用 intermediate_size
    const expertI = model.moe_config.expert_intermediate_size ?? I;
    // 每个 token 激活 expertsPerTok 个专家 + 共享专家
    ffnFlops = (2 * batchSize * seqLen * H * expertI * 3 + batchSize * seqLen * expertI) *
               (expertsPerTok + numSharedExperts);
    // Router FLOPs
    ffnFlops += 2 * batchSize * seqLen * H * model.moe_config.num_experts;
  }

  // LayerNorm: 约 5 × batch × seq × H × 2
  const layerNormFlops = 10 * batchSize * seqLen * H * 2;

  return attentionFlops + ffnFlops + layerNormFlops;
}

/**
 * 计算单层 Transformer 的 FLOPs (Decode 阶段，单 token)
 */
export function calculateLayerFlopsDecode(
  model: LLMModelConfig,
  batchSize: number,
  contextLen: number
): number {
  const H = model.hidden_size;
  const I = model.intermediate_size;
  const numHeads = model.num_attention_heads;
  const numKVHeads = model.num_kv_heads;
  const headDim = H / numHeads;

  // Decode 时 seq=1，但需要与整个 context 做 attention
  const seqLen = 1;

  // QKV projection (只处理 1 个 token)
  const qProj = 2 * batchSize * seqLen * H * H;
  const kvProj = 2 * 2 * batchSize * seqLen * H * (headDim * numKVHeads);
  const oProj = 2 * batchSize * seqLen * H * H;

  // Attention: 1 个 query 与 contextLen 个 key/value
  const attnScore = 2 * batchSize * numHeads * seqLen * contextLen * headDim;
  const attnOutput = 2 * batchSize * numHeads * seqLen * contextLen * headDim;

  const attentionFlops = qProj + kvProj + oProj + attnScore + attnOutput;

  // FFN FLOPs
  let ffnFlops = 2 * batchSize * seqLen * H * I * 3 + batchSize * seqLen * I;

  if (model.model_type === 'moe' && model.moe_config) {
    const expertsPerTok = model.moe_config.num_experts_per_tok;
    const numSharedExperts = model.moe_config.num_shared_experts ?? 0;
    // 使用 expert_intermediate_size，如果未设置则使用 intermediate_size
    const expertI = model.moe_config.expert_intermediate_size ?? I;
    ffnFlops = (2 * batchSize * seqLen * H * expertI * 3 + batchSize * seqLen * expertI) *
               (expertsPerTok + numSharedExperts);
    ffnFlops += 2 * batchSize * seqLen * H * model.moe_config.num_experts;
  }

  const layerNormFlops = 10 * batchSize * seqLen * H * 2;

  return attentionFlops + ffnFlops + layerNormFlops;
}

/**
 * 计算 Prefill 阶段总 FLOPs
 */
export function calculatePrefillFlops(
  model: LLMModelConfig,
  inference: InferenceConfig
): number {
  const layerFlops = calculateLayerFlopsPrefill(
    model,
    inference.batch_size,
    inference.input_seq_length
  );
  return layerFlops * model.num_layers;
}

/**
 * 计算 Decode 阶段每 token 的 FLOPs
 */
export function calculateDecodeFlopsPerToken(
  model: LLMModelConfig,
  inference: InferenceConfig,
  currentContextLen: number
): number {
  const layerFlops = calculateLayerFlopsDecode(
    model,
    inference.batch_size,
    currentContextLen
  );
  return layerFlops * model.num_layers;
}

/**
 * 计算完整推理的总 FLOPs
 */
export function calculateTotalInferenceFlops(
  model: LLMModelConfig,
  inference: InferenceConfig
): number {
  // Prefill FLOPs
  const prefillFlops = calculatePrefillFlops(model, inference);

  // Decode FLOPs: 每个输出 token 的 FLOPs 随 context 增长
  let decodeFlops = 0;
  for (let i = 0; i < inference.output_seq_length; i++) {
    const contextLen = inference.input_seq_length + i;
    decodeFlops += calculateDecodeFlopsPerToken(model, inference, contextLen);
  }

  return prefillFlops + decodeFlops;
}

/**
 * 计算每 token 的平均 FLOPs (用于估算吞吐量)
 */
export function calculateFlopsPerToken(model: LLMModelConfig): number {
  // 简化估算：使用 context=512 的单 token FLOPs
  const inference: InferenceConfig = {
    batch_size: 1,
    input_seq_length: 512,
    output_seq_length: 1,
    max_seq_length: 513,
  };
  return calculateDecodeFlopsPerToken(model, inference, 512);
}
