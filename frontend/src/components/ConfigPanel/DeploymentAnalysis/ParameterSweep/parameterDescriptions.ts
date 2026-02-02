/**
 * 参数说明映射 - 用于 Tooltip 显示
 */

export const PARAMETER_DESCRIPTIONS: Record<string, string> = {
  // === 模型配置 ===
  'model.hidden_size': '模型隐藏层维度，决定模型宽度和参数量',
  'model.num_layers': '模型层数，决定模型深度',
  'model.num_attention_heads': 'Attention头数，影响并行度要求',
  'model.num_kv_heads': 'KV头数，用于GQA/MQA架构',
  'model.intermediate_size': 'FFN中间层维度，通常为hidden_size的4倍',
  'model.vocab_size': '词表大小，影响Embedding层参数量',
  'model.max_position_embeddings': '最大位置编码长度，限制输入序列长度',

  // MoE
  'model.moe_config.num_experts': 'MoE专家总数',
  'model.moe_config.num_experts_per_tok': '每个token激活的专家数',
  'model.moe_config.num_shared_experts': '共享专家数（始终激活）',
  'model.moe_config.expert_capacity_factor': 'MoE专家容量因子，影响负载均衡',

  // MLA
  'model.mla_config.kv_lora_rank': 'MLA KV压缩秩，越小显存越少',
  'model.mla_config.q_lora_rank': 'MLA Query LoRA秩',
  'model.mla_config.qk_rope_dim': 'MLA QK RoPE维度',
  'model.mla_config.qk_nope_dim': 'MLA QK无RoPE维度',
  'model.mla_config.v_head_dim': 'MLA Value头维度',

  // === 推理配置 ===
  'inference.batch_size': '批次大小，影响吞吐量和延迟',
  'inference.input_seq_length': 'Prefill阶段输入长度（Prompt长度）',
  'inference.output_seq_length': 'Decode阶段输出长度（生成Token数）',
  'inference.num_micro_batches': 'Pipeline并行微批次数，影响流水线效率',

  // === 并行策略 ===
  'parallelism.dp': '数据并行度，独立副本数量',
  'parallelism.tp': '张量并行度，模型层内切分',
  'parallelism.pp': '流水线并行度，模型层间切分',
  'parallelism.ep': '专家并行度（MoE专用），专家分布数量',
  'parallelism.sp': '序列并行度，序列维度切分',

  // === 硬件参数 ===
  'compute_tflops_fp8': 'FP8精度峰值算力',
  'compute_tflops_bf16': 'BF16精度峰值算力',
  'compute_tflops_fp16': 'FP16精度峰值算力',
  'memory_capacity_gb': 'HBM显存容量',
  'memory_bandwidth_gbps': 'HBM显存带宽',
  'memory_bandwidth_utilization': '显存带宽利用率（实际/峰值）',
  'lmem_capacity_mb': '本地内存（LMEM）容量',
  'lmem_bandwidth_gbps': '本地内存带宽',
  'cube_m': 'Cube计算单元M维度（GEMM参数）',
  'cube_k': 'Cube计算单元K维度（GEMM参数）',
  'cube_n': 'Cube计算单元N维度（GEMM参数）',
  'sram_size_kb': '片上SRAM大小',
  'sram_utilization': 'SRAM利用率',
  'lane_num': 'DMA Lane数量，影响数据传输带宽',
  'compute_dma_overlap_rate': '计算与DMA重叠率（隐藏通信延迟）',
  'num_cores': '计算核心数',

  // === 互联参数 ===
  'bandwidth_gbps': '互联带宽（双向）',
  'latency_us': '互联延迟（单向）',

  // === 通信延迟配置 ===
  'comm_latency_config.rtt_tp_us': '张量并行通信往返时延',
  'comm_latency_config.rtt_ep_us': '专家并行通信往返时延',
  'comm_latency_config.bandwidth_utilization': '网络带宽利用率（考虑拥塞）',
  'comm_latency_config.sync_latency_us': 'AllReduce同步延迟',
  'comm_latency_config.switch_delay_us': '交换机转发延迟',
  'comm_latency_config.cable_delay_us': '线缆传输延迟',
  'comm_latency_config.memory_read_latency_us': 'HBM读取延迟',
  'comm_latency_config.memory_write_latency_us': 'HBM写入延迟',
  'comm_latency_config.noc_latency_us': '片上网络（NoC）延迟',
  'comm_latency_config.die_to_die_latency_us': 'Die间互联延迟',

  // === 拓扑配置 ===
  'topology.pod_count': 'Pod数量（集群规模）',
  'topology.racks_per_pod': '每个Pod的机柜数',

  // === 网络配置 ===
  'intra_board_bandwidth_gbps': 'Board内芯片互联带宽（如NVLink）',
  'inter_board_bandwidth_gbps': 'Board间互联带宽（如InfiniBand）',
  'intra_board_latency_us': 'Board内芯片互联延迟',
  'inter_board_latency_us': 'Board间互联延迟',
}

/**
 * 获取参数描述（支持前缀匹配）
 */
export function getParameterDescription(key: string): string | undefined {
  // 1. 精确匹配
  if (PARAMETER_DESCRIPTIONS[key]) {
    return PARAMETER_DESCRIPTIONS[key]
  }

  // 2. 匹配后缀（如 "hardware.chips.SG2262.compute_tflops_bf16" → "compute_tflops_bf16"）
  const parts = key.split('.')
  for (let i = parts.length - 1; i >= 0; i--) {
    const suffix = parts.slice(i).join('.')
    if (PARAMETER_DESCRIPTIONS[suffix]) {
      return PARAMETER_DESCRIPTIONS[suffix]
    }
  }

  return undefined
}
