/**
 * 参数遍历功能 - 参数提取逻辑（重构版本）
 * 递归提取所有配置中的数值参数
 */

import type { SweepableParameter } from './sweepTypes'
import type {
  LLMModelConfig,
  InferenceConfig,
  ParallelismStrategy,
} from '../../../../utils/llmDeployment/types'
import type { HardwareParams } from '../../shared'
import type { SavedConfig } from '../../../../utils/storage'

// ============================================
// 参数元数据映射 - 提供友好的中文标签和单位
// ============================================

interface ParameterMetadata {
  label: string
  unit?: string
  defaultRange?: { min: number; max: number; step: number }
}

/**
 * 参数路径到元数据的映射
 */
const PARAMETER_METADATA: Record<string, ParameterMetadata> = {
  // === 模型配置 ===
  'model.hidden_size': { label: '隐藏层维度', defaultRange: { min: 2048, max: 16384, step: 1024 } },
  'model.num_layers': { label: '层数', defaultRange: { min: 12, max: 128, step: 4 } },
  'model.num_attention_heads': { label: '注意力头数', defaultRange: { min: 8, max: 128, step: 8 } },
  'model.num_kv_heads': { label: 'KV头数', defaultRange: { min: 8, max: 128, step: 8 } },
  'model.intermediate_size': { label: 'FFN中间维度', defaultRange: { min: 4096, max: 65536, step: 4096 } },
  'model.moe_config.num_experts': { label: '专家数量', defaultRange: { min: 4, max: 256, step: 4 } },
  'model.moe_config.num_experts_per_tok': { label: '激活专家数', defaultRange: { min: 1, max: 8, step: 1 } },
  'model.moe_config.num_shared_experts': { label: '共享专家数', defaultRange: { min: 0, max: 16, step: 1 } },
  'model.vocab_size': { label: '词表大小', defaultRange: { min: 10000, max: 200000, step: 1000 } },
  'model.max_position_embeddings': { label: '最大位置编码', defaultRange: { min: 2048, max: 131072, step: 2048 } },
  'model.moe_config.expert_capacity_factor': { label: 'MoE容量因子', defaultRange: { min: 1.0, max: 2.0, step: 0.1 } },
  'model.mla_config.kv_lora_rank': { label: 'MLA KV LoRA秩', defaultRange: { min: 256, max: 2048, step: 128 } },
  'model.mla_config.q_lora_rank': { label: 'MLA Q LoRA秩', defaultRange: { min: 512, max: 4096, step: 256 } },
  'model.mla_config.qk_rope_dim': { label: 'MLA QK RoPE维度', defaultRange: { min: 32, max: 256, step: 32 } },
  'model.mla_config.qk_nope_dim': { label: 'MLA QK NoPE维度', defaultRange: { min: 64, max: 512, step: 64 } },
  'model.mla_config.v_head_dim': { label: 'MLA V头维度', defaultRange: { min: 64, max: 512, step: 64 } },

  // === 推理配置 ===
  'inference.batch_size': { label: 'Batch Size', defaultRange: { min: 1, max: 256, step: 1 } },
  'inference.input_seq_length': { label: '输入序列长度', defaultRange: { min: 128, max: 32768, step: 128 } },
  'inference.output_seq_length': { label: '输出序列长度', defaultRange: { min: 64, max: 8192, step: 64 } },
  'inference.num_micro_batches': { label: '微批次数', defaultRange: { min: 1, max: 32, step: 1 } },

  // === 并行策略 ===
  'parallelism.dp': { label: '数据并行 (DP)', defaultRange: { min: 1, max: 128, step: 1 } },
  'parallelism.tp': { label: '张量并行 (TP)', defaultRange: { min: 1, max: 16, step: 1 } },
  'parallelism.pp': { label: '流水线并行 (PP)', defaultRange: { min: 1, max: 32, step: 1 } },
  'parallelism.ep': { label: '专家并行 (EP)', defaultRange: { min: 1, max: 64, step: 1 } },
  'parallelism.sp': { label: '序列并行 (SP)', defaultRange: { min: 1, max: 16, step: 1 } },

  // === 硬件参数（通用模式匹配）===
  // 计算相关
  'compute_tflops_fp8': { label: 'FP8 算力', unit: 'TFLOPS', defaultRange: { min: 64, max: 3000, step: 64 } },
  'compute_tflops_bf16': { label: 'BF16 算力', unit: 'TFLOPS', defaultRange: { min: 64, max: 2048, step: 64 } },
  'compute_tflops_fp16': { label: 'FP16 算力', unit: 'TFLOPS', defaultRange: { min: 64, max: 2048, step: 64 } },
  // 显存相关
  'memory_capacity_gb': { label: '显存容量', unit: 'GB', defaultRange: { min: 16, max: 256, step: 16 } },
  'memory_bandwidth_gbps': { label: '显存带宽', unit: 'GB/s', defaultRange: { min: 500, max: 20000, step: 100 } },
  'memory_bandwidth_utilization': { label: '显存带宽利用率', defaultRange: { min: 0.5, max: 1.0, step: 0.05 } },
  // 本地内存
  'lmem_capacity_mb': { label: 'LMEM容量', unit: 'MB', defaultRange: { min: 16, max: 256, step: 16 } },
  'lmem_bandwidth_gbps': { label: 'LMEM带宽', unit: 'GB/s', defaultRange: { min: 1000, max: 10000, step: 500 } },
  // 微架构参数
  'cube_m': { label: 'Cube M维度', defaultRange: { min: 8, max: 32, step: 8 } },
  'cube_k': { label: 'Cube K维度', defaultRange: { min: 8, max: 64, step: 8 } },
  'cube_n': { label: 'Cube N维度', defaultRange: { min: 8, max: 32, step: 8 } },
  'sram_size_kb': { label: 'SRAM大小', unit: 'KB', defaultRange: { min: 512, max: 8192, step: 512 } },
  'sram_utilization': { label: 'SRAM利用率', defaultRange: { min: 0.3, max: 0.9, step: 0.05 } },
  'lane_num': { label: 'Lane数量', defaultRange: { min: 8, max: 32, step: 8 } },
  'compute_dma_overlap_rate': { label: '计算DMA重叠率', defaultRange: { min: 0.5, max: 1.0, step: 0.1 } },
  'num_cores': { label: '核心数', defaultRange: { min: 32, max: 128, step: 8 } },

  // === 互联参数 ===
  'bandwidth_gbps': { label: '带宽', unit: 'GB/s', defaultRange: { min: 50, max: 1000, step: 50 } },
  'latency_us': { label: '延迟', unit: 'μs', defaultRange: { min: 0.1, max: 10, step: 0.1 } },

  // === 通信延迟配置 ===
  'interconnect.comm_params.rtt_tp_us': { label: 'TP RTT', unit: 'μs', defaultRange: { min: 0.1, max: 2, step: 0.1 } },
  'interconnect.comm_params.rtt_ep_us': { label: 'EP RTT', unit: 'μs', defaultRange: { min: 0.1, max: 2, step: 0.1 } },
  'interconnect.comm_params.bandwidth_utilization': { label: '带宽利用率', defaultRange: { min: 0.7, max: 1.0, step: 0.05 } },
  'interconnect.comm_params.sync_latency_us': { label: '同步延迟', unit: 'μs', defaultRange: { min: 0, max: 5, step: 0.1 } },
  'interconnect.comm_params.switch_delay_us': { label: '交换机延迟', unit: 'μs', defaultRange: { min: 0.5, max: 5, step: 0.5 } },
  'interconnect.comm_params.cable_delay_us': { label: '线缆延迟', unit: 'μs', defaultRange: { min: 0.01, max: 0.5, step: 0.01 } },
  'interconnect.comm_params.memory_read_latency_us': { label: '显存读延迟', unit: 'μs', defaultRange: { min: 0.05, max: 1, step: 0.05 } },
  'interconnect.comm_params.memory_write_latency_us': { label: '显存写延迟', unit: 'μs', defaultRange: { min: 0.01, max: 0.5, step: 0.01 } },
  'interconnect.comm_params.noc_latency_us': { label: 'NoC延迟', unit: 'μs', defaultRange: { min: 0.01, max: 0.5, step: 0.01 } },
  'interconnect.comm_params.die_to_die_latency_us': { label: 'Die2Die延迟', unit: 'μs', defaultRange: { min: 0.01, max: 0.5, step: 0.01 } },

  // === 拓扑配置 ===
  'topology.pod_count': { label: 'Pod数量', defaultRange: { min: 1, max: 16, step: 1 } },
  'topology.racks_per_pod': { label: '每Pod机柜数', defaultRange: { min: 1, max: 32, step: 1 } },

  // === 网络配置 ===
  'intra_board_bandwidth_gbps': { label: 'Board内带宽', unit: 'GB/s', defaultRange: { min: 100, max: 2000, step: 100 } },
  'inter_board_bandwidth_gbps': { label: 'Board间带宽', unit: 'GB/s', defaultRange: { min: 50, max: 1000, step: 50 } },
  'intra_board_latency_us': { label: 'Board内延迟', unit: 'μs', defaultRange: { min: 0.1, max: 5, step: 0.1 } },
  'inter_board_latency_us': { label: 'Board间延迟', unit: 'μs', defaultRange: { min: 1, max: 20, step: 1 } },
}

// ============================================
// 工具函数
// ============================================

/**
 * 判断字段是否应跳过（非数值类型或内部字段）
 */
function shouldSkipField(key: string, value: any, fullPath: string): boolean {
  // 跳过非数值类型
  if (typeof value !== 'number') return true

  // 跳过内部ID字段
  if (key === 'id' || key.endsWith('_id')) return true

  // 跳过时间戳
  if (key.includes('timestamp') || key.includes('_at')) return true

  // 跳过数组索引（如 boards[0].count）
  if (/\[\d+\]/.test(fullPath)) return true



  return false
}

/**
 * 获取参数的友好标签和单位
 */
function getParameterLabel(fullPath: string, key: string): { label: string; unit?: string } {
  // 1. 精确匹配完整路径
  const exactMatch = PARAMETER_METADATA[fullPath]
  if (exactMatch) {
    return { label: exactMatch.label, unit: exactMatch.unit }
  }

  // 2. 特殊处理：互联拓扑参数（添加层级前缀）
  if (fullPath.includes('interconnect.')) {
    const interconnectLevelMatch = fullPath.match(/interconnect\.(c2c|b2b|r2r|p2p)\./)
    if (interconnectLevelMatch) {
      const level = interconnectLevelMatch[1].toUpperCase() // c2c → C2C
      const keyMatch = PARAMETER_METADATA[key]
      if (keyMatch) {
        return { label: `${level} ${keyMatch.label}`, unit: keyMatch.unit }
      }
      // 如果没有找到key的映射，使用默认名称
      const defaultLabels: Record<string, string> = {
        'bandwidth_gbps': '带宽',
        'latency_us': '延迟',
      }
      const defaultLabel = defaultLabels[key] || key
      const defaultUnits: Record<string, string> = {
        'bandwidth_gbps': 'GB/s',
        'latency_us': 'μs',
      }
      const defaultUnit = defaultUnits[key]
      return { label: `${level} ${defaultLabel}`, unit: defaultUnit }
    }
  }

  // 3. 匹配路径后缀（如 "*.compute_tflops_bf16"）
  const pathParts = fullPath.split('.')
  for (let i = pathParts.length - 1; i >= 0; i--) {
    const suffix = pathParts.slice(i).join('.')
    const suffixMatch = PARAMETER_METADATA[suffix]
    if (suffixMatch) {
      return { label: suffixMatch.label, unit: suffixMatch.unit }
    }
  }

  // 4. 匹配最后一个key（如 "bandwidth_gbps"）
  const keyMatch = PARAMETER_METADATA[key]
  if (keyMatch) {
    return { label: keyMatch.label, unit: keyMatch.unit }
  }

  // 5. 默认：使用原始key
  return { label: key }
}

/**
 * 获取参数的默认范围
 */
function getDefaultRange(fullPath: string, key: string, currentValue: number): { min: number; max: number; step: number } {
  // 1. 精确匹配
  const exactMatch = PARAMETER_METADATA[fullPath]
  if (exactMatch?.defaultRange) {
    return exactMatch.defaultRange
  }

  // 2. 匹配路径后缀
  const pathParts = fullPath.split('.')
  for (let i = pathParts.length - 1; i >= 0; i--) {
    const suffix = pathParts.slice(i).join('.')
    const suffixMatch = PARAMETER_METADATA[suffix]
    if (suffixMatch?.defaultRange) {
      return suffixMatch.defaultRange
    }
  }

  // 3. 匹配key
  const keyMatch = PARAMETER_METADATA[key]
  if (keyMatch?.defaultRange) {
    return keyMatch.defaultRange
  }

  // 4. 默认：根据当前值推断
  if (currentValue === 0) {
    return { min: 0, max: 100, step: 1 }
  }

  const magnitude = Math.pow(10, Math.floor(Math.log10(Math.abs(currentValue))))
  return {
    min: Math.max(0, currentValue - magnitude * 10),
    max: currentValue + magnitude * 10,
    step: magnitude,
  }
}

/**
 * 判断参数类别
 */
function getCategoryFromPath(fullPath: string): SweepableParameter['category'] {
  if (fullPath.startsWith('model.')) return 'model'
  if (fullPath.startsWith('inference.')) return 'inference'
  if (fullPath.startsWith('parallelism.')) return 'parallelism'
  if (fullPath.startsWith('topology.') || fullPath.startsWith('hardware.') || fullPath.startsWith('interconnect.comm_params.') || fullPath.startsWith('comm_latency_config.')) {
    return 'topology'
  }
  return 'hardware'
}

// ============================================
// 递归参数提取
// ============================================

/**
 * 递归遍历对象，提取所有数值参数
 */
function traverseObject(
  obj: any,
  prefix: string,
  category: SweepableParameter['category'],
  results: SweepableParameter[]
): void {
  if (obj === null || obj === undefined) return

  // 处理数组（跳过，因为数组索引不适合遍历）
  if (Array.isArray(obj)) return

  // 处理普通对象
  if (typeof obj === 'object') {
    for (const [key, value] of Object.entries(obj)) {
      const fullPath = prefix ? `${prefix}.${key}` : key

      // 跳过不需要的字段
      if (shouldSkipField(key, value, fullPath)) {
        // 如果是对象，继续递归（可能包含数值字段）
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          traverseObject(value, fullPath, category, results)
        }
        continue
      }

      // 提取数值参数
      if (typeof value === 'number') {
        const { label, unit } = getParameterLabel(fullPath, key)
        const defaultRange = getDefaultRange(fullPath, key, value)
        const paramCategory = getCategoryFromPath(fullPath)

        results.push({
          key: fullPath,
          label,
          currentValue: value,
          defaultRange,
          unit,
          category: paramCategory,
        })
      }

      // 递归处理嵌套对象
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        traverseObject(value, fullPath, category, results)
      }
    }
  }
}

/**
 * 从配置中提取所有可遍历参数（重构版本 - 支持拓扑配置）
 */
export function extractSweepableParameters(
  modelConfig: LLMModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareParams: HardwareParams | null,
  parallelism: ParallelismStrategy,
  topologyConfig?: SavedConfig | null
): SweepableParameter[] {
  const params: SweepableParameter[] = []

  // ========== 模型配置参数 ==========
  traverseObject(modelConfig, 'model', 'model', params)

  // ========== 推理配置参数 ==========
  traverseObject(inferenceConfig, 'inference', 'inference', params)

  // ========== 并行策略参数 ==========
  traverseObject(parallelism, 'parallelism', 'parallelism', params)

  // ========== 硬件配置参数 ==========
  if (hardwareParams) {
    traverseObject(hardwareParams, 'hardware', 'hardware', params)
  }

  // ========== 拓扑配置参数 ==========
  if (topologyConfig) {
    // 提取顶层参数
    if (typeof topologyConfig.pod_count === 'number') {
      params.push({
        key: 'topology.pod_count',
        label: 'Pod数量',
        currentValue: topologyConfig.pod_count,
        defaultRange: { min: 1, max: 16, step: 1 },
        category: 'topology',
      })
    }

    if (typeof topologyConfig.racks_per_pod === 'number') {
      params.push({
        key: 'topology.racks_per_pod',
        label: '每Pod机柜数',
        currentValue: topologyConfig.racks_per_pod,
        defaultRange: { min: 1, max: 32, step: 1 },
        category: 'topology',
      })
    }

    // 提取芯片参数（新格式: chips, 旧格式: hardware_params）
    const topoChips = topologyConfig.chips || (topologyConfig as any).hardware_params?.chips
    if (topoChips) {
      traverseObject(topoChips, 'topology.chips', 'topology', params)
    }

    // 提取互联链路参数（新格式: interconnect.links, 旧格式: hardware_params.interconnect）
    const topoInterconnectLinks = topologyConfig.interconnect?.links || (topologyConfig as any).hardware_params?.interconnect
    if (topoInterconnectLinks) {
      traverseObject(topoInterconnectLinks, 'topology.interconnect.links', 'topology', params)
    }

    // 提取通信参数（新格式: interconnect.comm_params, 旧格式: comm_latency_config）
    const topoCommParams = topologyConfig.interconnect?.comm_params || (topologyConfig as any).comm_latency_config
    if (topoCommParams) {
      traverseObject(topoCommParams, 'interconnect.comm_params', 'topology', params)
    }

    // 提取网络配置
    if (topologyConfig.network_config) {
      traverseObject(topologyConfig.network_config, 'topology.network_config', 'topology', params)
    }
  }

  return params
}

/**
 * 根据参数路径从配置中获取值
 */
export function getValueByPath(obj: any, path: string): number | undefined {
  const keys = path.split('.')
  let current = obj
  for (const key of keys) {
    if (current === null || current === undefined) return undefined
    current = current[key]
  }
  return typeof current === 'number' ? current : undefined
}

/**
 * 根据参数路径设置值（返回新对象）
 */
export function setValueByPath(obj: any, path: string, value: number): any {
  const keys = path.split('.')
  const newObj = JSON.parse(JSON.stringify(obj)) // 深拷贝
  let current = newObj
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (current[key] === undefined || current[key] === null) {
      current[key] = {}
    }
    current = current[key]
  }
  current[keys[keys.length - 1]] = value
  return newObj
}
