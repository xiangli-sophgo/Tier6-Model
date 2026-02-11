/**
 * Benchmark 命名生成工具
 *
 * 格式: {ModelName}-S{SeqLen}-O{SeqLen}-W{x}A{y}-B{BS}
 * 示例: DeepSeek-V3.2-671B-A37B-S512-O256-W8A8-B8
 *
 * ModelName 直接使用模型配置中的 name 字段，不做解析。
 * 模型名称已经包含了参数量信息（如 671B-A37B），无需额外推断。
 */

import { LLMModelConfig, InferenceConfig } from './types'

/**
 * 格式化序列长度
 * 4096 → "4K", 1024 → "1K", 512 → "512"
 */
function formatSeqLen(len: number): string {
  if (len >= 1024 && len % 1024 === 0) {
    return `${len / 1024}K`
  }
  return String(len)
}

/**
 * 格式化数据类型为数字
 * "fp16" → "16", "bf16" → "16", "int8" → "8", "int4" → "4", "fp8" → "8"
 */
function formatDtype(dtype: string | undefined): string {
  if (!dtype) return '16'
  const match = dtype.match(/\d+/)
  return match ? match[0] : '16'
}

/**
 * 生成 Benchmark 名称
 *
 * 直接使用模型的完整名称（含参数量），拼接推理参数。
 *
 * @param model 模型配置
 * @param inference 推理配置
 * @returns Benchmark 名称，如 "DeepSeek-V3.2-671B-A37B-S512-O256-W8A8-B8"
 */
export function generateBenchmarkName(
  model: LLMModelConfig,
  inference: InferenceConfig
): string {
  const parts: string[] = []

  // 模型完整名称（已包含参数量，如 DeepSeek-V3.2-671B-A37B）
  parts.push(model.model_name)

  // S[SeqLen] - 输入序列长度
  parts.push(`S${formatSeqLen(inference.input_seq_length)}`)

  // O[SeqLen] - 输出序列长度
  parts.push(`O${formatSeqLen(inference.output_seq_length)}`)

  // W[x]A[y] - 权重和激活精度
  const weightBits = formatDtype(model.weight_dtype)
  const actBits = formatDtype(model.activation_dtype)
  parts.push(`W${weightBits}A${actBits}`)

  // B[BS] - Batch Size
  parts.push(`B${inference.batch_size}`)

  return parts.join('-')
}

/** Benchmark 各部分的解释 */
export interface BenchmarkPart {
  key: string      // 简短标识，如 "S4K"
  label: string    // 中文说明，如 "输入序列长度"
  value: string    // 详细值，如 "4096 tokens"
}

/**
 * 解析 Benchmark 各部分并生成详细解释
 */
export function parseBenchmarkParts(
  model: LLMModelConfig,
  inference: InferenceConfig
): BenchmarkPart[] {
  const parts: BenchmarkPart[] = []

  // 模型名称（完整名称，已包含参数量）
  parts.push({
    key: model.model_name,
    label: '模型',
    value: model.model_name,
  })

  // 输入序列长度
  parts.push({
    key: `S${formatSeqLen(inference.input_seq_length)}`,
    label: '输入长度',
    value: `${inference.input_seq_length} tokens`,
  })

  // 输出序列长度
  parts.push({
    key: `O${formatSeqLen(inference.output_seq_length)}`,
    label: '输出长度',
    value: `${inference.output_seq_length} tokens`,
  })

  // 数据精度
  const weightBits = formatDtype(model.weight_dtype)
  const actBits = formatDtype(model.activation_dtype)
  const weightDtypeUpper = (model.weight_dtype || 'fp16').toUpperCase()
  const actDtypeUpper = (model.activation_dtype || 'fp16').toUpperCase()
  parts.push({
    key: `W${weightBits}A${actBits}`,
    label: '精度',
    value: `权重 ${weightDtypeUpper}，激活 ${actDtypeUpper}`,
  })

  // Batch Size
  parts.push({
    key: `B${inference.batch_size}`,
    label: 'Batch Size',
    value: String(inference.batch_size),
  })

  return parts
}
