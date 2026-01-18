/**
 * Benchmark 命名生成工具
 *
 * 格式: [Model]-[Size]-S[SeqLen]-O[SeqLen]-W[x]A[y]-B[BS]
 * 示例: DeepSeek-V3-S4K-O512-W8A16-B64
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
 * 解析模型名称，提取模型名和参数规模
 * "DeepSeek-V3-671B" → { name: "DeepSeek-V3", size: "671B" }
 * "Llama-3.1-70B-Instruct" → { name: "Llama-3.1", size: "70B" }
 */
function parseModelName(modelName: string): { name: string; size: string } {
  // 匹配参数规模模式: 数字+B/b (如 70B, 671B, 7b)
  const sizeMatch = modelName.match(/(\d+\.?\d*)[Bb]/)

  if (sizeMatch) {
    const size = sizeMatch[1] + 'B'
    // 移除参数规模部分，清理多余的连字符
    const name = modelName
      .replace(/[-_]?\d+\.?\d*[Bb][-_]?/, '')
      .replace(/-+$/, '')
      .replace(/^-+/, '')
      .replace(/-Instruct|-Chat|-Base/i, '')
      .trim()
    return { name, size }
  }

  // 没有匹配到参数规模，返回原名称
  return { name: modelName, size: '' }
}

/**
 * 生成 Benchmark 名称
 *
 * @param model 模型配置
 * @param inference 推理配置
 * @returns Benchmark 名称，如 "DeepSeek-V3-S4K-O512-W8A16-B64"
 */
export function generateBenchmarkName(
  model: LLMModelConfig,
  inference: InferenceConfig
): string {
  const { name, size } = parseModelName(model.model_name)

  const parts: string[] = []

  // [Model]-[Size]
  if (size) {
    parts.push(`${name}-${size}`)
  } else {
    parts.push(name)
  }

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
  const { name, size } = parseModelName(model.model_name)
  const parts: BenchmarkPart[] = []

  // 模型名称
  parts.push({
    key: size ? `${name}-${size}` : name,
    label: '模型',
    value: size ? `${name}，参数 ${size}` : name,
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
