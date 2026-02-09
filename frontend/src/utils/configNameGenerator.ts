/**
 * 配置名称生成工具函数
 *
 * 用于根据配置内容生成标准化的名称
 */

import { LLMModelConfig, InferenceConfig } from './llmDeployment/types'

/**
 * 格式化序列长度
 * - >= 1024 且整除 1024 时显示为 xK
 * - 否则直接显示数字
 */
function formatSeqLen(len: number): string {
  if (len >= 1024 && len % 1024 === 0) return `${len / 1024}K`
  return String(len)
}

/**
 * 获取数据类型的位数
 */
function getDtypeBits(dtype: string): number {
  const bitsMap: Record<string, number> = {
    'fp32': 32,
    'fp16': 16,
    'bf16': 16,
    'fp8': 8,
    'int8': 8,
    'int4': 4,
  }
  return bitsMap[dtype] || 16
}

/**
 * 解析模型名称，提取名称和参数规模
 */
function parseModelName(modelName: string): { name: string; size: string } {
  const sizeMatch = modelName.match(/(\d+\.?\d*)[BMK]/i)
  if (sizeMatch) {
    const size = sizeMatch[1] + sizeMatch[0].slice(-1).toUpperCase()
    const name = modelName
      .replace(/[-_]?\d+\.?\d*[BMK][-_]?/i, '')
      .replace(/-+$/, '')
      .replace(/^-+/, '')
      .replace(/-Instruct|-Chat|-Base/i, '')
      .trim()
    return { name, size }
  }
  return { name: modelName, size: '' }
}

/**
 * 生成 Benchmark 名称
 *
 * 格式: {ModelName}-S{InputSeq}-O{OutputSeq}-W{WeightBits}A{ActivationBits}-B{BatchSize}
 * 示例: DeepSeek-V3-671B-S32K-O1K-W8A8-B1
 *
 * 只有修改这些字段才更新名称：
 * - model.model_name
 * - model.weight_dtype
 * - model.activation_dtype
 * - inference.input_seq_length
 * - inference.output_seq_length
 * - inference.batch_size
 */
export function generateBenchmarkName(
  modelConfig: LLMModelConfig,
  inferenceConfig: InferenceConfig
): string {
  const { name, size } = parseModelName(modelConfig.model_name)
  const seqIn = formatSeqLen(inferenceConfig.input_seq_length)
  const seqOut = formatSeqLen(inferenceConfig.output_seq_length)
  const wBits = getDtypeBits(modelConfig.weight_dtype)
  const aBits = getDtypeBits(modelConfig.activation_dtype)

  const parts = [
    size ? `${name}-${size}` : name,
    `S${seqIn}`,
    `O${seqOut}`,
    `W${wBits}A${aBits}`,
    `B${inferenceConfig.batch_size}`,
  ]
  return parts.join('-')
}

/**
 * 生成 Topology 名称
 *
 * 格式: P{Pods}-R{Racks}-B{TotalBoards}-C{TotalChips}
 * 示例: P1-R4-B32-C256
 *
 * 从 grouped_pods 格式 (pods[].racks[].boards[].chips[]) 计算统计量。
 */
export function generateTopologyName(topology: { pods?: Array<{ count?: number; racks?: Array<{ count?: number; boards?: Array<{ count?: number; chips?: Array<{ count?: number }> }> }> }> }): string {
  const pods = topology.pods || []
  let totalPods = 0
  let totalRacks = 0
  let totalBoards = 0
  let totalChips = 0

  for (const podGroup of pods) {
    const pc = podGroup.count ?? 1
    totalPods += pc
    for (const rackGroup of podGroup.racks ?? []) {
      const rc = rackGroup.count ?? 1
      totalRacks += pc * rc
      for (const board of rackGroup.boards ?? []) {
        const bc = board.count ?? 1
        totalBoards += pc * rc * bc
        for (const chip of board.chips ?? []) {
          totalChips += pc * rc * bc * (chip.count ?? 1)
        }
      }
    }
  }

  return `P${totalPods}-R${totalRacks}-B${totalBoards}-C${totalChips}`
}
