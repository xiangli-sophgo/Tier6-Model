/**
 * 参数分类器
 * 将参数按类别分层，用于 TreeSelect 显示
 */

import type { ParameterInfo } from './parameterAnalysis'

/** 参数树节点 */
export interface ParameterTreeNode {
  /** 节点唯一标识 */
  key: string
  /** 显示标题 */
  title: string
  /** 单位（叶子节点） */
  unit?: string
  /** 子节点 */
  children?: ParameterTreeNode[]
  /** 是否可选择（分类节点不可选） */
  selectable?: boolean
  /** 是否为叶子节点 */
  isLeaf?: boolean
}

/** 参数显示配置 */
interface ParamDisplayConfig {
  title: string
  unit: string
  category: 'model' | 'inference' | 'hardware' | 'parallelism' | 'topology'
}

/** 参数显示配置映射表 */
const PARAM_DISPLAY_CONFIG: Record<string, ParamDisplayConfig> = {
  // 并行策略
  'parallelism.dp': { title: 'Data Parallelism (DP)', unit: '', category: 'parallelism' },
  'parallelism.tp': { title: 'Tensor Parallelism (TP)', unit: '', category: 'parallelism' },
  'parallelism.pp': { title: 'Pipeline Parallelism (PP)', unit: '', category: 'parallelism' },
  'parallelism.ep': { title: 'Expert Parallelism (EP)', unit: '', category: 'parallelism' },
  'parallelism.sp': { title: 'Sequence Parallelism (SP)', unit: '', category: 'parallelism' },
  'parallelism.moe_tp': { title: 'MoE Tensor Parallelism', unit: '', category: 'parallelism' },

  // 推理配置
  'inference.batch_size': { title: '批次大小 (Batch Size)', unit: '', category: 'inference' },
  'inference.input_seq_length': { title: '输入序列长度', unit: 'tokens', category: 'inference' },
  'inference.output_seq_length': { title: '输出序列长度', unit: 'tokens', category: 'inference' },

  // 模型配置
  'model.hidden_size': { title: '隐藏层维度', unit: '', category: 'model' },
  'model.num_layers': { title: '层数', unit: '', category: 'model' },
  'model.num_attention_heads': { title: '注意力头数', unit: '', category: 'model' },
  'model.intermediate_size': { title: 'FFN中间层维度', unit: '', category: 'model' },

  // 硬件参数
  'hardware.compute_tflops_fp8': { title: '算力 (FP8)', unit: 'TFLOPS', category: 'hardware' },
  'hardware.compute_tflops_bf16': { title: '算力 (BF16)', unit: 'TFLOPS', category: 'hardware' },
  'hardware.memory_capacity_gb': { title: '显存容量', unit: 'GB', category: 'hardware' },
  'hardware.memory_bandwidth_gbps': { title: '显存带宽', unit: 'GB/s', category: 'hardware' },
}

/** 分类名称映射 */
const CATEGORY_LABELS: Record<string, string> = {
  model: '模型配置',
  inference: '推理配置',
  hardware: '硬件参数',
  parallelism: '并行策略',
  topology: '拓扑配置',
}

/**
 * 将参数按类别分层
 */
export function classifyParameters(
  parametersMap: Map<string, ParameterInfo>
): ParameterTreeNode[] {
  // 按类别分组
  const categorizedParams = new Map<string, ParameterTreeNode[]>()

  parametersMap.forEach((info, path) => {
    const config = PARAM_DISPLAY_CONFIG[path]
    if (!config) return // 跳过未配置的参数

    const category = config.category
    if (!categorizedParams.has(category)) {
      categorizedParams.set(category, [])
    }

    // 获取参数的取值范围信息
    const values = Array.from(info.values).sort((a, b) => a - b)
    const valueRangeInfo = values.length > 0
      ? ` [${values[0]}${values.length > 1 ? ` - ${values[values.length - 1]}` : ''}]`
      : ''

    categorizedParams.get(category)!.push({
      key: path,
      title: `${config.title}${valueRangeInfo}`,
      unit: config.unit,
      selectable: true,
      isLeaf: true,
    })
  })

  // 构建树形结构
  const tree: ParameterTreeNode[] = []
  const categoryOrder = ['model', 'inference', 'hardware', 'parallelism', 'topology']

  categoryOrder.forEach(category => {
    const params = categorizedParams.get(category)
    if (params && params.length > 0) {
      tree.push({
        key: category,
        title: CATEGORY_LABELS[category],
        children: params.sort((a, b) => a.title.localeCompare(b.title)),
        selectable: false,
        isLeaf: false,
      })
    }
  })

  return tree
}

/**
 * 根据路径获取参数的显示信息
 */
export function getParamDisplayInfo(path: string): { title: string; unit: string } | null {
  const config = PARAM_DISPLAY_CONFIG[path]
  if (!config) return null
  return { title: config.title, unit: config.unit }
}

/**
 * 搜索参数（支持中文和英文搜索）
 */
export function searchParameters(
  tree: ParameterTreeNode[],
  keyword: string
): ParameterTreeNode[] {
  if (!keyword.trim()) return tree

  const lowerKeyword = keyword.toLowerCase()
  const result: ParameterTreeNode[] = []

  tree.forEach(categoryNode => {
    const matchedChildren = categoryNode.children?.filter(child =>
      child.title.toLowerCase().includes(lowerKeyword) ||
      child.key.toLowerCase().includes(lowerKeyword)
    )

    if (matchedChildren && matchedChildren.length > 0) {
      result.push({
        ...categoryNode,
        children: matchedChildren,
      })
    }
  })

  return result
}
