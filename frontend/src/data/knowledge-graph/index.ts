// 知识图谱数据加载器 - 按分类拆分便于维护和扩展

import parallelNodes from './parallel.json'
import modelNodes from './model.json'
import inferenceNodes from './inference.json'
import hardwareNodes from './hardware.json'
import interconnectNodes from './interconnect.json'
import communicationNodes from './communication.json'
import protocolNodes from './protocol.json'
import systemNodes from './system.json'
import relations from './relations.json'

import type { KnowledgeNode, KnowledgeRelation } from '../../components/KnowledgeGraph/types'

// ============================================
// 数据验证和合并
// ============================================

// 验证节点数组
const validateNodes = (nodes: unknown[]): KnowledgeNode[] => {
  if (!Array.isArray(nodes)) {
    console.warn('Invalid nodes type:', typeof nodes)
    return []
  }
  return nodes.filter(node => {
    if (!node || typeof node !== 'object') return false
    const n = node as Record<string, unknown>
    return Boolean(n.id && n.name && n.definition && n.category)
  }) as KnowledgeNode[]
}

// 验证关系数组
const validateRelations = (rels: unknown[]): KnowledgeRelation[] => {
  if (!Array.isArray(rels)) {
    console.warn('Invalid relations type:', typeof rels)
    return []
  }
  return rels.filter(rel => {
    if (!rel || typeof rel !== 'object') return false
    const r = rel as Record<string, unknown>
    return Boolean(r.source && r.target && r.type)
  }) as KnowledgeRelation[]
}

// 合并所有节点
const allNodeArrays = [
  parallelNodes,
  modelNodes,
  inferenceNodes,
  hardwareNodes,
  interconnectNodes,
  communicationNodes,
  protocolNodes,
  systemNodes,
]

export const nodes: KnowledgeNode[] = allNodeArrays
  .map(arr => validateNodes(arr))
  .flat()

// 导出关系
export const knowledgeRelations: KnowledgeRelation[] = validateRelations(relations as unknown[])

// 数据完整性检查
if (typeof window !== 'undefined') {
  // 仅在浏览器环境中执行
  const nodeCount = nodes.length
  const relationCount = knowledgeRelations.length


}

// 导出完整数据结构（兼容原有接口）
export const knowledgeGraphData = {
  nodes,
  relations: knowledgeRelations,
}

export default knowledgeGraphData
