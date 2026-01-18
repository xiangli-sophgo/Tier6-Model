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

// 合并所有节点
export const nodes: KnowledgeNode[] = [
  ...parallelNodes,
  ...modelNodes,
  ...inferenceNodes,
  ...hardwareNodes,
  ...interconnectNodes,
  ...communicationNodes,
  ...protocolNodes,
  ...systemNodes,
] as KnowledgeNode[]

// 导出关系
export const knowledgeRelations: KnowledgeRelation[] = relations as KnowledgeRelation[]

// 导出完整数据结构（兼容原有接口）
export const knowledgeGraphData = {
  nodes,
  relations: knowledgeRelations,
}

export default knowledgeGraphData
