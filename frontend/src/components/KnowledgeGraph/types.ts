/**
 * 知识图谱类型定义
 */

// 知识分类 (8个核心分组)
export type KnowledgeCategory =
  | 'hardware'           // 硬件 (GPU, NPU, TPU, HBM, Server, Rack, Pod)
  | 'interconnect'       // 互联 (NVLink, PCIe, InfiniBand, CXL, 拓扑)
  | 'parallel'           // 并行 (TP, SP, EP, DP, PP, Scale Up/Out)
  | 'communication'      // 通信 (AllReduce, RDMA, NCCL, MPI, 集合通信)
  | 'model'              // 模型 (Transformer, Attention, MoE, KV Cache)
  | 'inference'          // 推理 (Prefill, Decode, FlashAttention, 量化)
  | 'protocol'           // 协议 (ECC, FEC, PFC, CBFC, 编码调制)
  | 'system'             // 系统 (地址空间, 虚拟化, RAS, 事务)

// 分类颜色映射 - 按圆形布局位置使用色环配色
// 类别顺序: hardware(顶), interconnect(右上), parallel(右), inference(右下),
//          model(底), communication(左下), protocol(左), system(左上)
export const CATEGORY_COLORS: Record<KnowledgeCategory, string> = {
  hardware: '#2A9D8F',           // 青绿 hsl(168°, 58%, 39%) - 顶部
  interconnect: '#4A7DC4',       // 钴蓝 hsl(217°, 52%, 53%) - 右上
  parallel: '#7C5DBF',           // 紫罗兰 hsl(258°, 48%, 55%) - 右
  inference: '#C75B8E',          // 玫红 hsl(330°, 52%, 57%) - 右下
  model: '#C75B5B',              // 珊瑚红 hsl(0°, 52%, 57%) - 底部
  communication: '#D4845A',      // 陶土橙 hsl(22°, 58%, 59%) - 左下
  protocol: '#C4A83D',           // 芥末黄 hsl(48°, 56%, 50%) - 左
  system: '#3BA99F',             // 湖水青 hsl(173°, 50%, 45%) - 左上
}

// 分类中文名称
export const CATEGORY_NAMES: Record<KnowledgeCategory, string> = {
  hardware: '硬件',
  interconnect: '互联',
  parallel: '并行',
  communication: '通信',
  model: '模型',
  inference: '推理',
  protocol: '协议',
  system: '系统',
}

// 知识节点
export interface KnowledgeNode {
  id: string
  name: string
  fullName?: string
  definition: string
  category: KnowledgeCategory
  source?: string
  aliases?: string[]
}

// 关系类型
export type RelationType =
  | 'uses' | 'used_by' | 'used_in'
  | 'includes' | 'contains' | 'part_of' | 'composed_of'
  | 'implements' | 'implemented_by'
  | 'optimizes' | 'optimized_by' | 'optimized_for'
  | 'enables' | 'enabled_by'
  | 'extends' | 'basis_for' | 'based_on'
  | 'connects' | 'connects_via' | 'connects_in'
  | 'combines_with' | 'complements' | 'works_with'
  | 'reduces' | 'reduces_access' | 'compresses'
  | 'accelerates' | 'enhances'
  | 'trades_off' | 'constrains' | 'constrained_by' | 'limited_by'
  | 'variant_of' | 'type_of' | 'similar_to' | 'alternative_to'
  | 'feature_of' | 'requirement_of'
  | 'measured_by' | 'measures_with' | 'unit_of' | 'percentile_of'
  | 'runs_on' | 'memory_of' | 'stores_for' | 'caches'
  | 'analyzes' | 'defines' | 'generates' | 'manages' | 'builds'
  | 'decomposes_to' | 'simplifies_to' | 'normalizes' | 'maps_to'
  | 'inverse_of' | 'contrast_with' | 'equals'
  | 'succeeds' | 'precedes' | 'replaces' | 'refines'
  | 'starts_with' | 'scales_with' | 'partitioned_by' | 'translated_by'
  | 'supports' | 'affected_by' | 'determined_by' | 'format_for'
  | 'related_to' | 'belongs_to' | 'depends_on' | 'contrasts_with'

// 关系类型中文标签映射
export const RELATION_LABELS: Record<string, string> = {
  uses: '使用',
  used_by: '被使用',
  used_in: '用于',
  includes: '包含',
  contains: '包含',
  part_of: '属于',
  composed_of: '组成',
  implements: '实现',
  implemented_by: '实现方式',
  optimizes: '优化',
  optimized_by: '被优化',
  optimized_for: '优化目标',
  enables: '支持',
  enabled_by: '依赖',
  extends: '扩展',
  basis_for: '基础',
  based_on: '基于',
  connects: '连接',
  connects_via: '通过连接',
  connects_in: '连接于',
  combines_with: '组合',
  complements: '配合',
  works_with: '协作',
  reduces: '减少',
  reduces_access: '减少访问',
  compresses: '压缩',
  accelerates: '加速',
  enhances: '增强',
  trades_off: '权衡',
  constrains: '约束',
  constrained_by: '受限于',
  limited_by: '受限于',
  variant_of: '变体',
  type_of: '类型',
  similar_to: '类似',
  alternative_to: '替代',
  feature_of: '特性',
  requirement_of: '需求',
  measured_by: '衡量',
  measures_with: '测量',
  unit_of: '单位',
  percentile_of: '百分位',
  runs_on: '运行于',
  memory_of: '内存',
  stores_for: '存储',
  caches: '缓存',
  analyzes: '分析',
  defines: '定义',
  generates: '生成',
  manages: '管理',
  builds: '构建',
  decomposes_to: '分解',
  simplifies_to: '简化',
  normalizes: '归一化',
  maps_to: '映射',
  inverse_of: '逆',
  contrast_with: '对比',
  equals: '等于',
  succeeds: '后继',
  precedes: '前驱',
  replaces: '替换',
  refines: '细化',
  starts_with: '开始于',
  scales_with: '随...缩放',
  partitioned_by: '分区',
  translated_by: '转换',
  supports: '支持',
  affected_by: '受影响',
  determined_by: '决定于',
  format_for: '格式',
  related_to: '相关',
  belongs_to: '属于',
  depends_on: '依赖',
  contrasts_with: '对比',
}

// 关系样式（统一实线）
export const RELATION_STYLES: Record<string, { stroke: string }> = {
  related_to: { stroke: '#94A3B8' },
  belongs_to: { stroke: '#94A3B8' },
  depends_on: { stroke: '#94A3B8' },
  contrasts_with: { stroke: '#94A3B8' },
}

// 核心关系类型 - 只在图中显示这些重要的结构性关系
// 其他关系只在节点详情卡片中展示
export const CORE_RELATION_TYPES: Set<RelationType> = new Set([
  // 组成/包含关系
  'part_of', 'composed_of', 'contains', 'includes',
  // 类型/变体关系
  'type_of', 'variant_of',
  // 实现关系
  'implements', 'implemented_by',
  // 继承/扩展关系
  'based_on', 'basis_for', 'extends',
  // 核心依赖
  'enables', 'runs_on', 'memory_of',
  // 使用关系
  'uses',
])

// 每个节点在图中最多显示的连接数
export const MAX_EDGES_PER_NODE = 8

// 知识关系
export interface KnowledgeRelation {
  source: string
  target: string
  type: RelationType
  description?: string
}

// 知识图谱数据
export interface KnowledgeGraphData {
  nodes: KnowledgeNode[]
  relations: KnowledgeRelation[]
  metadata?: {
    version: string
    nodeCount: number
    relationCount: number
  }
}

// 力导向节点（扩展）
export interface ForceKnowledgeNode extends KnowledgeNode {
  x: number
  y: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
  index?: number
}
