import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ManualConnection,
  ConnectionMode,
  HierarchyLevel,
  LayoutType,
  MultiLevelViewOptions,
} from '../../types'
import { TopologyTrafficResult } from '../../utils/llmDeployment/types'

// 重新导出供子组件使用的类型
export type { HierarchyLevel, LayoutType, MultiLevelViewOptions, ManualConnection }
export type { TopologyTrafficResult }

// 根据板卡U高度区分颜色
export const BOARD_U_COLORS: Record<number, string> = {
  1: '#13c2c2',  // 1U - 青色
  2: '#722ed1',  // 2U - 紫色
  4: '#eb2f96',  // 4U - 洋红色
}

// Switch面板布局配置
export const SWITCH_PANEL_CONFIG = {
  minWidth: 100,           // 最小面板宽度
  maxWidth: 180,           // 最大面板宽度
  nodeWidth: 60,           // Switch节点宽度
  nodeHeight: 24,          // Switch节点高度
  layerGap: 60,            // 层间垂直间距
  nodeGap: 15,             // 同层节点间距
  padding: 15,             // 面板内边距
  panelGap: 20,            // 面板与主视图之间的间距
}

// Switch层级顺序（从下到上渲染）
export const SWITCH_LAYER_ORDER = ['leaf', 'spine', 'core']

// Switch面板布局结果
export interface SwitchPanelLayoutResult {
  panelWidth: number       // Switch面板宽度
  switchNodes: Node[]      // 布局后的Switch节点
  switchEdges: Edge[]      // Switch之间的连接
  deviceAreaOffset: number // 设备区域的X偏移量
}

export interface BreadcrumbItem {
  level: string
  id: string
  label: string
}

// 节点详细信息
export interface NodeDetail {
  id: string
  label: string
  type: string
  subType?: string
  connections: { id: string; label: string; bandwidth?: number; latency?: number }[]
  portInfo?: { uplink: number; downlink: number; inter: number }
}

// 连接详细信息
export interface LinkDetail {
  id: string  // source-target 格式
  sourceId: string
  sourceLabel: string
  sourceType: string
  targetId: string
  targetLabel: string
  targetType: string
  bandwidth?: number
  latency?: number
  isManual?: boolean
}

export interface TopologyGraphProps {
  visible: boolean
  onClose: () => void
  topology: HierarchicalTopology | null
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod?: PodConfig | null
  currentRack?: RackConfig | null
  currentBoard?: BoardConfig | null
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  onNodeClick?: (nodeDetail: NodeDetail | null) => void
  onLinkClick?: (linkDetail: LinkDetail | null) => void
  selectedNodeId?: string | null  // 当前选中的节点ID
  selectedLinkId?: string | null  // 当前选中的连接ID
  onNavigateBack?: () => void
  onBreadcrumbClick?: (index: number) => void
  breadcrumbs?: BreadcrumbItem[]
  canGoBack?: boolean
  embedded?: boolean  // 嵌入模式（非弹窗）
  // 编辑连接相关
  connectionMode?: ConnectionMode
  selectedNodes?: Set<string>  // 源节点集合
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>  // 目标节点集合
  onTargetNodesChange?: (nodes: Set<string>) => void
  sourceNode?: string | null  // 保留兼容
  onSourceNodeChange?: (nodeId: string | null) => void
  onManualConnect?: (sourceId: string, targetId: string, level: HierarchyLevel) => void
  manualConnections?: ManualConnection[]
  onDeleteManualConnection?: (connectionId: string) => void
  onDeleteConnection?: (source: string, target: string) => void  // 删除任意连接（包括自动生成的）
  layoutType?: LayoutType  // 布局类型
  onLayoutTypeChange?: (type: LayoutType) => void  // 布局类型变更回调
  // 多层级视图相关
  multiLevelOptions?: MultiLevelViewOptions
  onMultiLevelOptionsChange?: (options: MultiLevelViewOptions) => void
  // 流量热力图
  trafficResult?: TopologyTrafficResult | null
}

export interface Node {
  id: string
  label: string
  type: string
  subType?: string  // Switch的层级，如 "leaf", "spine"
  isSwitch?: boolean
  x: number
  y: number
  color: string
  portInfo?: {
    uplink: number
    downlink: number
    inter: number
  }
  // Torus布局的网格位置
  gridRow?: number
  gridCol?: number
  gridZ?: number  // 3D Torus的Z层
  uHeight?: number  // Board的U高度
  // 多层级视图属性
  parentId?: string              // 父节点ID（下层节点使用）
  hierarchyLevel?: HierarchyLevel // 所属层级
  isContainer?: boolean          // 是否为容器节点
  zLayer?: number                // Z层 (0=下层, 1=上层)
  containerBounds?: {            // 容器边界
    x: number
    y: number
    width: number
    height: number
  }
  // 多层级模式：容器内的单层级完整布局数据（用于展开动画）
  singleLevelData?: {
    nodes: Node[]
    edges: Edge[]
    viewBox: { width: number; height: number }
    scale: number  // 从单层级视图到容器内视图的缩放比例
    directTopology?: string  // 布局类型，用于判断是否需要曲线连接
    switchPanelWidth?: number  // Switch面板宽度
  }
  // Switch面板位置（用于独立的Switch面板区域）
  switchPanelPosition?: {
    x: number
    y: number
    layer: string      // leaf / spine / core
    layerIndex: number // 该层内的索引
  }
  // 标记是否在Switch面板中显示
  inSwitchPanel?: boolean
}

export interface Edge {
  source: string
  target: string
  bandwidth?: number
  latency?: number  // 延迟 (ns)
  isSwitch?: boolean  // 是否为Switch连接
  // 多层级视图属性
  connectionType?: 'intra_upper' | 'intra_lower' | 'inter_level'  // 连接类型
  // 单层级视图：跨层级连接属性
  isExternal?: boolean  // 是否连接到当前层级之外
  externalDirection?: 'upper' | 'lower'  // 外部连接方向（上层/下层）
  externalNodeId?: string  // 外部节点ID
  externalNodeLabel?: string  // 外部节点标签
  // 间接连接属性（通过上层Switch）
  isIndirect?: boolean  // 是否为间接连接
  viaNodeId?: string  // 中转节点ID
  viaNodeLabel?: string  // 中转节点标签
}

// ============================================
// 节点尺寸配置（用于边缘点计算）
// ============================================

// 节点尺寸映射（半宽和半高）
export const NODE_SIZE_MAP: Record<string, { hw: number; hh: number }> = {
  switch: { hw: 30, hh: 12 },      // Switch节点
  pod: { hw: 28, hh: 16 },         // Pod节点
  rack: { hw: 18, hh: 28 },        // Rack节点
  board: { hw: 32, hh: 18 },       // Board节点
  chip: { hw: 20, hh: 20 },        // Chip节点
  default: { hw: 25, hh: 18 },     // 默认尺寸
}

// 多层级视图中的节点尺寸（与MULTI_LEVEL_NODE_SIZE_CONFIG匹配）
export const NODE_SIZE_MAP_MULTI: Record<string, { hw: number; hh: number }> = {
  switch: { hw: 30.5, hh: 12 },  // 61x24 (与单层级相同)
  pod: { hw: 28, hh: 16 },       // 56x32
  rack: { hw: 18, hh: 28 },      // 36x56
  board: { hw: 32, hh: 18 },     // 64x36
  chip: { hw: 20, hh: 20 },      // 40x40
  default: { hw: 25, hh: 18 },   // 50x36
}

/**
 * 计算从节点中心到目标点方向的边缘点
 * @param cx 节点中心X坐标
 * @param cy 节点中心Y坐标
 * @param tx 目标点X坐标
 * @param ty 目标点Y坐标
 * @param nodeType 节点类型（用于确定尺寸）
 * @param isMultiLevel 是否为多层级视图（使用较大尺寸）
 * @param scale 缩放比例（默认1）
 * @returns 边缘点坐标 { x, y }
 */
export function getNodeEdgePoint(
  cx: number,
  cy: number,
  tx: number,
  ty: number,
  nodeType: string,
  isMultiLevel: boolean = false,
  scale: number = 1
): { x: number; y: number } {
  const sizeMap = isMultiLevel ? NODE_SIZE_MAP_MULTI : NODE_SIZE_MAP
  const type = nodeType.toLowerCase()
  const size = sizeMap[type] || sizeMap.default
  const hw = size.hw * scale  // 半宽
  const hh = size.hh * scale  // 半高

  // 计算方向向量
  const dx = tx - cx
  const dy = ty - cy

  // 如果目标点就是中心点，返回中心点
  if (dx === 0 && dy === 0) {
    return { x: cx, y: cy }
  }

  // 计算射线与矩形边界的交点
  // 使用参数化方法：找到射线 (cx + t*dx, cy + t*dy) 与矩形边界的交点
  // 矩形边界：x = cx ± hw, y = cy ± hh

  let t = Infinity

  // 检查与左右边界的交点
  if (dx !== 0) {
    const tRight = hw / Math.abs(dx)
    const tLeft = hw / Math.abs(dx)
    const tX = dx > 0 ? tRight : tLeft
    if (tX < t) {
      const yAtT = cy + tX * dy
      if (Math.abs(yAtT - cy) <= hh) {
        t = tX
      }
    }
  }

  // 检查与上下边界的交点
  if (dy !== 0) {
    const tBottom = hh / Math.abs(dy)
    const tTop = hh / Math.abs(dy)
    const tY = dy > 0 ? tBottom : tTop
    if (tY < t) {
      const xAtT = cx + tY * dx
      if (Math.abs(xAtT - cx) <= hw) {
        t = tY
      }
    }
  }

  // 如果没有找到交点（不应该发生），返回中心点
  if (t === Infinity) {
    return { x: cx, y: cy }
  }

  return {
    x: cx + t * dx,
    y: cy + t * dy,
  }
}
