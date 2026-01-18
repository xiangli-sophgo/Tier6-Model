import { Node, Edge, SWITCH_PANEL_CONFIG, SWITCH_LAYER_ORDER, SwitchPanelLayoutResult } from './shared'
import * as d3Force from 'd3-force'

// ============================================
// 力导向布局类型定义
// ============================================

export interface ForceLayoutOptions {
  // 力的强度参数
  chargeStrength: number      // 节点间斥力强度 (负值表示排斥，默认 -300)
  linkDistance: number        // 连接线的理想距离 (默认 100)
  linkStrength: number        // 连接线的刚度 (0-1，默认 0.5)
  collisionRadius: number     // 碰撞检测半径 (默认 35)
  centerStrength: number      // 向中心的吸引力 (默认 0.05)

  // 模拟参数
  alphaDecay: number          // 衰减率，越小运行越久 (默认 0.02)
  velocityDecay: number       // 速度衰减，越大越快停止 (默认 0.4)
  alphaMin: number            // 最小alpha值，低于此值停止 (默认 0.001)

  // 视图参数
  width: number
  height: number
}

export const DEFAULT_FORCE_OPTIONS: ForceLayoutOptions = {
  chargeStrength: -300,
  linkDistance: 100,
  linkStrength: 0.5,
  collisionRadius: 35,
  centerStrength: 0.05,
  alphaDecay: 0.02,
  velocityDecay: 0.4,
  alphaMin: 0.001,
  width: 800,
  height: 600,
}

// D3 力导向节点类型（扩展 Node）
export interface ForceNode extends Node {
  vx?: number  // x方向速度
  vy?: number  // y方向速度
  fx?: number | null  // 固定x位置（拖拽时使用）
  fy?: number | null  // 固定y位置（拖拽时使用）
  index?: number
}

// D3 力导向连接类型
export interface ForceLink {
  source: string | ForceNode
  target: string | ForceNode
  bandwidth?: number
  latency?: number
}

// 力导向模拟结果
export interface ForceSimulationResult {
  nodes: ForceNode[]
  simulation: d3Force.Simulation<ForceNode, ForceLink>
}

// ============================================
// 力导向布局管理器
// ============================================

export class ForceLayoutManager {
  private simulation: d3Force.Simulation<ForceNode, ForceLink> | null = null
  private nodes: ForceNode[] = []
  private links: ForceLink[] = []
  private options: ForceLayoutOptions
  private onTick: ((nodes: ForceNode[]) => void) | null = null
  private onEnd: (() => void) | null = null

  constructor(options: Partial<ForceLayoutOptions> = {}) {
    this.options = { ...DEFAULT_FORCE_OPTIONS, ...options }
  }

  // 初始化或更新模拟
  initialize(
    nodes: Node[],
    edges: Edge[],
    options?: Partial<ForceLayoutOptions>
  ): ForceNode[] {
    if (options) {
      this.options = { ...this.options, ...options }
    }

    const { width, height, chargeStrength, linkDistance, linkStrength, collisionRadius, centerStrength, alphaDecay, velocityDecay, alphaMin } = this.options

    // 转换节点，保留现有位置或初始化为中心附近随机位置
    this.nodes = nodes.map((node, i) => ({
      ...node,
      x: node.x || width / 2 + (Math.random() - 0.5) * 100,
      y: node.y || height / 2 + (Math.random() - 0.5) * 100,
      vx: 0,
      vy: 0,
      index: i,
    }))

    // 转换边
    this.links = edges.map(edge => ({
      source: edge.source,
      target: edge.target,
      bandwidth: edge.bandwidth,
      latency: edge.latency,
    }))

    // 停止之前的模拟
    if (this.simulation) {
      this.simulation.stop()
    }

    // 创建新的力模拟
    this.simulation = d3Force.forceSimulation<ForceNode, ForceLink>(this.nodes)
      // 节点间斥力（电荷力）
      .force('charge', d3Force.forceManyBody<ForceNode>()
        .strength(chargeStrength)
        .distanceMax(300)  // 限制斥力的最大距离
      )
      // 连接线弹簧力
      .force('link', d3Force.forceLink<ForceNode, ForceLink>(this.links)
        .id(d => d.id)
        .distance(linkDistance)
        .strength(linkStrength)
      )
      // 中心吸引力
      .force('center', d3Force.forceCenter<ForceNode>(width / 2, height / 2)
        .strength(centerStrength)
      )
      // 碰撞检测
      .force('collision', d3Force.forceCollide<ForceNode>()
        .radius(collisionRadius)
        .strength(0.8)
      )
      // 边界约束力（防止节点飞出视图）
      .force('bounds', this.createBoundsForce(width, height, 50))
      // 模拟参数
      .alphaDecay(alphaDecay)
      .velocityDecay(velocityDecay)
      .alphaMin(alphaMin)

    // 绑定事件
    this.simulation.on('tick', () => {
      if (this.onTick) {
        this.onTick(this.nodes)
      }
    })

    this.simulation.on('end', () => {
      if (this.onEnd) {
        this.onEnd()
      }
    })

    return this.nodes
  }

  // 创建边界约束力
  private createBoundsForce(width: number, height: number, padding: number) {
    return () => {
      for (const node of this.nodes) {
        // X边界
        if (node.x! < padding) {
          node.vx! += (padding - node.x!) * 0.1
        } else if (node.x! > width - padding) {
          node.vx! += (width - padding - node.x!) * 0.1
        }
        // Y边界
        if (node.y! < padding) {
          node.vy! += (padding - node.y!) * 0.1
        } else if (node.y! > height - padding) {
          node.vy! += (height - padding - node.y!) * 0.1
        }
      }
    }
  }

  // 设置tick回调
  setOnTick(callback: (nodes: ForceNode[]) => void) {
    this.onTick = callback
  }

  // 设置结束回调
  setOnEnd(callback: () => void) {
    this.onEnd = callback
  }

  // 开始/恢复模拟
  start(alpha: number = 0.3) {
    if (this.simulation) {
      this.simulation.alpha(alpha).restart()
    }
  }

  // 停止模拟
  stop() {
    if (this.simulation) {
      this.simulation.stop()
    }
  }

  // 暂停但不重置
  pause() {
    if (this.simulation) {
      this.simulation.stop()
    }
  }

  // 固定节点位置（拖拽开始）
  fixNode(nodeId: string, x: number, y: number) {
    const node = this.nodes.find(n => n.id === nodeId)
    if (node) {
      node.fx = x
      node.fy = y
    }
  }

  // 更新固定位置（拖拽中）
  dragNode(nodeId: string, x: number, y: number) {
    const node = this.nodes.find(n => n.id === nodeId)
    if (node) {
      node.fx = x
      node.fy = y
    }
    // 稍微加热模拟以响应拖拽
    if (this.simulation) {
      this.simulation.alpha(0.1).restart()
    }
  }

  // 释放节点（拖拽结束）
  releaseNode(nodeId: string) {
    const node = this.nodes.find(n => n.id === nodeId)
    if (node) {
      node.fx = null
      node.fy = null
    }
    // 释放后加热让节点自然运动
    if (this.simulation) {
      this.simulation.alpha(0.2).restart()
    }
  }

  // 加热模拟（重新激活）
  reheat(alpha: number = 0.3) {
    if (this.simulation) {
      this.simulation.alpha(alpha).restart()
    }
  }

  // 获取当前节点位置
  getNodes(): ForceNode[] {
    return this.nodes
  }

  // 获取模拟实例
  getSimulation(): d3Force.Simulation<ForceNode, ForceLink> | null {
    return this.simulation
  }

  // 更新选项
  updateOptions(options: Partial<ForceLayoutOptions>) {
    this.options = { ...this.options, ...options }

    if (!this.simulation) return

    const { chargeStrength, linkDistance, linkStrength, collisionRadius, centerStrength, width, height } = this.options

    // 更新各个力
    const chargeForce = this.simulation.force('charge') as d3Force.ForceManyBody<ForceNode>
    if (chargeForce) {
      chargeForce.strength(chargeStrength)
    }

    const linkForce = this.simulation.force('link') as d3Force.ForceLink<ForceNode, ForceLink>
    if (linkForce) {
      linkForce.distance(linkDistance).strength(linkStrength)
    }

    const centerForce = this.simulation.force('center') as d3Force.ForceCenter<ForceNode>
    if (centerForce) {
      centerForce.x(width / 2).y(height / 2).strength(centerStrength)
    }

    const collisionForce = this.simulation.force('collision') as d3Force.ForceCollide<ForceNode>
    if (collisionForce) {
      collisionForce.radius(collisionRadius)
    }

    // 更新边界力
    this.simulation.force('bounds', this.createBoundsForce(width, height, 50))

    // 重新加热
    this.reheat(0.3)
  }

  // 销毁模拟
  destroy() {
    if (this.simulation) {
      this.simulation.stop()
      this.simulation = null
    }
    this.nodes = []
    this.links = []
    this.onTick = null
    this.onEnd = null
  }
}

// ============================================
// 简单的力导向布局函数（一次性计算）
// ============================================

export function forceDirectedLayout(
  nodes: Node[],
  edges: Edge[],
  width: number,
  height: number,
  options: Partial<ForceLayoutOptions> = {}
): Node[] {
  const opts: ForceLayoutOptions = { ...DEFAULT_FORCE_OPTIONS, ...options, width, height }

  // 转换节点
  const forceNodes: ForceNode[] = nodes.map((node, i) => ({
    ...node,
    x: node.x || width / 2 + (Math.random() - 0.5) * 200,
    y: node.y || height / 2 + (Math.random() - 0.5) * 200,
    vx: 0,
    vy: 0,
    index: i,
  }))

  // 转换边
  const forceLinks: ForceLink[] = edges.map(edge => ({
    source: edge.source,
    target: edge.target,
  }))

  // 创建模拟
  const simulation = d3Force.forceSimulation<ForceNode, ForceLink>(forceNodes)
    .force('charge', d3Force.forceManyBody<ForceNode>().strength(opts.chargeStrength))
    .force('link', d3Force.forceLink<ForceNode, ForceLink>(forceLinks)
      .id(d => d.id)
      .distance(opts.linkDistance)
      .strength(opts.linkStrength)
    )
    .force('center', d3Force.forceCenter<ForceNode>(width / 2, height / 2))
    .force('collision', d3Force.forceCollide<ForceNode>().radius(opts.collisionRadius))
    .stop()

  // 运行模拟直到稳定（同步计算）
  const iterations = 300
  for (let i = 0; i < iterations; i++) {
    simulation.tick()
  }

  // 返回计算后的节点位置
  return forceNodes.map(n => ({
    ...n,
    x: Math.max(50, Math.min(width - 50, n.x!)),
    y: Math.max(50, Math.min(height - 50, n.y!)),
  }))
}

// ============================================
// 等轴测投影工具函数
// ============================================

// 等轴测投影参数
const ISO_ANGLE = Math.PI / 6  // 30度
const ISO_SCALE_X = Math.cos(ISO_ANGLE)  // X轴缩放 ≈ 0.866
const ISO_SCALE_Y = Math.sin(ISO_ANGLE)  // Y轴偏移 ≈ 0.5

// 等轴测坐标转换：将3D坐标(x, y, z)投影到2D
export function isoProject(x: number, y: number, z: number): { px: number; py: number } {
  const px = (x - z) * ISO_SCALE_X
  const py = (x + z) * ISO_SCALE_Y - y
  return { px, py }
}

// 容器边界
export interface ContainerBounds {
  x: number
  y: number
  width: number
  height: number
}

// 堆叠布局选项
export interface StackedLayoutOptions {
  layerGap: number           // 层间垂直距离
  containerPadding: number   // 容器内边距
  innerNodeScale: number     // 内部节点缩放比例
  upperNodeSize: number      // 上层节点大小
  lowerNodeSize: number      // 下层节点大小
}

// 堆叠布局结果
export interface StackedLayoutResult {
  upperNodes: Node[]
  lowerNodes: Node[]
  containerBounds: Map<string, ContainerBounds>
}

// 默认堆叠布局选项
const DEFAULT_STACKED_OPTIONS: StackedLayoutOptions = {
  layerGap: 120,
  containerPadding: 30,
  innerNodeScale: 0.6,
  upperNodeSize: 60,
  lowerNodeSize: 25,
}

// 多层堆叠布局 - Z轴垂直堆叠效果
// 每个上层节点展开为一个容器，内部显示下层节点的拓扑
// 多个容器在Z轴方向垂直堆叠，形成卡片堆叠效果，支持悬停抬起交互
export function isometricStackedLayout(
  upperNodes: Node[],
  lowerNodesMap: Map<string, Node[]>,
  width: number,
  height: number,
  options: Partial<StackedLayoutOptions> = {}
): StackedLayoutResult {
  const opts = { ...DEFAULT_STACKED_OPTIONS, ...options }
  const { containerPadding, innerNodeScale } = opts

  // 如果没有上层节点，返回空结果
  if (upperNodes.length === 0) {
    return { upperNodes: [], lowerNodes: [], containerBounds: new Map() }
  }

  // 分离Switch节点和容器节点
  const switchNodes = upperNodes.filter(n => n.isSwitch)
  const containerNodes = upperNodes.filter(n => !n.isSwitch)

  const containerBounds = new Map<string, ContainerBounds>()
  const layoutedLower: Node[] = []
  const layoutedUpper: Node[] = []

  // 计算Switch面板布局（如果有Switch）
  let switchPanelWidth = 0
  if (switchNodes.length > 0) {
    // 获取所有边（这里暂时传空数组，Switch内部连线在渲染时处理）
    const switchPanelResult = computeSwitchPanelLayout(switchNodes, [], height)
    switchPanelWidth = switchPanelResult.panelWidth
    // 添加布局后的Switch节点
    layoutedUpper.push(...switchPanelResult.switchNodes)
  }

  // 计算容器节点数量来确定布局（在Z方向堆叠）
  const containerCount = containerNodes.length
  if (containerCount === 0) {
    return { upperNodes: layoutedUpper, lowerNodes: layoutedLower, containerBounds }
  }

  // 每个容器的基础大小（考虑Switch面板宽度）
  const availableWidth = width - switchPanelWidth
  const containerWidth = Math.min(availableWidth * 0.85, 350)
  const containerHeight = 220  // 增大高度以容纳更多内容

  // 书本堆叠参数
  const bookThickness = 40   // 每本书的"厚度"（层间露出的高度）
  const depth3D = 80         // 3D深度（顶面和侧面的高度）

  // 计算整体堆叠的高度
  const totalStackHeight = containerHeight + (containerCount - 1) * bookThickness + depth3D

  // 计算起始位置，使整体居中（考虑Switch面板偏移）
  // zLayer=0 在最上面（Y最小），zLayer越大越在下面（Y越大）
  const baseX = switchPanelWidth + availableWidth / 2
  const baseY = (height - totalStackHeight) / 2 + depth3D + containerHeight / 2

  // 计算每个容器（展开的上层节点）及其内部的下层节点
  containerNodes.forEach((upperNode, idx) => {
    // 书本堆叠：idx=0 的 zLayer=0 在最上面
    const zIndex = idx

    // 容器中心位置 - zLayer=0在最上面（Y最小），zLayer越大越在下面（Y越大）
    const containerCenterX = baseX
    const containerCenterY = baseY + zIndex * bookThickness

    // 容器边界（作为展开的上层节点）
    const bounds: ContainerBounds = {
      x: containerCenterX - containerWidth / 2,
      y: containerCenterY - containerHeight / 2,
      width: containerWidth,
      height: containerHeight,
    }
    containerBounds.set(upperNode.id, bounds)

    // 上层节点作为容器
    layoutedUpper.push({
      ...upperNode,
      x: containerCenterX,
      y: containerCenterY,
      isContainer: true,
      zLayer: zIndex,
      containerBounds: bounds,
    })

    // 获取下层子节点
    const children = lowerNodesMap.get(upperNode.id) || []
    const childCount = children.length

    if (childCount === 0) return

    // 布局下层子节点（在容器内使用网格布局）
    const childCols = Math.ceil(Math.sqrt(childCount))
    const childRows = Math.ceil(childCount / childCols)
    const innerWidth = (containerWidth - containerPadding * 2) * innerNodeScale
    const innerHeight = (containerHeight - containerPadding * 2) * innerNodeScale
    const childSpacingX = childCols > 1 ? innerWidth / (childCols - 1) : 0
    const childSpacingY = childRows > 1 ? innerHeight / (childRows - 1) : 0

    children.forEach((child, i) => {
      const childCol = i % childCols
      const childRow = Math.floor(i / childCols)

      // 子节点在容器内的相对位置
      const relX = childCols === 1 ? 0 : (childCol - (childCols - 1) / 2) * childSpacingX
      const relY = childRows === 1 ? 0 : (childRow - (childRows - 1) / 2) * childSpacingY

      layoutedLower.push({
        ...child,
        x: containerCenterX + relX,
        y: containerCenterY + relY,
        parentId: upperNode.id,
        zLayer: zIndex,  // 与父容器相同的zLayer
      })
    })
  })

  return { upperNodes: layoutedUpper, lowerNodes: layoutedLower, containerBounds }
}

// 布局算法：圆形布局
export function circleLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  // 只有一个节点时，放在中心
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：环形拓扑布局（用于ring连接）
export function ringLayout(nodes: Node[], centerX: number, centerY: number, radius: number): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: centerX, y: centerY }]
  }
  return nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos((2 * Math.PI * i) / count - Math.PI / 2),
    y: centerY + radius * Math.sin((2 * Math.PI * i) / count - Math.PI / 2),
  }))
}

// 布局算法：2D Torus/网格布局（用于torus_2d和grid连接）
// 标准Torus可视化：节点排成规则网格，环绕边画在外围
export function torusLayout(nodes: Node[], width: number, height: number, padding: number = 120): Node[] {
  const count = nodes.length
  if (count === 1) {
    return [{ ...nodes[0], x: width / 2, y: height / 2 }]
  }
  // 计算最佳的行列数，尽量接近正方形
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)

  // 留出较大边距给环绕连接线
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const spacingX = cols > 1 ? innerWidth / (cols - 1) : 0
  const spacingY = rows > 1 ? innerHeight / (rows - 1) : 0

  // 居中偏移
  const offsetX = cols === 1 ? width / 2 : padding
  const offsetY = rows === 1 ? height / 2 : padding

  return nodes.map((node, i) => ({
    ...node,
    x: offsetX + (i % cols) * spacingX,
    y: offsetY + Math.floor(i / cols) * spacingY,
    // 存储网格位置信息用于连接线计算
    gridRow: Math.floor(i / cols),
    gridCol: i % cols,
  }))
}

// 计算Torus网格的行列数
export function getTorusGridSize(count: number): { cols: number; rows: number } {
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  return { cols, rows }
}

// 3D Torus专用布局：等轴测投影，呈现3D立方体效果
export function torus3DLayout(nodes: Node[], width: number, height: number, _padding: number = 100): Node[] {
  const count = nodes.length
  if (count <= 1) {
    return nodes.map(n => ({ ...n, x: width / 2, y: height / 2, gridRow: 0, gridCol: 0, gridZ: 0 }))
  }

  // 计算3D维度（尽量接近立方体）
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const nodesPerLayer = dim * dim

  // 等轴测投影参数
  const centerX = width / 2
  const centerY = height / 2
  const spacingX = 140  // X方向间距
  const spacingY = 120  // Y方向间距（垂直）
  const spacingZ = 90   // Z方向间距（深度，斜向）

  return nodes.map((node, i) => {
    const z = Math.floor(i / nodesPerLayer)
    const inLayerIndex = i % nodesPerLayer
    const row = Math.floor(inLayerIndex / dim)  // Y轴（上下）
    const col = inLayerIndex % dim              // X轴（左右）

    // 等轴测投影：
    // X轴向右，Y轴向下，Z轴向右上方（模拟深度）
    const x = centerX + (col - (dim - 1) / 2) * spacingX + (z - (dim - 1) / 2) * spacingZ * 0.6
    const y = centerY + (row - (dim - 1) / 2) * spacingY - (z - (dim - 1) / 2) * spacingZ * 0.5

    return {
      ...node,
      x,
      y,
      gridRow: row,
      gridCol: col,
      gridZ: z,
    }
  })
}

// 计算3D Torus的维度
export function getTorus3DSize(count: number): { dim: number; layers: number } {
  const dim = Math.max(2, Math.ceil(Math.pow(count, 1 / 3)))
  const layers = Math.ceil(count / (dim * dim))
  return { dim, layers }
}

// 根据直连拓扑类型选择最佳布局
export function getLayoutForTopology(
  topologyType: string,
  nodes: Node[],
  width: number,
  height: number
): Node[] {
  const centerX = width / 2
  const centerY = height / 2
  const radius = Math.min(width, height) * 0.35

  switch (topologyType) {
    case 'ring':
      return ringLayout(nodes, centerX, centerY, radius)
    case 'torus_2d':
      return torusLayout(nodes, width, height)
    case 'torus_3d':
      return torus3DLayout(nodes, width, height)
    case 'full_mesh_2d':
      // 2D FullMesh使用网格布局（行列全连接）
      return torusLayout(nodes, width, height)
    case 'full_mesh':
      // 全连接用圆形布局最清晰
      return circleLayout(nodes, centerX, centerY, radius)
    case 'none':
    default:
      // 无连接或默认用圆形
      return circleLayout(nodes, centerX, centerY, radius)
  }
}

// 布局算法：分层布局（用于显示Switch层级，设备节点排成一排）
export function hierarchicalLayout(nodes: Node[], width: number, height: number): Node[] {
  // 按类型分组
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，设备节点居中显示
  if (switchNodes.length === 0) {
    const centerY = height / 2
    if (deviceNodes.length === 1) {
      return [{ ...deviceNodes[0], x: width / 2, y: centerY }]
    }
    const spacing = width / (deviceNodes.length + 1)
    return deviceNodes.map((node, i) => ({
      ...node,
      x: spacing * (i + 1),
      y: centerY,
    }))
  }

  // Switch按subType分层
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })

  // 层级顺序：device在最下面，然后是leaf, spine, core
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  const totalLayers = sortedLayers.length + (deviceNodes.length > 0 ? 1 : 0)
  const layerSpacing = 100 // 每层之间的间距
  const totalHeight = (totalLayers - 1) * layerSpacing
  const startY = (height + totalHeight) / 2 // 垂直居中的起始Y（最底层）

  const result: Node[] = []

  // 设备节点在最底层
  if (deviceNodes.length > 0) {
    const y = startY
    const spacing = width / (deviceNodes.length + 1)
    deviceNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  }

  // Switch节点按层级向上排列（在设备上方）
  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = startY - layerSpacing * (layerIdx + (deviceNodes.length > 0 ? 1 : 0))
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
}

// 布局算法：混合布局（设备节点按拓扑排列，Switch节点在上方中央）
// 用于同时有Switch和节点直连的场景
export function hybridLayout(
  nodes: Node[],
  width: number,
  height: number,
  directTopology: string
): Node[] {
  const switchNodes = nodes.filter(n => n.isSwitch)
  const deviceNodes = nodes.filter(n => !n.isSwitch)

  // 如果没有Switch，使用普通拓扑布局
  if (switchNodes.length === 0) {
    return getLayoutForTopology(directTopology, deviceNodes, width, height)
  }

  // Switch层数决定Switch区域高度
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(n => {
    const layer = n.subType || 'default'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(n)
  })
  const switchLayerCount = Object.keys(switchLayers).length

  // 动态计算区域划分：Switch区域更紧凑
  const switchLayerHeight = 50  // 每层Switch的高度
  const switchAreaHeight = switchLayerCount * switchLayerHeight
  const switchAreaTop = 60  // Switch起始位置（留出顶部空间）
  const gapBetween = 40  // Switch和设备之间的间隙

  // 设备节点区域
  const deviceAreaTop = switchAreaTop + switchAreaHeight + gapBetween
  const deviceAreaHeight = height - deviceAreaTop - 30  // 底部留30px

  const result: Node[] = []

  // 1. 设备节点按拓扑类型布局（在下方区域）
  const centerX = width / 2
  const centerY = deviceAreaTop + deviceAreaHeight / 2
  const radius = Math.min(width * 0.4, deviceAreaHeight * 0.45)

  let layoutedDevices: Node[]
  switch (directTopology) {
    case 'ring':
      layoutedDevices = ringLayout(deviceNodes, centerX, centerY, radius)
      break
    case 'torus_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'torus_3d':
      layoutedDevices = torus3DLayout(deviceNodes, width, deviceAreaHeight, 60)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop - 30 }))
      break
    case 'full_mesh_2d':
      layoutedDevices = torusLayout(deviceNodes, width, deviceAreaHeight, 80)
      layoutedDevices = layoutedDevices.map(n => ({ ...n, y: n.y + deviceAreaTop }))
      break
    case 'full_mesh':
    default:
      layoutedDevices = circleLayout(deviceNodes, centerX, centerY, radius)
      break
  }
  result.push(...layoutedDevices)

  // 2. Switch节点按层级排列（在上方区域）
  const layerOrder = ['leaf', 'spine', 'core']
  const sortedLayers = Object.keys(switchLayers).sort((a, b) => {
    const aIdx = layerOrder.indexOf(a)
    const bIdx = layerOrder.indexOf(b)
    return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
  })

  sortedLayers.forEach((layer, layerIdx) => {
    const layerNodes = switchLayers[layer]
    const y = switchAreaTop + layerIdx * switchLayerHeight
    const spacing = width / (layerNodes.length + 1)
    layerNodes.forEach((node, i) => {
      result.push({ ...node, x: spacing * (i + 1), y })
    })
  })

  return result
}

// ==========================================
// useForceLayout Hook
// ==========================================

import { useRef, useCallback, useEffect, useState } from 'react'

export interface UseForceLayoutOptions extends Partial<ForceLayoutOptions> {
  enabled: boolean
  onNodePositionsChange?: (nodes: ForceNode[]) => void
}

export interface UseForceLayoutResult {
  isSimulating: boolean
  nodes: ForceNode[]
  initialize: (nodes: Node[], edges: Edge[]) => void
  start: () => void
  stop: () => void
  reheat: () => void
  onDragStart: (nodeId: string, x: number, y: number) => void
  onDrag: (nodeId: string, x: number, y: number) => void
  onDragEnd: (nodeId: string) => void
  updateOptions: (options: Partial<ForceLayoutOptions>) => void
}

export function useForceLayout(options: UseForceLayoutOptions): UseForceLayoutResult {
  const { enabled, onNodePositionsChange, ...forceOptions } = options

  const managerRef = useRef<ForceLayoutManager | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [nodes, setNodes] = useState<ForceNode[]>([])

  const getManager = useCallback(() => {
    if (!managerRef.current) {
      managerRef.current = new ForceLayoutManager({
        ...DEFAULT_FORCE_OPTIONS,
        ...forceOptions,
      })
    }
    return managerRef.current
  }, [])

  const initialize = useCallback((inputNodes: Node[], edges: Edge[]) => {
    if (!enabled) return

    const manager = getManager()
    const opts: Partial<ForceLayoutOptions> = { ...forceOptions }
    const initialNodes = manager.initialize(inputNodes, edges, opts)

    manager.setOnTick((updatedNodes) => {
      setNodes([...updatedNodes])
      if (onNodePositionsChange) {
        onNodePositionsChange(updatedNodes)
      }
    })

    manager.setOnEnd(() => {
      setIsSimulating(false)
    })

    setNodes(initialNodes)
    setIsSimulating(true)
    manager.start(1.0)
  }, [enabled, getManager, onNodePositionsChange, forceOptions])

  const start = useCallback(() => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      setIsSimulating(true)
      manager.start()
    }
  }, [enabled])

  const stop = useCallback(() => {
    const manager = managerRef.current
    if (manager) {
      manager.stop()
      setIsSimulating(false)
    }
  }, [])

  const reheat = useCallback(() => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      setIsSimulating(true)
      manager.reheat()
    }
  }, [enabled])

  const onDragStart = useCallback((nodeId: string, x: number, y: number) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.fixNode(nodeId, x, y)
      setIsSimulating(true)
    }
  }, [enabled])

  const onDrag = useCallback((nodeId: string, x: number, y: number) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.dragNode(nodeId, x, y)
    }
  }, [enabled])

  const onDragEnd = useCallback((nodeId: string) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.releaseNode(nodeId)
    }
  }, [enabled])

  const updateOptions = useCallback((newOptions: Partial<ForceLayoutOptions>) => {
    const manager = managerRef.current
    if (manager) {
      manager.updateOptions(newOptions)
    }
  }, [])

  useEffect(() => {
    if (!enabled && managerRef.current) {
      managerRef.current.stop()
      setIsSimulating(false)
    }
  }, [enabled])

  useEffect(() => {
    return () => {
      if (managerRef.current) {
        managerRef.current.destroy()
        managerRef.current = null
      }
    }
  }, [])

  return {
    isSimulating,
    nodes,
    initialize,
    start,
    stop,
    reheat,
    onDragStart,
    onDrag,
    onDragEnd,
    updateOptions,
  }
}

// ============================================
// Switch面板布局函数
// ============================================

/**
 * 计算Switch面板布局
 * Switch按层级（leaf/spine/core）分组，呈树形结构排列
 * @param switchNodes Switch节点列表
 * @param allEdges 所有边（用于提取Switch之间的连接）
 * @param totalHeight 总高度（用于垂直居中）
 * @returns Switch面板布局结果
 */
export function computeSwitchPanelLayout(
  switchNodes: Node[],
  allEdges: Edge[],
  totalHeight: number
): SwitchPanelLayoutResult {
  if (switchNodes.length === 0) {
    return {
      panelWidth: 0,
      switchNodes: [],
      switchEdges: [],
      deviceAreaOffset: 0,
    }
  }

  const { minWidth, maxWidth, nodeWidth, nodeHeight, layerGap, nodeGap, padding, panelGap } = SWITCH_PANEL_CONFIG

  // 按层分组
  const switchLayers: Record<string, Node[]> = {}
  switchNodes.forEach(node => {
    const layer = node.subType || 'leaf'
    if (!switchLayers[layer]) switchLayers[layer] = []
    switchLayers[layer].push(node)
  })

  // 按层级顺序排序（leaf在下，spine在上，core在最上）
  const sortedLayers = SWITCH_LAYER_ORDER.filter(l => switchLayers[l])
  const layerCount = sortedLayers.length

  if (layerCount === 0) {
    return {
      panelWidth: 0,
      switchNodes: [],
      switchEdges: [],
      deviceAreaOffset: 0,
    }
  }

  // 计算面板宽度（基于最大层的节点数）
  const maxNodesInLayer = Math.max(...Object.values(switchLayers).map(arr => arr.length))
  const panelWidth = Math.min(
    maxWidth,
    Math.max(
      minWidth,
      maxNodesInLayer * (nodeWidth + nodeGap) + padding * 2
    )
  )

  // 计算总层高度
  const totalLayerHeight = (layerCount - 1) * layerGap + nodeHeight
  const startY = (totalHeight - totalLayerHeight) / 2

  // 布局每层（leaf在下面，spine/core在上面 - 树形结构）
  const layoutedNodes: Node[] = []
  sortedLayers.forEach((layer, layerIdx) => {
    const nodesInLayer = switchLayers[layer]
    // layerIdx=0是leaf（最底层），layerIdx越大越靠上
    const layerY = startY + (layerCount - 1 - layerIdx) * layerGap + nodeHeight / 2
    const totalNodesWidth = nodesInLayer.length * nodeWidth + (nodesInLayer.length - 1) * nodeGap
    const startX = (panelWidth - totalNodesWidth) / 2 + nodeWidth / 2

    nodesInLayer.forEach((node, nodeIdx) => {
      layoutedNodes.push({
        ...node,
        x: startX + nodeIdx * (nodeWidth + nodeGap),
        y: layerY,
        inSwitchPanel: true,
        switchPanelPosition: {
          x: startX + nodeIdx * (nodeWidth + nodeGap),
          y: layerY,
          layer,
          layerIndex: nodeIdx,
        },
      })
    })
  })

  // 提取Switch之间的边
  const switchIds = new Set(switchNodes.map(n => n.id))
  const switchEdges = allEdges.filter(e =>
    switchIds.has(e.source) && switchIds.has(e.target)
  )

  return {
    panelWidth,
    switchNodes: layoutedNodes,
    switchEdges,
    deviceAreaOffset: panelWidth + panelGap,
  }
}

/**
 * 分离Switch节点和设备节点
 */
export function separateSwitchAndDeviceNodes(allNodes: Node[]): {
  switchNodes: Node[]
  deviceNodes: Node[]
} {
  return {
    switchNodes: allNodes.filter(n => n.isSwitch),
    deviceNodes: allNodes.filter(n => !n.isSwitch),
  }
}
