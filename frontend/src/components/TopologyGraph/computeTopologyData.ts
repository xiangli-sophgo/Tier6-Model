/**
 * 拓扑数据计算模块
 * 将节点和边的计算逻辑从主组件中分离，便于测试和维护
 */
import {
  CHIP_TYPE_COLORS,
  HierarchyLevel,
  LayoutType,
  MultiLevelViewOptions,
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ManualConnection,
  LEVEL_PAIR_NAMES,
} from '../../types'
import {
  BOARD_U_COLORS,
  Node,
  Edge,
} from './shared'
import { convertSwitchesToNodes, mergeManualConnections, buildSwitchLabelsMap, getSwitchIds } from './utils'
import {
  circleLayout,
  torusLayout,
  getLayoutForTopology,
  isometricStackedLayout,
  forceDirectedLayout,
  computeSwitchPanelLayout,
  separateSwitchAndDeviceNodes,
} from './layouts'

// ==========================================
// 类型定义
// ==========================================
export interface ComputeTopologyParams {
  topology: HierarchicalTopology | null
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  layoutType: LayoutType
  multiLevelOptions?: MultiLevelViewOptions
  manualConnections: ManualConnection[]
}

export interface TopologyDataResult {
  nodes: Node[]
  edges: Edge[]
  title: string
  directTopology: string
  switchPanelWidth: number  // Switch面板宽度（0表示无Switch面板）
}

// ==========================================
// 主计算函数
// ==========================================
export function computeTopologyData(params: ComputeTopologyParams): TopologyDataResult {
  const {
    topology,
    currentLevel,
    currentPod,
    currentRack,
    currentBoard,
    layoutType,
    multiLevelOptions,
    manualConnections,
  } = params

  if (!topology) return { nodes: [], edges: [], title: '', directTopology: 'full_mesh', switchPanelWidth: 0 }

  const width = 800
  const height = 600

  // 多层级模式
  if (multiLevelOptions?.enabled && multiLevelOptions.levelPair) {
    return computeMultiLevelData({
      topology,
      currentPod,
      currentRack,
      currentBoard,
      multiLevelOptions,
      layoutType,
      width,
      height,
    })
  }

  // 单层级模式
  return computeSingleLevelData({
    topology,
    currentLevel,
    currentPod,
    currentRack,
    currentBoard,
    layoutType,
    manualConnections,
    width,
    height,
  })
}

// ==========================================
// 多层级数据计算
// ==========================================
interface MultiLevelParams {
  topology: HierarchicalTopology
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  multiLevelOptions: MultiLevelViewOptions
  layoutType: LayoutType
  width: number
  height: number
}

function computeMultiLevelData(params: MultiLevelParams): TopologyDataResult {
  const { topology, currentPod, currentRack, currentBoard, multiLevelOptions, layoutType, width, height } = params
  const levelPair = multiLevelOptions.levelPair!

  let upperNodes: Node[] = []
  let lowerNodesMap = new Map<string, Node[]>()
  let allEdges: Edge[] = []
  let graphTitle = LEVEL_PAIR_NAMES[levelPair] + ' 拓扑'

  // 根据层级组合提取节点
  if (levelPair === 'datacenter_pod') {
    // 上层：Pod节点 + Datacenter级别的Switch（inter_pod）
    upperNodes = topology.pods.map(pod => ({
      id: pod.id,
      label: pod.label,
      type: 'pod',
      x: 0, y: 0,
      color: '#1890ff',
      hierarchyLevel: 'datacenter' as HierarchyLevel,
    }))
    // 添加Datacenter级别的Switch到上层
    upperNodes.push(...convertSwitchesToNodes(topology.switches, 'inter_pod'))

    // 下层：每个Pod容器内的Rack + Pod级别的Switch（inter_rack）
    topology.pods.forEach(pod => {
      const racks: Node[] = pod.racks.map(rack => ({
        id: rack.id,
        label: rack.label,
        type: 'rack',
        x: 0, y: 0,
        color: '#52c41a',
        hierarchyLevel: 'pod' as HierarchyLevel,
      }))
      // 添加Pod级别的Switch到容器内
      const podSwitches = convertSwitchesToNodes(topology.switches, 'inter_rack', pod.id)
      lowerNodesMap.set(pod.id, [...racks, ...podSwitches])
    })

    // 收集所有节点ID（包括Switch）
    const dcSwitchIds = getSwitchIds(topology.switches, 'inter_pod')
    const podSwitchIds = new Set(
      topology.pods.flatMap(p =>
        Array.from(getSwitchIds(topology.switches, 'inter_rack', p.id))
      )
    )
    const allNodeIds = new Set([
      ...topology.pods.map(p => p.id),
      ...topology.pods.flatMap(p => p.racks.map(r => r.id)),
      ...dcSwitchIds,
      ...podSwitchIds,
    ])
    allEdges = topology.connections
      .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
      .map(c => {
        const sourceIsPod = topology.pods.some(p => p.id === c.source)
        const targetIsPod = topology.pods.some(p => p.id === c.target)
        const sourceIsDcSwitch = dcSwitchIds.has(c.source)
        const targetIsDcSwitch = dcSwitchIds.has(c.target)
        let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
        // 上层连接：Pod之间 或 Pod与DC级Switch之间 或 DC级Switch之间
        if ((sourceIsPod || sourceIsDcSwitch) && (targetIsPod || targetIsDcSwitch)) {
          connectionType = 'intra_upper'
        } else if (!sourceIsPod && !targetIsPod && !sourceIsDcSwitch && !targetIsDcSwitch) {
          // 下层连接：同一Pod内的Rack/Switch之间
          const sourcePod = topology.pods.find(p =>
            p.racks.some(r => r.id === c.source) ||
            getSwitchIds(topology.switches, 'inter_rack', p.id).has(c.source)
          )
          const targetPod = topology.pods.find(p =>
            p.racks.some(r => r.id === c.target) ||
            getSwitchIds(topology.switches, 'inter_rack', p.id).has(c.target)
          )
          if (sourcePod && targetPod && sourcePod.id === targetPod.id) {
            connectionType = 'intra_lower'
          }
        }
        return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
      })
  } else if (levelPair === 'pod_rack' && currentPod) {
    // 上层：Rack节点 + Pod级别的Switch（inter_rack）
    upperNodes = currentPod.racks.map(rack => ({
      id: rack.id,
      label: rack.label,
      type: 'rack',
      x: 0, y: 0,
      color: '#52c41a',
      hierarchyLevel: 'pod' as HierarchyLevel,
    }))
    // 添加Pod级别的Switch到上层
    upperNodes.push(...convertSwitchesToNodes(topology.switches, 'inter_rack', currentPod.id))

    // 下层：每个Rack容器内的Board + Rack级别的Switch（inter_board）
    currentPod.racks.forEach(rack => {
      const boards: Node[] = rack.boards.map(board => ({
        id: board.id,
        label: board.label,
        type: 'board',
        x: 0, y: 0,
        color: BOARD_U_COLORS[board.u_height] || '#666',
        uHeight: board.u_height,
        hierarchyLevel: 'rack' as HierarchyLevel,
      }))
      // 添加Rack级别的Switch到容器内
      const rackSwitches = convertSwitchesToNodes(topology.switches, 'inter_board', rack.id)
      lowerNodesMap.set(rack.id, [...boards, ...rackSwitches])
    })
    graphTitle = `${currentPod.label} - ${LEVEL_PAIR_NAMES[levelPair]}`

    // 收集所有节点ID（包括Switch）
    const podSwitchIds = getSwitchIds(topology.switches, 'inter_rack', currentPod.id)
    const rackSwitchIds = new Set(
      currentPod.racks.flatMap(r =>
        Array.from(getSwitchIds(topology.switches, 'inter_board', r.id))
      )
    )
    const allNodeIds = new Set([
      ...currentPod.racks.map(r => r.id),
      ...currentPod.racks.flatMap(r => r.boards.map(b => b.id)),
      ...podSwitchIds,
      ...rackSwitchIds,
    ])
    allEdges = topology.connections
      .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
      .map(c => {
        const sourceIsRack = currentPod.racks.some(r => r.id === c.source)
        const targetIsRack = currentPod.racks.some(r => r.id === c.target)
        const sourceIsPodSwitch = podSwitchIds.has(c.source)
        const targetIsPodSwitch = podSwitchIds.has(c.target)
        let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
        // 上层连接：Rack之间 或 Rack与Pod级Switch之间 或 Pod级Switch之间
        if ((sourceIsRack || sourceIsPodSwitch) && (targetIsRack || targetIsPodSwitch)) {
          connectionType = 'intra_upper'
        } else if (!sourceIsRack && !targetIsRack && !sourceIsPodSwitch && !targetIsPodSwitch) {
          // 下层连接：同一Rack内的Board/Switch之间
          const sourceRack = currentPod.racks.find(r =>
            r.boards.some(b => b.id === c.source) ||
            getSwitchIds(topology.switches, 'inter_board', r.id).has(c.source)
          )
          const targetRack = currentPod.racks.find(r =>
            r.boards.some(b => b.id === c.target) ||
            getSwitchIds(topology.switches, 'inter_board', r.id).has(c.target)
          )
          if (sourceRack && targetRack && sourceRack.id === targetRack.id) {
            connectionType = 'intra_lower'
          }
        }
        return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
      })
  } else if (levelPair === 'rack_board' && currentRack) {
    // 上层：Board节点 + Rack级别的Switch（inter_board）
    upperNodes = currentRack.boards.map(board => ({
      id: board.id,
      label: board.label,
      type: 'board',
      x: 0, y: 0,
      color: BOARD_U_COLORS[board.u_height] || '#666',
      uHeight: board.u_height,
      hierarchyLevel: 'rack' as HierarchyLevel,
    }))
    // 添加Rack级别的Switch到上层
    upperNodes.push(...convertSwitchesToNodes(topology.switches, 'inter_board', currentRack.id))

    // 下层：每个Board容器内的Chip + Board级别的Switch（inter_chip）
    currentRack.boards.forEach(board => {
      const chips: Node[] = board.chips.map(chip => ({
        id: chip.id,
        label: chip.label || chip.id,
        type: 'chip',
        x: 0, y: 0,
        color: CHIP_TYPE_COLORS[chip.type] || '#666',
        hierarchyLevel: 'board' as HierarchyLevel,
      }))
      // 添加Board级别的Switch到容器内
      const boardSwitches = convertSwitchesToNodes(topology.switches, 'inter_chip', board.id)
      lowerNodesMap.set(board.id, [...chips, ...boardSwitches])
    })
    graphTitle = `${currentRack.label} - ${LEVEL_PAIR_NAMES[levelPair]}`

    // 收集所有节点ID（包括Switch）
    const rackSwitchIds = getSwitchIds(topology.switches, 'inter_board', currentRack.id)
    const boardSwitchIds = new Set(
      currentRack.boards.flatMap(b =>
        Array.from(getSwitchIds(topology.switches, 'inter_chip', b.id))
      )
    )
    const allNodeIds = new Set([
      ...currentRack.boards.map(b => b.id),
      ...currentRack.boards.flatMap(b => b.chips.map(c => c.id)),
      ...rackSwitchIds,
      ...boardSwitchIds,
    ])
    allEdges = topology.connections
      .filter(c => allNodeIds.has(c.source) && allNodeIds.has(c.target))
      .map(c => {
        const sourceIsBoard = currentRack.boards.some(b => b.id === c.source)
        const targetIsBoard = currentRack.boards.some(b => b.id === c.target)
        const sourceIsRackSwitch = rackSwitchIds.has(c.source)
        const targetIsRackSwitch = rackSwitchIds.has(c.target)
        let connectionType: 'intra_upper' | 'intra_lower' | 'inter_level' = 'inter_level'
        // 上层连接：Board之间 或 Board与Rack级Switch之间 或 Rack级Switch之间
        if ((sourceIsBoard || sourceIsRackSwitch) && (targetIsBoard || targetIsRackSwitch)) {
          connectionType = 'intra_upper'
        } else if (!sourceIsBoard && !targetIsBoard && !sourceIsRackSwitch && !targetIsRackSwitch) {
          // 下层连接：同一Board内的Chip/Switch之间
          const sourceBoard = currentRack.boards.find(b =>
            b.chips.some(ch => ch.id === c.source) ||
            getSwitchIds(topology.switches, 'inter_chip', b.id).has(c.source)
          )
          const targetBoard = currentRack.boards.find(b =>
            b.chips.some(ch => ch.id === c.target) ||
            getSwitchIds(topology.switches, 'inter_chip', b.id).has(c.target)
          )
          if (sourceBoard && targetBoard && sourceBoard.id === targetBoard.id) {
            connectionType = 'intra_lower'
          }
        }
        return { source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType }
      })
  } else if (levelPair === 'board_chip' && currentBoard) {
    upperNodes = currentBoard.chips.map(chip => ({
      id: chip.id,
      label: chip.label || chip.id,
      type: 'chip',
      x: 0, y: 0,
      color: CHIP_TYPE_COLORS[chip.type] || '#666',
      hierarchyLevel: 'board' as HierarchyLevel,
    }))
    graphTitle = `${currentBoard.label} - Chip 拓扑`
    const chipIds = new Set(currentBoard.chips.map(c => c.id))
    allEdges = topology.connections
      .filter(c => chipIds.has(c.source) && chipIds.has(c.target))
      .map(c => ({ source: c.source, target: c.target, bandwidth: c.bandwidth, latency: c.latency, connectionType: 'intra_upper' as const }))
  }

  // 应用堆叠布局
  if (upperNodes.length > 0) {
    const layoutResult = isometricStackedLayout(upperNodes, lowerNodesMap, width, height)

    // 为每个容器计算单层级布局数据
    layoutResult.upperNodes.forEach(containerNode => {
      const children = lowerNodesMap.get(containerNode.id) || []
      if (children.length === 0) return

      const bounds = containerNode.containerBounds
      if (!bounds) return

      const singleLevelWidth = 800
      const singleLevelHeight = 600

      let directTopology = 'full_mesh'
      if (topology.switch_config) {
        if (containerNode.type === 'pod') {
          directTopology = topology.switch_config.inter_rack?.direct_topology || 'full_mesh'
        } else if (containerNode.type === 'rack') {
          directTopology = topology.switch_config.inter_board?.direct_topology || 'full_mesh'
        } else if (containerNode.type === 'board') {
          directTopology = topology.switch_config.inter_chip?.direct_topology || 'full_mesh'
        }
      }

      // 分离Switch节点和设备节点
      const { switchNodes: containerSwitchNodes, deviceNodes: containerDeviceNodes } = separateSwitchAndDeviceNodes(children)
      const childIds = new Set(children.map(c => c.id))
      const containerEdges = allEdges.filter(e =>
        childIds.has(e.source) && childIds.has(e.target) &&
        e.connectionType === 'intra_lower'
      )

      let layoutedChildren: Node[]
      let containerSwitchPanelWidth = 0

      if (containerSwitchNodes.length > 0) {
        // 有Switch：使用Switch面板布局
        const switchPanelResult = computeSwitchPanelLayout(containerSwitchNodes, containerEdges, singleLevelHeight)
        containerSwitchPanelWidth = switchPanelResult.panelWidth
        const layoutedSwitchNodes = switchPanelResult.switchNodes

        // 计算设备区域的实际宽度和偏移
        const deviceAreaWidth = singleLevelWidth - switchPanelResult.deviceAreaOffset
        const deviceAreaOffsetX = switchPanelResult.deviceAreaOffset

        // 对设备节点应用布局
        let layoutedDevices: Node[]
        if (layoutType === 'circle') {
          const radius = Math.min(deviceAreaWidth, singleLevelHeight) * 0.35
          layoutedDevices = circleLayout(containerDeviceNodes, deviceAreaWidth / 2, singleLevelHeight / 2, radius)
        } else if (layoutType === 'grid') {
          layoutedDevices = torusLayout(containerDeviceNodes, deviceAreaWidth, singleLevelHeight)
        } else if (layoutType === 'force') {
          const deviceIds = new Set(containerDeviceNodes.map(n => n.id))
          const deviceEdges = containerEdges.filter(e => deviceIds.has(e.source) && deviceIds.has(e.target))
          layoutedDevices = forceDirectedLayout(containerDeviceNodes, deviceEdges, deviceAreaWidth, singleLevelHeight, {
            chargeStrength: -300,
            linkDistance: 100,
            collisionRadius: 35,
          })
        } else {
          // auto模式
          layoutedDevices = getLayoutForTopology(directTopology, containerDeviceNodes, deviceAreaWidth, singleLevelHeight)
        }

        // 偏移设备节点位置
        layoutedDevices = layoutedDevices.map(node => ({
          ...node,
          x: node.x + deviceAreaOffsetX,
        }))

        // 合并Switch和设备节点
        layoutedChildren = [...layoutedSwitchNodes, ...layoutedDevices]
      } else {
        // 无Switch：使用原有布局逻辑
        if (layoutType === 'circle') {
          const radius = Math.min(singleLevelWidth, singleLevelHeight) * 0.35
          layoutedChildren = circleLayout(containerDeviceNodes, singleLevelWidth / 2, singleLevelHeight / 2, radius)
        } else if (layoutType === 'grid') {
          layoutedChildren = torusLayout(containerDeviceNodes, singleLevelWidth, singleLevelHeight)
        } else if (layoutType === 'force') {
          layoutedChildren = forceDirectedLayout(containerDeviceNodes, containerEdges, singleLevelWidth, singleLevelHeight, {
            chargeStrength: -300,
            linkDistance: 100,
            collisionRadius: 35,
          })
        } else {
          layoutedChildren = getLayoutForTopology(directTopology, containerDeviceNodes, singleLevelWidth, singleLevelHeight)
        }
      }

      const containerPadding = 40
      const availableWidth = bounds.width - containerPadding * 2
      const availableHeight = bounds.height - containerPadding * 2
      const scaleX = availableWidth / singleLevelWidth
      const scaleY = availableHeight / singleLevelHeight
      const scale = Math.min(scaleX, scaleY)

      containerNode.singleLevelData = {
        nodes: layoutedChildren,
        edges: containerEdges,
        viewBox: { width: singleLevelWidth, height: singleLevelHeight },
        scale,
        directTopology,
        switchPanelWidth: containerSwitchPanelWidth,
      }
    })

    const allNodes = [...layoutResult.upperNodes, ...layoutResult.lowerNodes]
    return {
      nodes: allNodes,
      edges: allEdges,
      title: graphTitle,
      directTopology: 'full_mesh',
      switchPanelWidth: 0,  // 多层级模式下Switch面板在各容器内处理
    }
  }

  return { nodes: [], edges: [], title: graphTitle, directTopology: 'full_mesh', switchPanelWidth: 0 }
}

// ==========================================
// 单层级数据计算
// ==========================================
interface SingleLevelParams {
  topology: HierarchicalTopology
  currentLevel: 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  layoutType: LayoutType
  manualConnections: ManualConnection[]
  width: number
  height: number
}

function computeSingleLevelData(params: SingleLevelParams): TopologyDataResult {
  const { topology, currentLevel, currentPod, currentRack, currentBoard, layoutType, manualConnections, width, height } = params

  let nodeList: Node[] = []
  let edgeList: Edge[] = []
  let graphTitle = ''

  if (currentLevel === 'datacenter') {
    const result = computeDatacenterLevel(topology)
    nodeList = result.nodes
    edgeList = result.edges
    graphTitle = result.title
  } else if (currentLevel === 'pod' && currentPod) {
    const result = computePodLevel(topology, currentPod, manualConnections)
    nodeList = result.nodes
    edgeList = result.edges
    graphTitle = result.title
  } else if (currentLevel === 'rack' && currentRack) {
    const result = computeRackLevel(topology, currentRack, manualConnections)
    nodeList = result.nodes
    edgeList = result.edges
    graphTitle = result.title
  } else if (currentLevel === 'board' && currentBoard) {
    const result = computeBoardLevel(topology, currentBoard, manualConnections)
    nodeList = result.nodes
    edgeList = result.edges
    graphTitle = result.title
  }

  // 获取拓扑类型
  let directTopology = 'full_mesh'
  let keepDirectTopology = false
  if (topology.switch_config) {
    if (currentLevel === 'datacenter') {
      const dcConfig = topology.switch_config.inter_pod
      directTopology = dcConfig?.direct_topology || 'full_mesh'
      keepDirectTopology = dcConfig?.enabled && dcConfig?.keep_direct_topology || false
    } else if (currentLevel === 'pod') {
      const podConfig = topology.switch_config.inter_rack
      directTopology = podConfig?.direct_topology || 'full_mesh'
      keepDirectTopology = podConfig?.enabled && podConfig?.keep_direct_topology || false
    } else if (currentLevel === 'rack') {
      const rackConfig = topology.switch_config.inter_board
      directTopology = rackConfig?.direct_topology || 'full_mesh'
      keepDirectTopology = rackConfig?.enabled && rackConfig?.keep_direct_topology || false
    } else if (currentLevel === 'board') {
      const boardConfig = topology.switch_config.inter_chip
      directTopology = boardConfig?.direct_topology || 'full_mesh'
      keepDirectTopology = boardConfig?.enabled && boardConfig?.keep_direct_topology || false
    }
  }

  // 分离Switch节点和设备节点
  const { switchNodes, deviceNodes } = separateSwitchAndDeviceNodes(nodeList)

  // 如果有Switch，计算Switch面板布局
  let switchPanelWidth = 0
  let layoutedSwitchNodes: Node[] = []

  if (switchNodes.length > 0) {
    const switchPanelResult = computeSwitchPanelLayout(switchNodes, edgeList, height)
    switchPanelWidth = switchPanelResult.panelWidth
    layoutedSwitchNodes = switchPanelResult.switchNodes
    // switchEdges 在渲染时从 edges 中动态过滤，这里不需要存储

    // 计算设备区域的实际宽度
    const deviceAreaWidth = width - switchPanelResult.deviceAreaOffset
    const deviceAreaOffsetX = switchPanelResult.deviceAreaOffset

    // 对设备节点应用布局
    let layoutedDevices: Node[]
    if (layoutType === 'circle') {
      const radius = Math.min(deviceAreaWidth, height) * 0.35
      layoutedDevices = circleLayout(deviceNodes, deviceAreaWidth / 2, height / 2, radius)
    } else if (layoutType === 'grid') {
      layoutedDevices = torusLayout(deviceNodes, deviceAreaWidth, height)
    } else if (layoutType === 'force') {
      // 过滤出设备间的边
      const deviceIds = new Set(deviceNodes.map(n => n.id))
      const deviceEdges = edgeList.filter(e => deviceIds.has(e.source) && deviceIds.has(e.target))
      layoutedDevices = forceDirectedLayout(deviceNodes, deviceEdges, deviceAreaWidth, height, {
        chargeStrength: -300,
        linkDistance: 100,
        collisionRadius: 35,
      })
    } else {
      // auto模式
      if (keepDirectTopology && directTopology !== 'none') {
        layoutedDevices = getLayoutForTopology(directTopology, deviceNodes, deviceAreaWidth, height)
      } else {
        layoutedDevices = getLayoutForTopology(directTopology, deviceNodes, deviceAreaWidth, height)
      }
    }

    // 偏移设备节点位置（右移Switch面板宽度）
    layoutedDevices = layoutedDevices.map(node => ({
      ...node,
      x: node.x + deviceAreaOffsetX,
    }))

    // 合并Switch和设备节点
    nodeList = [...layoutedSwitchNodes, ...layoutedDevices]
  } else {
    // 无Switch时，使用原有逻辑
    if (layoutType === 'circle') {
      const radius = Math.min(width, height) * 0.35
      nodeList = circleLayout(deviceNodes, width / 2, height / 2, radius)
    } else if (layoutType === 'grid') {
      nodeList = torusLayout(deviceNodes, width, height)
    } else if (layoutType === 'force') {
      nodeList = forceDirectedLayout(deviceNodes, edgeList, width, height, {
        chargeStrength: -300,
        linkDistance: 100,
        collisionRadius: 35,
      })
    } else {
      nodeList = getLayoutForTopology(directTopology, deviceNodes, width, height)
    }
  }

  return { nodes: nodeList, edges: edgeList, title: graphTitle, directTopology, switchPanelWidth }
}

// ==========================================
// 各层级计算函数
// ==========================================
function computeDatacenterLevel(topology: HierarchicalTopology): { nodes: Node[]; edges: Edge[]; title: string } {
  const graphTitle = '数据中心拓扑'
  const nodeList: Node[] = topology.pods.map((pod) => ({
    id: pod.id,
    label: pod.label,
    type: 'pod',
    x: 0,
    y: 0,
    color: '#1890ff',
  }))

  nodeList.push(...convertSwitchesToNodes(topology.switches, 'inter_pod'))

  const podIds = new Set(topology.pods.map(p => p.id))
  const dcSwitchIds = getSwitchIds(topology.switches, 'inter_pod')
  const podSwitchToPod: Record<string, string> = {}
  ;(topology.switches || [])
    .filter((s): s is typeof s & { parent_id: string } => s.hierarchy_level === 'inter_rack' && !!s.parent_id)
    .forEach(s => { podSwitchToPod[s.id] = s.parent_id })

  const edgeList: Edge[] = topology.connections
    .filter(c => {
      const sourceValid = podIds.has(c.source) || dcSwitchIds.has(c.source)
      const targetValid = podIds.has(c.target) || dcSwitchIds.has(c.target)
      if (sourceValid && targetValid) return true
      if (dcSwitchIds.has(c.source) && podSwitchToPod[c.target]) return true
      if (dcSwitchIds.has(c.target) && podSwitchToPod[c.source]) return true
      return false
    })
    .map(c => {
      let source = c.source
      let target = c.target
      if (podSwitchToPod[c.source]) source = podSwitchToPod[c.source]
      if (podSwitchToPod[c.target]) target = podSwitchToPod[c.target]
      return {
        source,
        target,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: c.type === 'switch',
      }
    })

  return { nodes: nodeList, edges: edgeList, title: graphTitle }
}

function computePodLevel(topology: HierarchicalTopology, currentPod: PodConfig, manualConnections: ManualConnection[]): { nodes: Node[]; edges: Edge[]; title: string } {
  const graphTitle = `${currentPod.label} - Rack拓扑`
  const nodeList: Node[] = currentPod.racks.map((rack) => ({
    id: rack.id,
    label: rack.label,
    type: 'rack',
    x: 0,
    y: 0,
    color: '#52c41a',
  }))

  nodeList.push(...convertSwitchesToNodes(topology.switches, 'inter_rack', currentPod.id))

  const rackIds = new Set(currentPod.racks.map(r => r.id))
  const podSwitchIds = getSwitchIds(topology.switches, 'inter_rack', currentPod.id)
  const rackSwitchToRack: Record<string, string> = {}
  ;(topology.switches || [])
    .filter((s): s is typeof s & { parent_id: string } => s.hierarchy_level === 'inter_board' && !!s.parent_id && rackIds.has(s.parent_id))
    .forEach(s => { rackSwitchToRack[s.id] = s.parent_id })

  const dcSwitchIds = getSwitchIds(topology.switches, 'inter_pod')
  const dcSwitchLabels = buildSwitchLabelsMap(topology.switches, 'inter_pod')

  const allPodConnections = mergeManualConnections(topology.connections, manualConnections)

  const edgeList: Edge[] = allPodConnections
    .filter(c => {
      const sourceInPod = rackIds.has(c.source) || podSwitchIds.has(c.source)
      const targetInPod = rackIds.has(c.target) || podSwitchIds.has(c.target)
      if (sourceInPod && targetInPod) return true
      if (podSwitchIds.has(c.source) && rackSwitchToRack[c.target]) return true
      if (podSwitchIds.has(c.target) && rackSwitchToRack[c.source]) return true
      return false
    })
    .map(c => {
      let source = c.source
      let target = c.target
      if (rackSwitchToRack[c.source]) source = rackSwitchToRack[c.source]
      if (rackSwitchToRack[c.target]) target = rackSwitchToRack[c.target]
      return {
        source,
        target,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: c.type === 'switch',
      }
    })

  // 添加外部连接
  topology.connections
    .filter(c => {
      const podSwitchToDC = podSwitchIds.has(c.source) && dcSwitchIds.has(c.target)
      const dcToPodSwitch = dcSwitchIds.has(c.source) && podSwitchIds.has(c.target)
      return podSwitchToDC || dcToPodSwitch
    })
    .forEach(c => {
      const isSourcePodSwitch = podSwitchIds.has(c.source)
      const internalNode = isSourcePodSwitch ? c.source : c.target
      const externalNode = isSourcePodSwitch ? c.target : c.source
      edgeList.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: true,
        isExternal: true,
        externalDirection: 'upper',
        externalNodeId: externalNode,
        externalNodeLabel: dcSwitchLabels[externalNode] || externalNode,
      })
    })

  // 添加间接连接
  const dcSwitchToNodes: Record<string, string[]> = {}
  topology.connections
    .filter(c => {
      const podSwitchToDC = podSwitchIds.has(c.source) && dcSwitchIds.has(c.target)
      const dcToPodSwitch = dcSwitchIds.has(c.source) && podSwitchIds.has(c.target)
      return podSwitchToDC || dcToPodSwitch
    })
    .forEach(c => {
      const dcSwitch = dcSwitchIds.has(c.source) ? c.source : c.target
      const podNode = dcSwitchIds.has(c.source) ? c.target : c.source
      if (!dcSwitchToNodes[dcSwitch]) dcSwitchToNodes[dcSwitch] = []
      if (!dcSwitchToNodes[dcSwitch].includes(podNode)) {
        dcSwitchToNodes[dcSwitch].push(podNode)
      }
    })

  const addedIndirectPairs = new Set<string>()
  Object.entries(dcSwitchToNodes).forEach(([dcSwitchId, nodes]) => {
    if (nodes.length < 2) return
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pairKey = [nodes[i], nodes[j]].sort().join('-')
        if (addedIndirectPairs.has(pairKey)) continue
        addedIndirectPairs.add(pairKey)
        edgeList.push({
          source: nodes[i],
          target: nodes[j],
          isSwitch: true,
          isIndirect: true,
          viaNodeId: dcSwitchId,
          viaNodeLabel: dcSwitchLabels[dcSwitchId] || dcSwitchId,
        })
      }
    }
  })

  // 添加跨层级连接
  const currentPodNodeIds = new Set([...rackIds, ...podSwitchIds])
  currentPodNodeIds.add(currentPod.id)

  allPodConnections
    .filter(c => {
      const sourceInPod = rackIds.has(c.source)
      const targetInPod = rackIds.has(c.target)
      if (!((sourceInPod && !targetInPod) || (!sourceInPod && targetInPod))) return false
      const externalNode = sourceInPod ? c.target : c.source
      if (dcSwitchIds.has(externalNode) || podSwitchIds.has(externalNode)) return false
      return true
    })
    .forEach(c => {
      const isSourceInternal = rackIds.has(c.source)
      const internalNode = isSourceInternal ? c.source : c.target
      const externalNode = isSourceInternal ? c.target : c.source

      let externalDirection: 'upper' | 'lower' = 'upper'
      let externalLabel = externalNode

      const externalParts = externalNode.split('/')
      const internalParts = currentPod.id.split('/')

      if (externalParts.length > internalParts.length) {
        externalDirection = 'lower'
      }

      if (externalParts.length >= 1) {
        externalLabel = externalParts.slice(-2).join('/').replace(/_/g, ' ')
      }

      edgeList.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: false,
        isExternal: true,
        externalDirection,
        externalNodeId: externalNode,
        externalNodeLabel: externalLabel,
      })
    })

  return { nodes: nodeList, edges: edgeList, title: graphTitle }
}

function computeRackLevel(topology: HierarchicalTopology, currentRack: RackConfig, manualConnections: ManualConnection[]): { nodes: Node[]; edges: Edge[]; title: string } {
  const graphTitle = `${currentRack.label} - Board拓扑`
  const nodeList: Node[] = currentRack.boards.map((board) => ({
    id: board.id,
    label: board.label,
    type: 'board',
    x: 0,
    y: 0,
    color: BOARD_U_COLORS[board.u_height] || '#722ed1',
    uHeight: board.u_height,
  }))

  nodeList.push(...convertSwitchesToNodes(topology.switches, 'inter_board', currentRack.id))

  const boardIds = new Set(currentRack.boards.map(b => b.id))
  const rackSwitchIds = getSwitchIds(topology.switches, 'inter_board', currentRack.id)
  const boardSwitchToBoard: Record<string, string> = {}
  ;(topology.switches || [])
    .filter(s => s.hierarchy_level === 'inter_chip' && s.parent_id?.startsWith(currentRack.id))
    .forEach(s => { boardSwitchToBoard[s.id] = s.parent_id! })

  const parentPodId = topology.pods.find(p => p.racks.some(r => r.id === currentRack.id))?.id
  const podSwitchIds = getSwitchIds(topology.switches, 'inter_rack', parentPodId)
  const podSwitchLabels = buildSwitchLabelsMap(topology.switches, 'inter_rack', parentPodId)

  const edgeList: Edge[] = topology.connections
    .filter(c => {
      const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
      const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
      if (sourceInRack && targetInRack) return true
      if (rackSwitchIds.has(c.source) && boardSwitchToBoard[c.target]) return true
      if (rackSwitchIds.has(c.target) && boardSwitchToBoard[c.source]) return true
      return false
    })
    .map(c => {
      let source = c.source
      let target = c.target
      if (boardSwitchToBoard[c.source]) source = boardSwitchToBoard[c.source]
      if (boardSwitchToBoard[c.target]) target = boardSwitchToBoard[c.target]
      return {
        source,
        target,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: c.type === 'switch',
      }
    })

  // 添加外部连接
  topology.connections
    .filter(c => {
      const rackSwitchToPod = rackSwitchIds.has(c.source) && podSwitchIds.has(c.target)
      const podToRackSwitch = podSwitchIds.has(c.source) && rackSwitchIds.has(c.target)
      return rackSwitchToPod || podToRackSwitch
    })
    .forEach(c => {
      const isSourceRackSwitch = rackSwitchIds.has(c.source)
      const internalNode = isSourceRackSwitch ? c.source : c.target
      const externalNode = isSourceRackSwitch ? c.target : c.source
      edgeList.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: true,
        isExternal: true,
        externalDirection: 'upper',
        externalNodeId: externalNode,
        externalNodeLabel: podSwitchLabels[externalNode] || externalNode,
      })
    })

  // 添加间接连接
  const podSwitchToNodes: Record<string, string[]> = {}
  topology.connections
    .filter(c => {
      const rackSwitchToPod = rackSwitchIds.has(c.source) && podSwitchIds.has(c.target)
      const podToRackSwitch = podSwitchIds.has(c.source) && rackSwitchIds.has(c.target)
      return rackSwitchToPod || podToRackSwitch
    })
    .forEach(c => {
      const podSwitch = podSwitchIds.has(c.source) ? c.source : c.target
      const rackNode = podSwitchIds.has(c.source) ? c.target : c.source
      if (!podSwitchToNodes[podSwitch]) podSwitchToNodes[podSwitch] = []
      if (!podSwitchToNodes[podSwitch].includes(rackNode)) {
        podSwitchToNodes[podSwitch].push(rackNode)
      }
    })

  const addedIndirectPairs = new Set<string>()
  Object.entries(podSwitchToNodes).forEach(([podSwitchId, nodes]) => {
    if (nodes.length < 2) return
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pairKey = [nodes[i], nodes[j]].sort().join('-')
        if (addedIndirectPairs.has(pairKey)) continue
        addedIndirectPairs.add(pairKey)
        edgeList.push({
          source: nodes[i],
          target: nodes[j],
          isSwitch: true,
          isIndirect: true,
          viaNodeId: podSwitchId,
          viaNodeLabel: podSwitchLabels[podSwitchId] || podSwitchId,
        })
      }
    }
  })

  // 添加跨层级连接
  const currentRackNodeIds = new Set([...boardIds, ...rackSwitchIds])
  currentRackNodeIds.add(currentRack.id)

  const allConnections = mergeManualConnections(topology.connections, manualConnections)

  allConnections
    .filter(c => {
      const sourceInRack = boardIds.has(c.source)
      const targetInRack = boardIds.has(c.target)
      if (!((sourceInRack && !targetInRack) || (!sourceInRack && targetInRack))) return false
      const externalNode = sourceInRack ? c.target : c.source
      if (podSwitchIds.has(externalNode) || rackSwitchIds.has(externalNode)) return false
      return true
    })
    .forEach(c => {
      const isSourceInternal = boardIds.has(c.source)
      const internalNode = isSourceInternal ? c.source : c.target
      const externalNode = isSourceInternal ? c.target : c.source

      let externalDirection: 'upper' | 'lower' = 'upper'
      let externalLabel = externalNode

      const externalParts = externalNode.split('/')
      const internalParts = currentRack.id.split('/')

      if (externalParts.length > internalParts.length) {
        externalDirection = 'lower'
      }

      if (externalParts.length >= 2) {
        externalLabel = externalParts.slice(-2).join('/').replace(/_/g, ' ')
      }

      edgeList.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: false,
        isExternal: true,
        externalDirection,
        externalNodeId: externalNode,
        externalNodeLabel: externalLabel,
      })
    })

  return { nodes: nodeList, edges: edgeList, title: graphTitle }
}

function computeBoardLevel(topology: HierarchicalTopology, currentBoard: BoardConfig, manualConnections: ManualConnection[]): { nodes: Node[]; edges: Edge[]; title: string } {
  const graphTitle = `${currentBoard.label} - Chip拓扑`
  const nodeList: Node[] = currentBoard.chips.map((chip) => ({
    id: chip.id,
    label: chip.label || chip.id,
    type: 'chip',
    x: 0,
    y: 0,
    color: CHIP_TYPE_COLORS[chip.type] || '#666',
  }))

  nodeList.push(...convertSwitchesToNodes(topology.switches, 'inter_chip', currentBoard.id))

  const chipIds = new Set(currentBoard.chips.map(c => c.id))
  const boardSwitchIds = getSwitchIds(topology.switches, 'inter_chip', currentBoard.id)

  let parentRackId: string | undefined
  for (const pod of topology.pods) {
    for (const rack of pod.racks) {
      if (rack.boards.some(b => b.id === currentBoard.id)) {
        parentRackId = rack.id
        break
      }
    }
    if (parentRackId) break
  }

  const rackSwitchIds = getSwitchIds(topology.switches, 'inter_board', parentRackId)
  const rackSwitchLabels = buildSwitchLabelsMap(topology.switches, 'inter_board', parentRackId)

  const allBoardConnections = mergeManualConnections(topology.connections, manualConnections)

  const edgeList: Edge[] = allBoardConnections
    .filter(c => {
      const sourceInBoard = chipIds.has(c.source) || boardSwitchIds.has(c.source)
      const targetInBoard = chipIds.has(c.target) || boardSwitchIds.has(c.target)
      return sourceInBoard && targetInBoard
    })
    .map(c => ({
      source: c.source,
      target: c.target,
      bandwidth: c.bandwidth,
      isSwitch: c.type === 'switch',
      latency: c.latency,
    }))

  // 添加外部连接
  allBoardConnections
    .filter(c => {
      const boardSwitchToRack = boardSwitchIds.has(c.source) && rackSwitchIds.has(c.target)
      const rackToBoardSwitch = rackSwitchIds.has(c.source) && boardSwitchIds.has(c.target)
      return boardSwitchToRack || rackToBoardSwitch
    })
    .forEach(c => {
      const isSourceBoardSwitch = boardSwitchIds.has(c.source)
      const internalNode = isSourceBoardSwitch ? c.source : c.target
      const externalNode = isSourceBoardSwitch ? c.target : c.source
      edgeList.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: true,
        isExternal: true,
        externalDirection: 'upper',
        externalNodeId: externalNode,
        externalNodeLabel: rackSwitchLabels[externalNode] || externalNode,
      })
    })

  // 添加间接连接
  const rackSwitchToNodes: Record<string, string[]> = {}
  allBoardConnections
    .filter(c => {
      const boardSwitchToRack = boardSwitchIds.has(c.source) && rackSwitchIds.has(c.target)
      const rackToBoardSwitch = rackSwitchIds.has(c.source) && boardSwitchIds.has(c.target)
      return boardSwitchToRack || rackToBoardSwitch
    })
    .forEach(c => {
      const rackSwitch = rackSwitchIds.has(c.source) ? c.source : c.target
      const boardNode = rackSwitchIds.has(c.source) ? c.target : c.source
      if (!rackSwitchToNodes[rackSwitch]) rackSwitchToNodes[rackSwitch] = []
      if (!rackSwitchToNodes[rackSwitch].includes(boardNode)) {
        rackSwitchToNodes[rackSwitch].push(boardNode)
      }
    })

  const addedIndirectPairs = new Set<string>()
  Object.entries(rackSwitchToNodes).forEach(([rackSwitchId, nodes]) => {
    if (nodes.length < 2) return
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pairKey = [nodes[i], nodes[j]].sort().join('-')
        if (addedIndirectPairs.has(pairKey)) continue
        addedIndirectPairs.add(pairKey)
        edgeList.push({
          source: nodes[i],
          target: nodes[j],
          isSwitch: true,
          isIndirect: true,
          viaNodeId: rackSwitchId,
          viaNodeLabel: rackSwitchLabels[rackSwitchId] || rackSwitchId,
        })
      }
    }
  })

  return { nodes: nodeList, edges: edgeList, title: graphTitle }
}

export default computeTopologyData
