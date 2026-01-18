import React from 'react'
import { Node, Edge, LayoutType, ManualConnection, getNodeEdgePoint } from './shared'
import { ManualConnectionLine, TorusArcs } from './components'
import { HierarchyLevel } from '../../types'

// 多层级视图渲染所需的props
export interface MultiLevelViewProps {
  // 数据
  displayNodes: Node[]
  edges: Edge[]
  manualConnections: ManualConnection[]
  // 状态
  zoom: number
  selectedNodeId: string | null
  selectedLinkId: string | null
  hoveredLayerIndex: number | null
  setHoveredLayerIndex: (idx: number | null) => void
  // 展开/收缩动画
  expandingContainer: { id: string; type: string } | null
  collapsingContainer: { id: string; type: string } | null
  collapseAnimationStarted: boolean
  // 动画完成回调
  onExpandAnimationEnd?: (nodeId: string, nodeType: string) => void
  // 回调
  connectionMode: 'view' | 'select' | 'connect' | 'select_source' | 'select_target'
  onNodeClick?: (node: any) => void
  onLinkClick?: (link: any) => void
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  // 其他
  layoutType: LayoutType
  renderNode: (node: Node, options: { keyPrefix: string; scale?: number; isSelected?: boolean; onClick?: () => void; useMultiLevelConfig?: boolean }) => JSX.Element
  getCurrentHierarchyLevel: () => HierarchyLevel
  // 拖拽相关
  handleDragMove: (e: React.MouseEvent) => void
  handleDragEnd: () => void
  // 选中节点
  selectedNodes: Set<string>
  targetNodes: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  onTargetNodesChange?: (nodes: Set<string>) => void
}

export const MultiLevelView: React.FC<MultiLevelViewProps> = ({
  displayNodes,
  edges,
  manualConnections,
  zoom,
  selectedNodeId,
  selectedLinkId,
  hoveredLayerIndex,
  setHoveredLayerIndex,
  expandingContainer,
  collapsingContainer,
  collapseAnimationStarted,
  onExpandAnimationEnd,
  connectionMode,
  onNodeClick,
  onLinkClick,
  onNodeDoubleClick,
  layoutType,
  renderNode,
  getCurrentHierarchyLevel,
  handleDragMove,
  handleDragEnd,
  selectedNodes,
  targetNodes,
  onSelectedNodesChange,
  onTargetNodesChange,
}) => {
  const containers = displayNodes
    .filter(n => n.isContainer && n.containerBounds)
    .sort((a, b) => (b.zLayer ?? 0) - (a.zLayer ?? 0))  // 远处先渲染

  // 悬停时上面的层向上移动的距离
  const liftDistance = 150

  // 查找选中容器的 zLayer（支持选中容器或容器内节点，用于保持选中时的移动效果）
  let selectedLayerIndex: number | null = null
  const selectedContainer = containers.find(c => c.id === selectedNodeId)
  if (selectedContainer) {
    selectedLayerIndex = selectedContainer.zLayer ?? null
  } else if (selectedNodeId) {
    for (const container of containers) {
      if (container.singleLevelData?.nodes.some(n => n.id === selectedNodeId)) {
        selectedLayerIndex = container.zLayer ?? null
        break
      }
    }
  }

  // 获取所有inter_level边（用于跨层级连线渲染）
  const interLevelEdges = edges.filter(e => e.connectionType === 'inter_level')
  const baseSkewAngle = -25
  const skewTan = Math.tan(baseSkewAngle * Math.PI / 180)

  // 上层Switch节点（容器外的Switch）- 需要在containersRendered之前定义
  const upperSwitchNodes = displayNodes.filter(n => n.isSwitch && n.inSwitchPanel)

  // 计算选中link的两端节点（用于高亮）
  const selectedLinkEndpoints = new Set<string>()
  if (selectedLinkId) {
    const selectedConn = manualConnections.find(conn => {
      const edgeId = `${conn.source}-${conn.target}`
      return selectedLinkId === edgeId || selectedLinkId === `${conn.target}-${conn.source}`
    })
    if (selectedConn) {
      selectedLinkEndpoints.add(selectedConn.source)
      selectedLinkEndpoints.add(selectedConn.target)
    }
  }

  // 计算节点位置（考虑yOffset和skew变换）
  const getNodePosition = (nodeId: string, activeLayerIdx: number | null) => {
    // 首先检查是否是容器外的Switch节点
    const upperSwitch = displayNodes.find(n => n.id === nodeId && n.isSwitch && n.inSwitchPanel)
    if (upperSwitch) {
      // 容器外Switch节点直接使用其坐标，不受skew变换影响
      return { x: upperSwitch.x, y: upperSwitch.y, zLayer: -1 }
    }

    // 然后在容器内查找
    for (const container of containers) {
      if (container.singleLevelData) {
        const slNode = container.singleLevelData.nodes.find(n => n.id === nodeId)
        if (slNode && container.containerBounds) {
          const bounds = container.containerBounds
          const viewBox = container.singleLevelData.viewBox
          const zLayer = container.zLayer ?? 0

          let yOffset = 0
          if (activeLayerIdx !== null && zLayer < activeLayerIdx) {
            yOffset = -liftDistance
          }

          const labelHeight = 10
          const sidePadding = 10
          const topPadding = 10
          const svgWidth = bounds.width - sidePadding * 2
          const svgHeight = bounds.height - labelHeight - topPadding
          const svgX = bounds.x + sidePadding
          const svgY = bounds.y + topPadding

          const scaleX = svgWidth / viewBox.width
          const scaleY = svgHeight / viewBox.height
          const scale = Math.min(scaleX, scaleY)

          const offsetX = (svgWidth - viewBox.width * scale) / 2
          const offsetY = (svgHeight - viewBox.height * scale) / 2

          const baseX = svgX + offsetX + slNode.x * scale
          const baseY = svgY + offsetY + slNode.y * scale
          const centerY = bounds.y + bounds.height / 2
          const skewedX = baseX + (baseY - centerY) * skewTan

          return { x: skewedX, y: baseY + yOffset, zLayer }
        }
      }
    }
    return null
  }

  // 视口中心
  const viewportCenterX = 400
  const viewportCenterY = 300

  // 获取当前层级的手动连接
  const currentManualConnections = manualConnections.filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())

  // 计算容器的公共状态（用于三层分离渲染）
  const getContainerState = (containerNode: Node) => {
    const bounds = containerNode.containerBounds!
    const zLayer = containerNode.zLayer ?? 0
    const isExpanding = expandingContainer?.id === containerNode.id
    const isOtherExpanding = expandingContainer !== null && !isExpanding
    const isCollapsing = collapsingContainer?.id === containerNode.id
    const isOtherCollapsing = collapsingContainer !== null && !isCollapsing

    const activeLayerIndex = hoveredLayerIndex ?? selectedLayerIndex
    const activeContainer = hoveredLayerIndex !== null
      ? containers.find(c => c.zLayer === hoveredLayerIndex)
      : containers.find(c => c.id === selectedNodeId)

    let yOffset = 0
    let layerOpacity = 1
    const shouldDimContainer = activeLayerIndex !== null && zLayer < activeLayerIndex && !expandingContainer && !collapsingContainer
    if (shouldDimContainer) {
      yOffset = -liftDistance
      layerOpacity = 0.15
    }

    // 获取与活跃容器有连线的节点ID集合，以及选中link两端的节点
    const getConnectedNodeIds = (): Set<string> => {
      const connectedIds = new Set<string>()

      // 如果有选中的link，添加两端节点
      if (selectedLinkId) {
        // 首先检查 manualConnections
        const selectedConn = currentManualConnections.find(conn => {
          const edgeId = `${conn.source}-${conn.target}`
          return selectedLinkId === edgeId || selectedLinkId === `${conn.target}-${conn.source}`
        })
        if (selectedConn) {
          connectedIds.add(selectedConn.source)
          connectedIds.add(selectedConn.target)
        }

        // 然后检查容器内的普通边
        if (connectedIds.size === 0 && containerNode.singleLevelData) {
          const slEdges = containerNode.singleLevelData.edges || []
          const selectedEdge = slEdges.find((edge: Edge) => {
            const edgeId = `${edge.source}-${edge.target}`
            return selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`
          })
          if (selectedEdge) {
            connectedIds.add(selectedEdge.source)
            connectedIds.add(selectedEdge.target)
          }
        }

        // 也检查跨层级边（使用 inter: 前缀）
        if (connectedIds.size === 0) {
          const selectedInterEdge = interLevelEdges.find(edge => {
            const edgeId = `inter:${edge.source}-${edge.target}`
            return selectedLinkId === edgeId || selectedLinkId === `inter:${edge.target}-${edge.source}`
          })
          if (selectedInterEdge) {
            connectedIds.add(selectedInterEdge.source)
            connectedIds.add(selectedInterEdge.target)
          }
        }
      }

      // 如果有活跃容器，添加与之有连线的节点
      if (activeContainer?.singleLevelData) {
        const activeNodeIds = new Set(activeContainer.singleLevelData.nodes.map((n: any) => n.id))
        currentManualConnections.forEach(conn => {
          if (activeNodeIds.has(conn.source)) {
            connectedIds.add(conn.target)
          }
          if (activeNodeIds.has(conn.target)) {
            connectedIds.add(conn.source)
          }
        })
      }
      return connectedIds
    }
    const connectedNodeIds = (activeContainer || selectedLinkId) ? getConnectedNodeIds() : new Set<string>()

    if (isOtherExpanding) {
      layerOpacity = 0
    }
    if (isOtherCollapsing) {
      if (!collapseAnimationStarted) {
        layerOpacity = 0
      } else {
        layerOpacity = 1
      }
    }

    const x = bounds.x
    const w = bounds.width
    const h = bounds.height

    let animX = x
    let animY = bounds.y + yOffset
    let animW = w
    let animH = h
    let animSkewAngle = baseSkewAngle
    let animOpacity = layerOpacity

    if (isExpanding) {
      const viewportWidth = 800 / zoom
      const viewportHeight = 600 / zoom
      animX = viewportCenterX - viewportWidth / 2
      animY = viewportCenterY - viewportHeight / 2
      animW = viewportWidth
      animH = viewportHeight
      animSkewAngle = 0
      animOpacity = 1
    }

    if (isCollapsing) {
      if (!collapseAnimationStarted) {
        const viewportWidth = 800 / zoom
        const viewportHeight = 600 / zoom
        animX = viewportCenterX - viewportWidth / 2
        animY = viewportCenterY - viewportHeight / 2
        animW = viewportWidth
        animH = viewportHeight
        animSkewAngle = 0
        animOpacity = 1
      } else {
        animX = x
        animY = bounds.y + yOffset
        animW = w
        animH = h
        animSkewAngle = baseSkewAngle
        animOpacity = layerOpacity
      }
    }

    const isSelected = selectedNodeId === containerNode.id
    const isHovered = activeLayerIndex === zLayer
    const isAnimating = expandingContainer || collapsingContainer
    const containerGroupOpacity = isAnimating ? animOpacity : 1

    const layerEdges = edges.filter(e => {
      const sourceInContainer = containerNode.singleLevelData?.nodes.some(n => n.id === e.source)
      const targetInContainer = containerNode.singleLevelData?.nodes.some(n => n.id === e.target)
      return sourceInContainer || targetInContainer
    })
    const layerNodes = containerNode.singleLevelData?.nodes.filter(n => !n.isSwitch) || []

    return {
      bounds, zLayer, isExpanding, isCollapsing, isAnimating,
      activeLayerIndex, shouldDimContainer, connectedNodeIds,
      animX, animY, animW, animH, animSkewAngle, animOpacity,
      isSelected, isHovered, containerGroupOpacity,
      layerEdges, layerNodes, yOffset
    }
  }

  // 第一层：容器背景
  const containerBackgrounds = containers.map(containerNode => {
    const state = getContainerState(containerNode)
    const { bounds, isAnimating,
            shouldDimContainer, animX, animY, animW, animH, animSkewAngle,
            isSelected, isHovered, containerGroupOpacity, layerEdges, layerNodes } = state

    return (
      <g
        key={`bg-${containerNode.id}`}
        style={{
          transition: isAnimating
            ? 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.5s ease'
            : 'transform 0.3s ease',
          opacity: containerGroupOpacity,
        }}
        transform={`translate(0, ${animY - bounds.y})`}
      >
        <g
          style={{
            transition: isAnimating
              ? 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
              : 'all 0.3s ease',
            transformOrigin: `${animX + animW / 2}px ${bounds.y + animH / 2}px`,
            transform: `skewX(${animSkewAngle}deg)`,
          }}
        >
          <rect
            x={animX}
            y={bounds.y}
            width={animW}
            height={animH}
            fill={isHovered ? '#f8fafc' : 'white'}
            stroke={isSelected ? '#2563eb' : isHovered ? '#6b7280' : '#e5e7eb'}
            strokeWidth={isSelected ? 2 : 1}
            rx={8}
            style={{
              cursor: connectionMode === 'view' ? 'pointer' : 'default',
              opacity: shouldDimContainer ? 0.15 : 1,
              filter: isSelected
                ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.4))'
                : isHovered
                  ? 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.15))'
                  : 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1))',
              transition: 'filter 0.2s ease, fill 0.2s ease, stroke 0.2s ease, opacity 0.2s ease',
              pointerEvents: 'auto',
            }}
            onClick={(e) => {
              e.stopPropagation()
              if (connectionMode !== 'view') return
              if (onNodeClick) {
                const containerConnections = layerEdges.map(edge => {
                  const targetNode = layerNodes.find(n => n.id === (edge.source === containerNode.id ? edge.target : edge.source))
                  return {
                    id: edge.source === containerNode.id ? edge.target : edge.source,
                    label: targetNode?.label || '',
                    bandwidth: edge.bandwidth,
                    latency: edge.latency,
                  }
                })
                onNodeClick({
                  id: containerNode.id,
                  label: containerNode.label,
                  type: containerNode.type,
                  subType: containerNode.type,
                  connections: containerConnections,
                })
              }
              onLinkClick?.(null)
            }}
            onDoubleClick={(e) => {
              e.stopPropagation()
              onNodeDoubleClick?.(containerNode.id, containerNode.type)
            }}
          />
          {/* 容器标签 */}
          <text
            x={animX + 10}
            y={bounds.y + animH - 5}
            fontSize={12}
            fill="#666"
            style={{ pointerEvents: 'none', opacity: shouldDimContainer ? 0.15 : 1, transition: 'opacity 0.2s ease' }}
          >
            {containerNode.label}
          </text>
        </g>
      </g>
    )
  })

  // 第二层：容器内边（不包含节点）
  const containerEdgesLayer = containers.map(containerNode => {
    const state = getContainerState(containerNode)
    const { bounds, zLayer: _zLayer, isAnimating, shouldDimContainer, connectedNodeIds,
            animX, animY, animW, animH, animSkewAngle, containerGroupOpacity } = state

    if (!containerNode.singleLevelData) return null

    const { nodes: slNodes, edges: slEdges, viewBox, switchPanelWidth, directTopology } = containerNode.singleLevelData
    const slSwitchPanelWidth = switchPanelWidth ?? 0
    const slSwitchNodes = slNodes.filter(n => n.isSwitch && n.inSwitchPanel)
    const labelHeight = 10
    const sidePadding = 10
    const topPadding = 10
    const svgWidth = animW - sidePadding * 2
    const svgHeight = animH - labelHeight - topPadding
    const svgX = animX + sidePadding
    const svgY = bounds.y + topPadding

    const deviceCount = slNodes.filter(n => !n.isSwitch).length
    const slNodeScale = deviceCount > 20 ? 1.2 : deviceCount > 10 ? 1.2 : 1.4

    return (
      <g
        key={`edges-${containerNode.id}`}
        style={{
          transition: isAnimating
            ? 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.5s ease'
            : 'transform 0.3s ease',
          opacity: containerGroupOpacity,
        }}
        transform={`translate(0, ${animY - bounds.y})`}
      >
        <g
          style={{
            transition: isAnimating
              ? 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
              : 'all 0.3s ease',
            transformOrigin: `${animX + animW / 2}px ${bounds.y + animH / 2}px`,
            transform: `skewX(${animSkewAngle}deg)`,
          }}
        >
          <svg
            x={svgX}
            y={svgY}
            width={svgWidth}
            height={svgHeight}
            viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
            preserveAspectRatio="xMidYMid meet"
            overflow="visible"
            style={{ pointerEvents: 'none' }}
          >
            {/* Switch面板连线 */}
            {slSwitchPanelWidth > 0 && slSwitchNodes.length > 0 && (
              <g className="container-switch-panel-edges" style={{ opacity: shouldDimContainer ? 0.15 : 1, transition: 'opacity 0.2s ease' }}>
                {(() => {
                  const switchIds = new Set(slSwitchNodes.map(n => n.id))
                  const switchInternalEdges = slEdges.filter(e =>
                    switchIds.has(e.source) && switchIds.has(e.target)
                  )
                  return switchInternalEdges.map((edge, idx) => {
                    const sourceNode = slSwitchNodes.find(n => n.id === edge.source)
                    const targetNode = slSwitchNodes.find(n => n.id === edge.target)
                    if (!sourceNode || !targetNode) return null

                    const upperNode = sourceNode.y < targetNode.y ? sourceNode : targetNode
                    const lowerNode = sourceNode.y < targetNode.y ? targetNode : sourceNode

                    const midY = (upperNode.y + lowerNode.y) / 2
                    const pathD = `M ${upperNode.x} ${upperNode.y + 10}
                                   L ${upperNode.x} ${midY}
                                   L ${lowerNode.x} ${midY}
                                   L ${lowerNode.x} ${lowerNode.y - 10}`

                    return (
                      <path
                        key={`sl-sw-edge-${idx}`}
                        d={pathD}
                        fill="none"
                        stroke="#1890ff"
                        strokeWidth={1.5}
                        strokeOpacity={0.6}
                      />
                    )
                  })
                })()}
              </g>
            )}

            {/* 普通边 */}
            {slEdges.map((edge, i) => {
              const sourceNode = slNodes.find(n => n.id === edge.source)
              const targetNode = slNodes.find(n => n.id === edge.target)
              if (!sourceNode || !targetNode) return null
              if (sourceNode.inSwitchPanel && targetNode.inSwitchPanel) return null

              const edgeId = `${edge.source}-${edge.target}`
              const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`
              const sourceConnected = connectedNodeIds.has(edge.source)
              const targetConnected = connectedNodeIds.has(edge.target)
              const edgeOpacity = shouldDimContainer && !sourceConnected && !targetConnected ? 0.15 : 0.6

              const sourceScale = sourceNode.isSwitch ? 1.8 : slNodeScale
              const targetScale = targetNode.isSwitch ? 1.8 : slNodeScale
              const sourceEdge = getNodeEdgePoint(sourceNode.x, sourceNode.y, targetNode.x, targetNode.y, sourceNode.type, false, sourceScale)
              const targetEdge = getNodeEdgePoint(targetNode.x, targetNode.y, sourceNode.x, sourceNode.y, targetNode.type, false, targetScale)

              const handleEdgeClick = (e: React.MouseEvent) => {
                e.stopPropagation()
                if (connectionMode !== 'view') return
                onLinkClick?.({
                  id: edgeId,
                  sourceId: edge.source,
                  sourceLabel: sourceNode.label,
                  sourceType: sourceNode.type,
                  targetId: edge.target,
                  targetLabel: targetNode.label,
                  targetType: targetNode.type,
                  bandwidth: edge.bandwidth,
                  latency: edge.latency,
                  isManual: false
                })
              }

              return (
                <g key={`sl-edge-${i}`}>
                  {/* 透明点击层 */}
                  <line
                    x1={sourceEdge.x}
                    y1={sourceEdge.y}
                    x2={targetEdge.x}
                    y2={targetEdge.y}
                    stroke="transparent"
                    strokeWidth={12}
                    style={{ cursor: 'pointer', pointerEvents: 'auto' }}
                    onClick={handleEdgeClick}
                  />
                  {/* 可见线条 */}
                  <line
                    x1={sourceEdge.x}
                    y1={sourceEdge.y}
                    x2={targetEdge.x}
                    y2={targetEdge.y}
                    stroke={isLinkSelected ? '#2563eb' : '#b0b0b0'}
                    strokeWidth={isLinkSelected ? 2.5 : 1}
                    strokeOpacity={isLinkSelected ? 1 : edgeOpacity}
                    style={{ pointerEvents: 'none', transition: 'stroke-opacity 0.2s ease', filter: isLinkSelected ? 'drop-shadow(0 0 4px #2563eb)' : 'none' }}
                  />
                </g>
              )
            })}

            {/* Torus/FullMesh2D 弧线连接 */}
            {(directTopology === 'torus_2d' || directTopology === 'torus_3d' || directTopology === 'full_mesh_2d') && (
              <TorusArcs
                nodes={slNodes.filter(n => !n.isSwitch)}
                directTopology={directTopology}
                opacity={shouldDimContainer ? 0.1 : 0.6}
                selectedLinkId={selectedLinkId}
                onLinkClick={onLinkClick}
                connectionMode={connectionMode}
                isManualMode={false}
              />
            )}
          </svg>
        </g>
      </g>
    )
  })

  // 第三层：容器内节点
  const containerNodesLayer = containers.map(containerNode => {
    const state = getContainerState(containerNode)
    const { bounds, zLayer: _zLayer2, isAnimating, shouldDimContainer, connectedNodeIds,
            animX, animY, animW, animH, animSkewAngle, containerGroupOpacity } = state

    if (!containerNode.singleLevelData) return null

    const { nodes: slNodes, edges: slEdges, viewBox, switchPanelWidth } = containerNode.singleLevelData
    const slSwitchPanelWidth = switchPanelWidth ?? 0
    const slSwitchNodes = slNodes.filter(n => n.isSwitch && n.inSwitchPanel)
    const labelHeight = 10
    const sidePadding = 10
    const topPadding = 10
    const svgWidth = animW - sidePadding * 2
    const svgHeight = animH - labelHeight - topPadding
    const svgX = animX + sidePadding
    const svgY = bounds.y + topPadding

    const deviceCount = slNodes.filter(n => !n.isSwitch).length
    const slNodeScale = deviceCount > 20 ? 1.2 : deviceCount > 10 ? 1.2 : 1.4
    const isActiveLayer = !shouldDimContainer

    return (
      <g
        key={`nodes-${containerNode.id}`}
        style={{
          transition: isAnimating
            ? 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.5s ease'
            : 'transform 0.3s ease',
          opacity: containerGroupOpacity,
        }}
        transform={`translate(0, ${animY - bounds.y})`}
      >
        <g
          style={{
            transition: isAnimating
              ? 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
              : 'all 0.3s ease',
            transformOrigin: `${animX + animW / 2}px ${bounds.y + animH / 2}px`,
            transform: `skewX(${animSkewAngle}deg)`,
          }}
        >
          <svg
            x={svgX}
            y={svgY}
            width={svgWidth}
            height={svgHeight}
            viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
            preserveAspectRatio="xMidYMid meet"
            overflow="visible"
            style={{ pointerEvents: 'all' }}
            onMouseMove={handleDragMove}
            onMouseUp={handleDragEnd}
            onMouseLeave={handleDragEnd}
          >
            {/* 透明背景 - 用于拖拽事件，点击事件由containerBackgrounds处理 */}
            <rect
              x={0}
              y={0}
              width={viewBox.width}
              height={viewBox.height}
              fill="rgba(0,0,0,0)"
              style={{ pointerEvents: 'none' }}
            />

            {/* Switch节点 */}
            {slSwitchPanelWidth > 0 && slSwitchNodes.length > 0 && (
              <g className="container-switch-nodes" style={{ opacity: shouldDimContainer ? 0.15 : 1, transition: 'opacity 0.2s ease' }}>
                {slSwitchNodes.map(swNode => {
                  const isSwSourceSelected = selectedNodes.has(swNode.id)
                  const isSwTargetSelected = targetNodes.has(swNode.id)
                  const canSelectSw = connectionMode === 'view' || isActiveLayer

                  return (
                    <g
                      key={`sl-sw-wrapper-${swNode.id}`}
                      style={{
                        cursor: !canSelectSw ? 'default' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
                        pointerEvents: canSelectSw ? 'all' : 'none',
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        if (!isActiveLayer) return
                        if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                          const currentSet = new Set(selectedNodes)
                          if (currentSet.has(swNode.id)) {
                            currentSet.delete(swNode.id)
                          } else {
                            currentSet.add(swNode.id)
                          }
                          onSelectedNodesChange?.(currentSet)
                        } else if (connectionMode === 'select_target') {
                          const currentSet = new Set(targetNodes)
                          if (currentSet.has(swNode.id)) {
                            currentSet.delete(swNode.id)
                          } else {
                            currentSet.add(swNode.id)
                          }
                          onTargetNodesChange?.(currentSet)
                        }
                      }}
                    >
                      {/* 选中指示器 */}
                      {(isSwSourceSelected || isSwTargetSelected) && (() => {
                        const size = { w: 110, h: 44 }
                        const padding = 6
                        const isBoth = isSwSourceSelected && isSwTargetSelected

                        if (isBoth) {
                          return (
                            <>
                              <rect
                                x={swNode.x - (size.w / 2 + padding + 4)}
                                y={swNode.y - (size.h / 2 + padding + 4)}
                                width={size.w + (padding + 4) * 2}
                                height={size.h + (padding + 4) * 2}
                                rx={8}
                                ry={8}
                                fill="none"
                                stroke="#10b981"
                                strokeWidth={2.5}
                                strokeDasharray="6 3"
                                style={{ filter: 'drop-shadow(0 0 8px #10b981) drop-shadow(0 0 16px rgba(16, 185, 129, 0.5))' }}
                              />
                              <rect
                                x={swNode.x - (size.w / 2 + padding)}
                                y={swNode.y - (size.h / 2 + padding)}
                                width={size.w + padding * 2}
                                height={size.h + padding * 2}
                                rx={6}
                                ry={6}
                                fill="none"
                                stroke="#2563eb"
                                strokeWidth={2.5}
                                strokeDasharray="6 3"
                                style={{ filter: 'drop-shadow(0 0 8px #2563eb) drop-shadow(0 0 16px rgba(37, 99, 235, 0.5))' }}
                              />
                            </>
                          )
                        }

                        const color = isSwSourceSelected ? '#2563eb' : '#10b981'
                        const glowColor = isSwSourceSelected ? 'rgba(37, 99, 235, 0.5)' : 'rgba(16, 185, 129, 0.5)'
                        return (
                          <rect
                            x={swNode.x - (size.w / 2 + padding)}
                            y={swNode.y - (size.h / 2 + padding)}
                            width={size.w + padding * 2}
                            height={size.h + padding * 2}
                            rx={6}
                            ry={6}
                            fill="none"
                            stroke={color}
                            strokeWidth={2.5}
                            strokeDasharray="6 3"
                            style={{ filter: `drop-shadow(0 0 8px ${color}) drop-shadow(0 0 16px ${glowColor})` }}
                          />
                        )
                      })()}
                      {renderNode(swNode, {
                        keyPrefix: 'sl-sw',
                        scale: 1.8,
                        isSelected: selectedNodeId === swNode.id,
                        useMultiLevelConfig: true,
                        onClick: () => {
                          if (connectionMode === 'view' && onNodeClick) {
                            const swConnections = slEdges
                              .filter(edge => edge.source === swNode.id || edge.target === swNode.id)
                              .map(edge => {
                                const otherId = edge.source === swNode.id ? edge.target : edge.source
                                const otherNode = slNodes.find(n => n.id === otherId)
                                return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                              })
                            onNodeClick({
                              id: swNode.id,
                              label: swNode.label,
                              type: swNode.type,
                              subType: swNode.subType,
                              connections: swConnections,
                            })
                          }
                        }
                      })}
                    </g>
                  )
                })}
              </g>
            )}

            {/* 设备节点 */}
            {slNodes.filter(n => !n.isSwitch && !n.inSwitchPanel).map(node => {
              const isNodeSelected = selectedNodeId === node.id
              const isSourceSelected = selectedNodes.has(node.id)
              const isTargetSelected = targetNodes.has(node.id)
              const isConnectedNode = connectedNodeIds.has(node.id)
              const isLinkEndpoint = selectedLinkId && isConnectedNode
              const nodeOpacity = shouldDimContainer && !isConnectedNode ? 0.15 : 1
              const canSelect = connectionMode === 'view' || isActiveLayer

              // 节点高亮：选中时蓝色，作为link端点时绿色
              const nodeFilter = isNodeSelected
                ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
                : isLinkEndpoint
                  ? 'drop-shadow(0 0 8px rgba(82, 196, 26, 0.6)) drop-shadow(0 0 16px rgba(82, 196, 26, 0.3))'
                  : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'

              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y}) scale(${slNodeScale})`}
                  style={{
                    cursor: !canSelect ? 'default' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
                    opacity: nodeOpacity,
                    filter: nodeFilter,
                    transition: 'filter 0.15s ease, opacity 0.2s ease',
                    pointerEvents: canSelect ? 'all' : 'none',
                  }}
                  onMouseDown={(e) => {
                    e.stopPropagation()
                    e.preventDefault()
                    if (!isActiveLayer) return
                    if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                      const currentSet = new Set(selectedNodes)
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                      onSelectedNodesChange?.(currentSet)
                    } else if (connectionMode === 'select_target') {
                      const currentSet = new Set(targetNodes)
                      if (currentSet.has(node.id)) {
                        currentSet.delete(node.id)
                      } else {
                        currentSet.add(node.id)
                      }
                      onTargetNodesChange?.(currentSet)
                    }
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    if (connectionMode === 'view' && onNodeClick) {
                      const nodeConnections = slEdges
                        .filter(edge => edge.source === node.id || edge.target === node.id)
                        .map(edge => {
                          const otherId = edge.source === node.id ? edge.target : edge.source
                          const otherNode = slNodes.find(n => n.id === otherId)
                          return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                        })
                      onNodeClick({
                        id: node.id,
                        label: node.label,
                        type: node.type,
                        subType: node.subType,
                        connections: nodeConnections,
                      })
                    }
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation()
                    onNodeDoubleClick?.(node.id, node.type)
                  }}
                >
                  {/* 选中指示器 */}
                  {(isSourceSelected || isTargetSelected) && (() => {
                    const nodeType = node.isSwitch ? 'switch' : node.type.toLowerCase()
                    const sizeMap: Record<string, { w: number; h: number }> = {
                      switch: { w: 61, h: 24 },
                      pod: { w: 56, h: 32 },
                      rack: { w: 36, h: 56 },
                      board: { w: 64, h: 36 },
                      chip: { w: 40, h: 40 },
                      default: { w: 50, h: 36 },
                    }
                    const size = sizeMap[nodeType] || sizeMap.default
                    const padding = 6
                    const isBoth = isSourceSelected && isTargetSelected

                    if (isBoth) {
                      return (
                        <>
                          <rect
                            x={-(size.w / 2 + padding + 4)}
                            y={-(size.h / 2 + padding + 4)}
                            width={size.w + (padding + 4) * 2}
                            height={size.h + (padding + 4) * 2}
                            rx={8}
                            ry={8}
                            fill="none"
                            stroke="#10b981"
                            strokeWidth={2.5}
                            strokeDasharray="6 3"
                            style={{
                              filter: 'drop-shadow(0 0 8px #10b981) drop-shadow(0 0 16px rgba(16, 185, 129, 0.5))',
                            }}
                          />
                          <rect
                            x={-(size.w / 2 + padding)}
                            y={-(size.h / 2 + padding)}
                            width={size.w + padding * 2}
                            height={size.h + padding * 2}
                            rx={6}
                            ry={6}
                            fill="none"
                            stroke="#2563eb"
                            strokeWidth={2.5}
                            strokeDasharray="6 3"
                            style={{
                              filter: 'drop-shadow(0 0 8px #2563eb) drop-shadow(0 0 16px rgba(37, 99, 235, 0.5))',
                            }}
                          />
                        </>
                      )
                    }

                    const color = isSourceSelected ? '#2563eb' : '#10b981'
                    const glowColor = isSourceSelected ? 'rgba(37, 99, 235, 0.5)' : 'rgba(16, 185, 129, 0.5)'
                    return (
                      <rect
                        x={-(size.w / 2 + padding)}
                        y={-(size.h / 2 + padding)}
                        width={size.w + padding * 2}
                        height={size.h + padding * 2}
                        rx={6}
                        ry={6}
                        fill="none"
                        stroke={color}
                        strokeWidth={2.5}
                        strokeDasharray="6 3"
                        style={{
                          filter: `drop-shadow(0 0 8px ${color}) drop-shadow(0 0 16px ${glowColor})`,
                        }}
                      />
                    )
                  })()}
                  {renderNode({ ...node, x: 0, y: 0 }, {
                    keyPrefix: 'sl-node',
                    scale: 1,
                    isSelected: isNodeSelected,
                    useMultiLevelConfig: true,
                    onClick: () => {}
                  })}
                </g>
              )
            })}
          </svg>
        </g>
      </g>
    )
  })

  // 渲染上层Switch面板
  const renderUpperSwitchPanel = () => {
    if (upperSwitchNodes.length === 0) return null

    return (
      <g className="upper-switch-panel">
        {/* Switch之间的连线 */}
        {(() => {
          const switchIds = new Set(upperSwitchNodes.map(n => n.id))
          const switchInternalEdges = edges.filter(e =>
            switchIds.has(e.source) && switchIds.has(e.target)
          )
          return switchInternalEdges.map((edge, idx) => {
            const sourceNode = upperSwitchNodes.find(n => n.id === edge.source)
            const targetNode = upperSwitchNodes.find(n => n.id === edge.target)
            if (!sourceNode || !targetNode) return null

            const upperNode = sourceNode.y < targetNode.y ? sourceNode : targetNode
            const lowerNode = sourceNode.y < targetNode.y ? targetNode : sourceNode

            const midY = (upperNode.y + lowerNode.y) / 2
            const pathD = `M ${upperNode.x} ${upperNode.y + 12}
                           L ${upperNode.x} ${midY}
                           L ${lowerNode.x} ${midY}
                           L ${lowerNode.x} ${lowerNode.y - 12}`

            return (
              <path
                key={`upper-sw-edge-${idx}`}
                d={pathD}
                fill="none"
                stroke="#1890ff"
                strokeWidth={1.5}
                strokeOpacity={0.6}
              />
            )
          })
        })()}
        {/* Switch节点 */}
        {upperSwitchNodes.map(swNode => {
          const isSourceSelected = selectedNodes.has(swNode.id)
          const isTargetSelected = targetNodes.has(swNode.id)
          const isLinkEndpoint = selectedLinkEndpoints.has(swNode.id)
          return (
            <g
              key={`upper-sw-wrapper-${swNode.id}`}
              style={{
                cursor: connectionMode !== 'view' ? 'crosshair' : 'pointer',
                filter: isLinkEndpoint
                  ? 'drop-shadow(0 0 8px rgba(82, 196, 26, 0.6)) drop-shadow(0 0 16px rgba(82, 196, 26, 0.3))'
                  : 'none',
              }}
              onMouseDown={(e) => {
                e.stopPropagation()
                e.preventDefault()
                // 在连接模式下处理选择
                if (connectionMode === 'select_source' || connectionMode === 'select' || connectionMode === 'connect') {
                  const currentSet = new Set(selectedNodes)
                  if (currentSet.has(swNode.id)) {
                    currentSet.delete(swNode.id)
                  } else {
                    currentSet.add(swNode.id)
                  }
                  onSelectedNodesChange?.(currentSet)
                } else if (connectionMode === 'select_target') {
                  const currentSet = new Set(targetNodes)
                  if (currentSet.has(swNode.id)) {
                    currentSet.delete(swNode.id)
                  } else {
                    currentSet.add(swNode.id)
                  }
                  onTargetNodesChange?.(currentSet)
                }
              }}
            >
              {/* 选中指示器（连接模式）*/}
              {(isSourceSelected || isTargetSelected) && (() => {
                const size = { w: 80, h: 30 }
                const padding = 6
                const isBoth = isSourceSelected && isTargetSelected

                if (isBoth) {
                  return (
                    <>
                      <rect
                        x={swNode.x - (size.w / 2 + padding + 4)}
                        y={swNode.y - (size.h / 2 + padding + 4)}
                        width={size.w + (padding + 4) * 2}
                        height={size.h + (padding + 4) * 2}
                        rx={8}
                        ry={8}
                        fill="none"
                        stroke="#10b981"
                        strokeWidth={2.5}
                        strokeDasharray="6 3"
                        style={{ filter: 'drop-shadow(0 0 8px #10b981) drop-shadow(0 0 16px rgba(16, 185, 129, 0.5))' }}
                      />
                      <rect
                        x={swNode.x - (size.w / 2 + padding)}
                        y={swNode.y - (size.h / 2 + padding)}
                        width={size.w + padding * 2}
                        height={size.h + padding * 2}
                        rx={6}
                        ry={6}
                        fill="none"
                        stroke="#2563eb"
                        strokeWidth={2.5}
                        strokeDasharray="6 3"
                        style={{ filter: 'drop-shadow(0 0 8px #2563eb) drop-shadow(0 0 16px rgba(37, 99, 235, 0.5))' }}
                      />
                    </>
                  )
                }

                const color = isSourceSelected ? '#2563eb' : '#10b981'
                const glowColor = isSourceSelected ? 'rgba(37, 99, 235, 0.5)' : 'rgba(16, 185, 129, 0.5)'
                return (
                  <rect
                    x={swNode.x - (size.w / 2 + padding)}
                    y={swNode.y - (size.h / 2 + padding)}
                    width={size.w + padding * 2}
                    height={size.h + padding * 2}
                    rx={6}
                    ry={6}
                    fill="none"
                    stroke={color}
                    strokeWidth={2.5}
                    strokeDasharray="6 3"
                    style={{ filter: `drop-shadow(0 0 8px ${color}) drop-shadow(0 0 16px ${glowColor})` }}
                  />
                )
              })()}
              {renderNode(swNode, {
                keyPrefix: 'upper-sw',
                scale: 1,
                isSelected: selectedNodeId === swNode.id,
                useMultiLevelConfig: true,
                onClick: () => {
                  if (connectionMode === 'view' && onNodeClick) {
                    const swConnections = edges
                      .filter(edge => edge.source === swNode.id || edge.target === swNode.id)
                      .map(edge => {
                        const otherId = edge.source === swNode.id ? edge.target : edge.source
                        const otherNode = displayNodes.find(n => n.id === otherId)
                        return { id: otherId, label: otherNode?.label || otherId, bandwidth: edge.bandwidth, latency: edge.latency }
                      })
                    onNodeClick({
                      id: swNode.id,
                      label: swNode.label,
                      type: swNode.type,
                      subType: swNode.subType,
                      connections: swConnections,
                    })
                  }
                }
              })}
            </g>
          )
        })}
      </g>
    )
  }

  // 渲染跨层级边（带深度效果）
  const renderInterLevelEdges = () => {
    const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex
    return interLevelEdges.map((edge, idx) => {
      const sourcePos = getNodePosition(edge.source, activeLayerIdx)
      const targetPos = getNodePosition(edge.target, activeLayerIdx)
      if (!sourcePos || !targetPos) return null

      // 跨层级边使用 inter: 前缀
      const edgeId = `inter:${edge.source}-${edge.target}`
      const reverseEdgeId = `inter:${edge.target}-${edge.source}`
      const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === reverseEdgeId

      // 计算层级差异，用于深度效果
      const zDiff = Math.abs((sourcePos.zLayer ?? 0) - (targetPos.zLayer ?? 0))

      // 动态样式：根据层级差异调整
      const baseStrokeWidth = isLinkSelected ? 2.5 : 2
      const strokeWidth = Math.max(1, baseStrokeWidth - zDiff * 0.2)
      const strokeOpacity = isLinkSelected ? 0.9 : Math.max(0.5, 0.8 - zDiff * 0.1)
      const shadowBlur = 4 + zDiff * 2
      const shadowOpacity = 0.15 + zDiff * 0.05

      const srcX = sourcePos.x
      const srcY = sourcePos.y
      const tgtX = targetPos.x
      const tgtY = targetPos.y

      const isDownward = tgtY > srcY
      const verticalDistance = Math.abs(tgtY - srcY)
      const horizontalDistance = tgtX - srcX

      const spreadRatio = 0.8
      const ctrl1Offset = horizontalDistance * spreadRatio
      const ctrl1X = srcX + ctrl1Offset
      const ctrl2X = tgtX

      const ctrl1Y = isDownward
        ? srcY + verticalDistance * 0.1
        : srcY - verticalDistance * 0.1
      const ctrl2Y = isDownward
        ? srcY + verticalDistance * 0.90
        : srcY - verticalDistance * 0.90

      const pathD = `M ${srcX} ${srcY} C ${ctrl1X} ${ctrl1Y}, ${ctrl2X} ${ctrl2Y}, ${tgtX} ${tgtY}`

      const getDetailedLabel = (nodeId: string) => {
        const parts = nodeId.split('/')
        return parts.length >= 2 ? parts.slice(-2).join('/') : nodeId
      }

      const getNodeType = (nodeId: string): string => {
        const upperSwitch = upperSwitchNodes.find(n => n.id === nodeId)
        if (upperSwitch) return 'switch'
        for (const container of containers) {
          const slNode = container.singleLevelData?.nodes.find(n => n.id === nodeId)
          if (slNode) return slNode.isSwitch ? 'switch' : slNode.type
        }
        return 'default'
      }

      // 选中时使用蓝色，未选中时使用渐变色增加深度感
      const strokeColor = isLinkSelected ? '#2563eb' : '#1890ff'
      const glowColor = isLinkSelected ? 'rgba(37, 99, 235, 0.4)' : `rgba(24, 144, 255, ${shadowOpacity})`

      return (
        <g
          key={`inter-level-${idx}`}
          style={{
            filter: `drop-shadow(0 ${2 + zDiff}px ${shadowBlur}px ${glowColor})`,
          }}
        >
          <path
            d={pathD}
            fill="none"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeOpacity={strokeOpacity}
            strokeLinecap="round"
            style={{ cursor: 'pointer', transition: 'stroke-opacity 0.2s ease, stroke-width 0.2s ease' }}
            onClick={(e) => {
              e.stopPropagation()
              if (connectionMode !== 'view') return
              onLinkClick?.({
                id: edgeId,
                sourceId: edge.source,
                sourceLabel: getDetailedLabel(edge.source),
                sourceType: getNodeType(edge.source),
                targetId: edge.target,
                targetLabel: getDetailedLabel(edge.target),
                targetType: getNodeType(edge.target),
                bandwidth: edge.bandwidth,
                latency: edge.latency,
                isManual: false
              })
            }}
          />
        </g>
      )
    })
  }

  // 渲染选中高亮层（选中的边复制到最上层）
  const renderHighlightLayer = () => {
    if (!selectedLinkId) return null

    const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex

    // 检查是否是跨层级边（使用 inter: 前缀）
    const interLevelEdge = interLevelEdges.find(edge => {
      const edgeId = `inter:${edge.source}-${edge.target}`
      return selectedLinkId === edgeId || selectedLinkId === `inter:${edge.target}-${edge.source}`
    })

    if (interLevelEdge) {
      const sourcePos = getNodePosition(interLevelEdge.source, activeLayerIdx)
      const targetPos = getNodePosition(interLevelEdge.target, activeLayerIdx)
      if (!sourcePos || !targetPos) return null

      const srcX = sourcePos.x
      const srcY = sourcePos.y
      const tgtX = targetPos.x
      const tgtY = targetPos.y

      const isDownward = tgtY > srcY
      const verticalDistance = Math.abs(tgtY - srcY)
      const horizontalDistance = tgtX - srcX

      const spreadRatio = 0.8
      const ctrl1Offset = horizontalDistance * spreadRatio
      const ctrl1X = srcX + ctrl1Offset
      const ctrl2X = tgtX

      const ctrl1Y = isDownward
        ? srcY + verticalDistance * 0.1
        : srcY - verticalDistance * 0.1
      const ctrl2Y = isDownward
        ? srcY + verticalDistance * 0.90
        : srcY - verticalDistance * 0.90

      const pathD = `M ${srcX} ${srcY} C ${ctrl1X} ${ctrl1Y}, ${ctrl2X} ${ctrl2Y}, ${tgtX} ${tgtY}`

      return (
        <g
          className="highlight-layer"
          style={{ filter: 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.5))' }}
        >
          <path
            d={pathD}
            fill="none"
            stroke="#2563eb"
            strokeWidth={3}
            strokeOpacity={0.9}
            strokeLinecap="round"
            style={{ pointerEvents: 'none' }}
          />
        </g>
      )
    }

    // 手动连接的高亮已由ManualConnectionLine组件自己处理，无需额外渲染

    return null
  }

  // 为特定容器渲染手动连接
  const renderManualConnectionsForContainer = (containerId: string) => {
    const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex
    const container = containers.find(c => c.id === containerId)
    if (!container) return null

    // 获取与这个容器相关的手动连接（只渲染源节点在这个容器内的连接，避免重复渲染）
    const containerConnections = currentManualConnections.filter(conn => {
      const sourceInContainer = container.singleLevelData?.nodes.some(n => n.id === conn.source)
      return sourceInContainer
    })

    const getParentIdx = (nodeId: string): number => {
      const parts = nodeId.split('/')
      if (parts.length >= 2) {
        const parentPart = parts[parts.length - 2]
        const match = parentPart.match(/_(\d+)$/)
        return match ? parseInt(match[1], 10) : 0
      }
      return 0
    }

    const isNodeInContainer = (nodeId: string, cId: string): boolean => {
      const c = containers.find(cont => cont.id === cId)
      if (!c || !c.singleLevelData) return false
      return c.singleLevelData.nodes.some((n: any) => n.id === nodeId)
    }

    const selectedContainerId = containers.find(c => c.id === selectedNodeId)?.id || null
    const hoveredContainerId = hoveredLayerIndex !== null
      ? containers.find(c => c.zLayer === hoveredLayerIndex)?.id || null
      : null
    const defaultContainerId = containers.length > 0
      ? containers.reduce((min, c) => (c.zLayer ?? Infinity) < (min.zLayer ?? Infinity) ? c : min).id
      : null
    const activeContainerId = hoveredContainerId || selectedContainerId || defaultContainerId

    const containerBoundsInfo = containers.map(c => ({
      zLayer: c.zLayer ?? 0,
      bounds: c.containerBounds!,
    }))

    const getNodeType = (nodeId: string): string => {
      const parts = nodeId.split('/')
      const lastPart = parts[parts.length - 1] || ''
      return lastPart.split('_')[0] || 'default'
    }

    return containerConnections.map((conn) => {
      const sourcePos = getNodePosition(conn.source, activeLayerIdx)
      const targetPos = getNodePosition(conn.target, activeLayerIdx)

      const sourceParentIdx = getParentIdx(conn.source)
      const targetParentIdx = getParentIdx(conn.target)
      const indexDiff = Math.abs(sourceParentIdx - targetParentIdx)
      const isCrossContainer = true

      const manualEdgeId = `${conn.source}-${conn.target}`
      const isLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

      let linkOpacity = 1
      if (activeContainerId && !isLinkSelected) {
        const sourceInActive = isNodeInContainer(conn.source, activeContainerId)
        const targetInActive = isNodeInContainer(conn.target, activeContainerId)
        if (!sourceInActive && !targetInActive) {
          linkOpacity = 0.2
        }
      }

      const handleManualClick = (e: React.MouseEvent) => {
        e.stopPropagation()
        if (connectionMode !== 'view') return
        const getDetailedLabel = (nodeId: string) => {
          const parts = nodeId.split('/')
          return parts.length >= 2 ? parts.slice(-2).join('/') : nodeId
        }
        onLinkClick?.({
          id: manualEdgeId,
          sourceId: conn.source,
          sourceLabel: getDetailedLabel(conn.source),
          sourceType: conn.source.split('/').pop()?.split('_')[0] || 'unknown',
          targetId: conn.target,
          targetLabel: getDetailedLabel(conn.target),
          targetType: conn.target.split('/').pop()?.split('_')[0] || 'unknown',
          isManual: true
        })
      }

      return (
        <g key={`manual-conn-${conn.id}-${containerId}`} style={{ opacity: linkOpacity, transition: 'opacity 0.2s ease' }}>
          <ManualConnectionLine
            conn={conn}
            sourcePos={sourcePos}
            targetPos={targetPos}
            isSelected={isLinkSelected}
            isCrossContainer={isCrossContainer}
            indexDiff={indexDiff}
            onClick={handleManualClick}
            layoutType={layoutType}
            containers={containerBoundsInfo}
            sourceType={getNodeType(conn.source)}
            targetType={getNodeType(conn.target)}
            isMultiLevel={true}
          />
        </g>
      )
    })
  }

  // 获取目标在指定容器内的upperSwitch手动连接
  const upperSwitchIds = new Set(upperSwitchNodes.map(n => n.id))
  const getUpperSwitchConnectionsForContainer = (containerId: string) => {
    const container = containers.find(c => c.id === containerId)
    if (!container?.singleLevelData) return []
    const containerNodeIds = new Set(container.singleLevelData.nodes.map((n: any) => n.id))
    return currentManualConnections.filter(conn => {
      const sourceIsUpperSwitch = upperSwitchIds.has(conn.source)
      const targetInContainer = containerNodeIds.has(conn.target)
      const targetIsUpperSwitch = upperSwitchIds.has(conn.target)
      const sourceInContainer = containerNodeIds.has(conn.source)
      return (sourceIsUpperSwitch && targetInContainer) || (targetIsUpperSwitch && sourceInContainer)
    })
  }

  // 渲染指定容器的upperSwitch手动连接
  const renderUpperSwitchConnectionsForContainer = (containerId: string) => {
    const connections = getUpperSwitchConnectionsForContainer(containerId)
    if (connections.length === 0) return null

    const activeLayerIdx = hoveredLayerIndex ?? selectedLayerIndex

    const getParentIdx = (nodeId: string): number => {
      const parts = nodeId.split('/')
      if (parts.length >= 2) {
        const parentPart = parts[parts.length - 2]
        const match = parentPart.match(/_(\d+)$/)
        return match ? parseInt(match[1], 10) : 0
      }
      return 0
    }

    const containerBoundsInfo = containers.map(c => ({
      zLayer: c.zLayer ?? 0,
      bounds: c.containerBounds!,
    }))

    const getNodeType = (nodeId: string): string => {
      const parts = nodeId.split('/')
      const lastPart = parts[parts.length - 1] || ''
      return lastPart.split('_')[0] || 'default'
    }

    return connections.map((conn) => {
      const sourcePos = getNodePosition(conn.source, activeLayerIdx)
      const targetPos = getNodePosition(conn.target, activeLayerIdx)

      const sourceParentIdx = getParentIdx(conn.source)
      const targetParentIdx = getParentIdx(conn.target)
      const indexDiff = Math.abs(sourceParentIdx - targetParentIdx)

      const manualEdgeId = `${conn.source}-${conn.target}`
      const isLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

      const handleManualClick = (e: React.MouseEvent) => {
        e.stopPropagation()
        if (connectionMode !== 'view') return
        const getDetailedLabel = (nodeId: string) => {
          const parts = nodeId.split('/')
          return parts.length >= 2 ? parts.slice(-2).join('/') : nodeId
        }
        onLinkClick?.({
          id: manualEdgeId,
          sourceId: conn.source,
          sourceLabel: getDetailedLabel(conn.source),
          sourceType: conn.source.split('/').pop()?.split('_')[0] || 'unknown',
          targetId: conn.target,
          targetLabel: getDetailedLabel(conn.target),
          targetType: conn.target.split('/').pop()?.split('_')[0] || 'unknown',
          isManual: true
        })
      }

      return (
        <g key={`manual-conn-upper-${conn.id}`}>
          <ManualConnectionLine
            conn={conn}
            sourcePos={sourcePos}
            targetPos={targetPos}
            isSelected={isLinkSelected}
            isCrossContainer={true}
            indexDiff={indexDiff}
            onClick={handleManualClick}
            layoutType={layoutType}
            containers={containerBoundsInfo}
            sourceType={getNodeType(conn.source)}
            targetType={getNodeType(conn.target)}
            isMultiLevel={true}
          />
        </g>
      )
    })
  }

  // 按zLayer交错渲染容器（保持容器之间的遮挡关系）
  const containersInterleavedRender = containers.map((containerNode, idx) => {
    const zLayer = containerNode.zLayer ?? 0
    const isExpanding = expandingContainer?.id === containerNode.id

    // 获取与这个容器相关的手动连接
    const containerManualConnections = currentManualConnections.filter(conn => {
      const sourceInContainer = containerNode.singleLevelData?.nodes.some(n => n.id === conn.source)
      return sourceInContainer // 只渲染源节点在这个容器内的连接
    })

    return (
      <g
        key={`container-group-${containerNode.id}`}
        onMouseEnter={() => !expandingContainer && !collapsingContainer && setHoveredLayerIndex(zLayer)}
        onMouseLeave={() => {
          if (connectionMode !== 'view') return
          if (!expandingContainer && !collapsingContainer) {
            setHoveredLayerIndex(null)
          }
        }}
        onTransitionEnd={(e) => {
          if (isExpanding && e.propertyName === 'transform' && onExpandAnimationEnd) {
            onExpandAnimationEnd(containerNode.id, containerNode.type)
          }
        }}
      >
        {/* 容器背景 */}
        {containerBackgrounds[idx]}
        {/* 容器内边 */}
        {containerEdgesLayer[idx]}
        {/* 与这个容器相关的手动连接（在节点下面）*/}
        {containerManualConnections.length > 0 && renderManualConnectionsForContainer(containerNode.id)}
        {/* upperSwitch到这个容器的手动连接（在节点下面，被上层容器遮挡）*/}
        {renderUpperSwitchConnectionsForContainer(containerNode.id)}
        {/* 容器内节点 */}
        {containerNodesLayer[idx]}
      </g>
    )
  })

  return (
    <>
      {/* 按zLayer交错渲染的容器（背景+边+手动连接+upperSwitch连接+节点）*/}
      {containersInterleavedRender}
      {/* 跨层级边（容器外Switch到容器内节点）*/}
      {renderInterLevelEdges()}
      {/* 上层Switch面板 */}
      {renderUpperSwitchPanel()}
      {/* 选中高亮层（确保选中的边始终可见）*/}
      {renderHighlightLayer()}
    </>
  )
}
