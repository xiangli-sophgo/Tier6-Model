import React from 'react'
import { Node, Edge, LayoutType, ManualConnection, getNodeEdgePoint } from './shared'
import { ManualConnectionLine, renderIndirectEdge, EdgeRendererProps } from './components'
import { getTorusGridSize, getTorus3DSize } from './layouts'
import { HierarchyLevel } from '../../types'

// 单层级视图渲染所需的props
export interface SingleLevelViewProps {
  // 数据
  displayNodes: Node[]
  nodes: Node[]
  edges: Edge[]
  manualConnections: ManualConnection[]
  nodePositions: Map<string, { x: number; y: number }>
  // 状态
  zoom: number
  selectedNodeId: string | null
  selectedLinkId: string | null
  hoveredNodeId: string | null
  setHoveredNodeId: (id: string | null) => void
  // 模式
  connectionMode: 'view' | 'select' | 'connect' | 'select_source' | 'select_target'
  isManualMode: boolean
  isForceMode: boolean
  directTopology: string
  switchPanelWidth: number
  // 回调
  onNodeClick?: (node: any) => void
  onNodeDoubleClick?: (nodeId: string, nodeType: string) => void
  onLinkClick?: (link: any) => void
  setTooltip: (tooltip: { x: number; y: number; content: string } | null) => void
  svgRef: React.RefObject<SVGSVGElement>
  // 拖拽
  draggingNode: string | null
  handleDragStart: (nodeId: string, e: React.MouseEvent) => void
  // 其他
  layoutType: LayoutType
  nodeScale: number
  renderNode: (node: Node, options: { keyPrefix: string; scale?: number; isSelected?: boolean; onClick?: () => void }) => JSX.Element
  renderNodeShape: (node: Node) => JSX.Element
  getCurrentHierarchyLevel: () => HierarchyLevel
  getTrafficHeatmapStyle: (source: string, target: string) => { stroke: string; strokeWidth: number; utilization: number; trafficMb: number } | null
  // 选中节点
  selectedNodes: Set<string>
  targetNodes: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  onTargetNodesChange?: (nodes: Set<string>) => void
}

export const SingleLevelView: React.FC<SingleLevelViewProps> = ({
  displayNodes,
  nodes,
  edges,
  manualConnections,
  nodePositions,
  zoom,
  selectedNodeId,
  selectedLinkId,
  hoveredNodeId,
  setHoveredNodeId,
  connectionMode,
  isManualMode,
  isForceMode,
  directTopology,
  switchPanelWidth,
  onNodeClick,
  onNodeDoubleClick,
  onLinkClick,
  setTooltip,
  svgRef,
  draggingNode,
  handleDragStart,
  layoutType,
  nodeScale,
  renderNode,
  renderNodeShape,
  getCurrentHierarchyLevel,
  getTrafficHeatmapStyle,
  selectedNodes,
  targetNodes,
  onSelectedNodesChange,
  onTargetNodesChange,
}) => {
  // 创建边渲染器的props
  const edgeProps: EdgeRendererProps = {
    edges,
    nodes,
    nodePositions,
    zoom,
    selectedLinkId,
    connectionMode,
    isManualMode,
    onLinkClick,
    setTooltip,
    svgRef,
    getTrafficHeatmapStyle,
    directTopology,
    nodeScale,
  }

  // 渲染普通边
  const renderEdges = () => {
    return edges.map((edge, i) => {
      // 外部连接 - 单层级视图中不显示跨容器连接
      if (edge.isExternal) {
        return null
      }

      // 间接连接
      if (edge.isIndirect) {
        return renderIndirectEdge(edge, i, edgeProps)
      }

      const sourcePos = nodePositions.get(edge.source)
      const targetPos = nodePositions.get(edge.target)
      if (!sourcePos || !targetPos) return null

      const sourceNode = nodes.find(n => n.id === edge.source)
      const targetNode = nodes.find(n => n.id === edge.target)

      const edgeId = `${edge.source}-${edge.target}`
      const isLinkSelected = selectedLinkId === edgeId || selectedLinkId === `${edge.target}-${edge.source}`

      const bandwidthStr = edge.bandwidth ? `${edge.bandwidth} GB/s` : ''
      const latencyStr = edge.latency ? `${edge.latency} us` : ''
      const trafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
      const trafficStr = trafficStyle ? `流量: ${trafficStyle.trafficMb.toFixed(1)}MB, 利用率: ${(trafficStyle.utilization * 100).toFixed(0)}%` : ''
      const propsStr = [bandwidthStr, latencyStr, trafficStr].filter(Boolean).join(', ')
      const tooltipContent = `${sourceNode?.label || edge.source} ↔ ${targetNode?.label || edge.target}${propsStr ? ` (${propsStr})` : ''}`

      const handleLinkClick = (e: React.MouseEvent) => {
        e.stopPropagation()
        if (connectionMode !== 'view' || isManualMode) return
        if (onLinkClick) {
          onLinkClick({
            id: edgeId,
            sourceId: edge.source,
            sourceLabel: sourceNode?.label || edge.source,
            sourceType: sourceNode?.type || 'unknown',
            targetId: edge.target,
            targetLabel: targetNode?.label || edge.target,
            targetType: targetNode?.type || 'unknown',
            bandwidth: edge.bandwidth,
            latency: edge.latency,
            isManual: false
          })
        }
      }

      // Torus环绕连接检测
      const deviceNodes = nodes.filter(n => !n.isSwitch)

      if (directTopology === 'torus_2d') {
        const { cols, rows } = getTorusGridSize(deviceNodes.length)
        const isHorizontalWrap = sourceNode?.gridRow === targetNode?.gridRow &&
          Math.abs((sourceNode?.gridCol || 0) - (targetNode?.gridCol || 0)) === cols - 1
        const isVerticalWrap = sourceNode?.gridCol === targetNode?.gridCol &&
          Math.abs((sourceNode?.gridRow || 0) - (targetNode?.gridRow || 0)) === rows - 1
        if (isHorizontalWrap || isVerticalWrap) {
          return null
        }
      }

      if (directTopology === 'torus_3d') {
        void getTorus3DSize(deviceNodes.length)
        const sameZ = sourceNode?.gridZ === targetNode?.gridZ
        const sameRow = sourceNode?.gridRow === targetNode?.gridRow
        const sameCol = sourceNode?.gridCol === targetNode?.gridCol

        const rowNodes = deviceNodes.filter(n => n.gridZ === sourceNode?.gridZ && n.gridRow === sourceNode?.gridRow)
        const isXWrap = sameZ && sameRow && rowNodes.length >= 3 &&
          Math.abs((sourceNode?.gridCol || 0) - (targetNode?.gridCol || 0)) === rowNodes.length - 1

        const colNodes = deviceNodes.filter(n => n.gridZ === sourceNode?.gridZ && n.gridCol === sourceNode?.gridCol)
        const isYWrap = sameZ && sameCol && colNodes.length >= 3 &&
          Math.abs((sourceNode?.gridRow || 0) - (targetNode?.gridRow || 0)) === colNodes.length - 1

        const depthNodes = deviceNodes.filter(n => n.gridRow === sourceNode?.gridRow && n.gridCol === sourceNode?.gridCol)
        const isZWrap = sameRow && sameCol && depthNodes.length >= 3 &&
          Math.abs((sourceNode?.gridZ || 0) - (targetNode?.gridZ || 0)) === depthNodes.length - 1

        if (isXWrap || isYWrap || isZWrap) {
          return null
        }
      }

      // 计算边缘点（检查isSwitch来确定正确的节点类型，使用nodeScale）
      const sourceType = sourceNode?.isSwitch ? 'switch' : (sourceNode?.type || 'default')
      const targetType = targetNode?.isSwitch ? 'switch' : (targetNode?.type || 'default')
      const sourceEdge = getNodeEdgePoint(sourcePos.x, sourcePos.y, targetPos.x, targetPos.y, sourceType, false, nodeScale)
      const targetEdge = getNodeEdgePoint(targetPos.x, targetPos.y, sourcePos.x, sourcePos.y, targetType, false, nodeScale)

      // 渲染普通直线边
      return (
        <g
          key={`edge-${i}`}
          style={{ cursor: connectionMode === 'view' && !isManualMode ? 'pointer' : 'default' }}
          onMouseEnter={(e) => {
            if (connectionMode === 'view' && !isManualMode && svgRef.current) {
              const rect = svgRef.current.getBoundingClientRect()
              setTooltip({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top - 10,
                content: tooltipContent
              })
            }
          }}
          onMouseLeave={() => setTooltip(null)}
          onClick={handleLinkClick}
        >
          {/* 透明触发层 */}
          <line
            x1={sourceEdge.x}
            y1={sourceEdge.y}
            x2={targetEdge.x}
            y2={targetEdge.y}
            stroke="transparent"
            strokeWidth={12}
            style={{ cursor: connectionMode === 'view' && !isManualMode ? 'pointer' : 'default' }}
          />
          {/* 可见线条 */}
          <line
            x1={sourceEdge.x}
            y1={sourceEdge.y}
            x2={targetEdge.x}
            y2={targetEdge.y}
            stroke={isLinkSelected ? '#2563eb' : (trafficStyle?.stroke || '#b0b0b0')}
            strokeWidth={isLinkSelected ? 3 : 1.5}
            strokeOpacity={isLinkSelected ? 1 : 0.6}
            style={{ pointerEvents: 'none', filter: isLinkSelected ? 'drop-shadow(0 0 4px #2563eb)' : 'none' }}
          />
        </g>
      )
    })
  }

  // 渲染手动连接
  const renderManualConnections = () => {
    const currentManualConnections = manualConnections.filter(mc => mc.hierarchy_level === getCurrentHierarchyLevel())

    return currentManualConnections.map((conn) => {
      const sourcePos = nodePositions.get(conn.source)
      const targetPos = nodePositions.get(conn.target)
      if (!sourcePos || !targetPos) return null

      const manualEdgeId = `${conn.source}-${conn.target}`
      const isLinkSelected = selectedLinkId === manualEdgeId || selectedLinkId === `${conn.target}-${conn.source}`

      const sourceNode = displayNodes.find(n => n.id === conn.source)
      const targetNode = displayNodes.find(n => n.id === conn.target)

      const handleManualClick = (e: React.MouseEvent) => {
        e.stopPropagation()
        if (connectionMode !== 'view') return
        onLinkClick?.({
          id: manualEdgeId,
          sourceId: conn.source,
          sourceLabel: sourceNode?.label || conn.source,
          sourceType: sourceNode?.type || 'unknown',
          targetId: conn.target,
          targetLabel: targetNode?.label || conn.target,
          targetType: targetNode?.type || 'unknown',
          isManual: true
        })
      }

      // 检查isSwitch来确定正确的节点类型
      const sourceNodeType = sourceNode?.isSwitch ? 'switch' : (sourceNode?.type || 'default')
      const targetNodeType = targetNode?.isSwitch ? 'switch' : (targetNode?.type || 'default')

      return (
        <ManualConnectionLine
          key={`manual-conn-${conn.id}`}
          conn={conn}
          sourcePos={{ ...sourcePos, zLayer: 0 }}
          targetPos={{ ...targetPos, zLayer: 0 }}
          isSelected={isLinkSelected}
          isCrossContainer={false}
          indexDiff={0}
          onClick={handleManualClick}
          layoutType={layoutType}
          containers={[]}
          sourceType={sourceNodeType}
          targetType={targetNodeType}
          isMultiLevel={false}
          nodeScale={nodeScale}
        />
      )
    })
  }

  // 渲染Switch面板
  const renderSwitchPanel = () => {
    const switchPanelNodes = displayNodes.filter(n => n.inSwitchPanel)
    if (switchPanelNodes.length === 0) return null

    const switchIds = new Set(switchPanelNodes.map(n => n.id))
    const switchInternalEdges = edges.filter(e =>
      switchIds.has(e.source) && switchIds.has(e.target)
    )

    return (
      <g className="switch-panel">
        {/* Switch之间的连线 */}
        {switchInternalEdges.map((edge, idx) => {
          const sourceNode = switchPanelNodes.find(n => n.id === edge.source)
          const targetNode = switchPanelNodes.find(n => n.id === edge.target)
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
              key={`switch-edge-${idx}`}
              d={pathD}
              fill="none"
              stroke="#1890ff"
              strokeWidth={2}
              strokeOpacity={0.6}
            />
          )
        })}

        {/* Switch节点 */}
        {switchPanelNodes.map(node => renderNode(node, {
          keyPrefix: 'switch',
          isSelected: selectedNodeId === node.id,
          onClick: () => onNodeClick?.({
            id: node.id,
            label: node.label,
            type: 'switch',
            subType: node.subType,
            connections: edges.filter(e => e.source === node.id || e.target === node.id)
              .map(e => ({
                id: e.source === node.id ? e.target : e.source,
                label: displayNodes.find(n => n.id === (e.source === node.id ? e.target : e.source))?.label || '',
                bandwidth: e.bandwidth,
                latency: e.latency,
              })),
            portInfo: node.portInfo,
          })
        }))}
      </g>
    )
  }

  // 渲染设备节点
  const renderDeviceNodes = () => {
    return displayNodes.map((node) => {
      if (node.isContainer || node.inSwitchPanel) {
        return null
      }

      const isSwitch = node.isSwitch
      const nodeConnections = edges.filter(e => e.source === node.id || e.target === node.id)
      const isSourceSelected = selectedNodes.has(node.id)
      const isTargetSelected = targetNodes.has(node.id)
      const isDragging = draggingNode === node.id
      const isNodeSelected = selectedNodeId === node.id
      const isLinkEndpoint = selectedLinkId && (
        selectedLinkId.startsWith(node.id + '-') ||
        selectedLinkId.endsWith('-' + node.id)
      )
      const isHovered = hoveredNodeId === node.id && connectionMode === 'view' && !isManualMode && !isDragging
      const shouldHighlight = isNodeSelected || isHovered || isLinkEndpoint

      return (
        <g
          key={node.id}
          transform={`translate(${node.x}, ${node.y}) scale(${nodeScale * (isDragging ? 1.08 : 1)})`}
          className={isForceMode ? 'force-node-hover' : ''}
          style={{
            cursor: isForceMode ? (isDragging ? 'grabbing' : 'grab') : isManualMode ? 'move' : connectionMode !== 'view' ? 'crosshair' : 'pointer',
            opacity: isDragging ? 0.85 : 1,
            filter: isDragging
              ? 'drop-shadow(0 8px 16px rgba(0,0,0,0.25))'
              : shouldHighlight
                ? 'drop-shadow(0 0 8px rgba(37, 99, 235, 0.6)) drop-shadow(0 0 16px rgba(37, 99, 235, 0.3))'
                : 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
            transition: isDragging
              ? 'none'
              : 'transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), filter 0.15s ease, opacity 0.15s ease',
          }}
          onMouseDown={(e) => handleDragStart(node.id, e)}
          onMouseEnter={() => setHoveredNodeId(node.id)}
          onMouseLeave={() => setHoveredNodeId(null)}
          onClick={() => {
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
            } else if (connectionMode === 'view' && !isManualMode) {
              onNodeClick?.({
                id: node.id,
                label: node.label,
                type: isSwitch ? 'switch' : node.type,
                subType: node.subType,
                connections: nodeConnections.map(e => ({
                  id: e.source === node.id ? e.target : e.source,
                  label: displayNodes.find(n => n.id === (e.source === node.id ? e.target : e.source))?.label || '',
                  bandwidth: e.bandwidth,
                  latency: e.latency,
                })),
                portInfo: node.portInfo,
              })
            }
          }}
          onDoubleClick={(e) => {
            e.stopPropagation()
            onNodeDoubleClick?.(node.id, isSwitch ? 'switch' : node.type)
          }}
        >
          {/* 选中指示器（连接模式）- 发光矩形效果 */}
          {(isSourceSelected || isTargetSelected) && (() => {
            // 根据节点类型获取尺寸
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
              // 同时是源节点和目标节点：显示双层边框（外层虚线）
              return (
                <>
                  {/* 外层：目标节点绿色虚线 */}
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
                  {/* 内层：源节点蓝色虚线 */}
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
          {renderNodeShape(node)}
        </g>
      )
    })
  }

  return (
    <>
      {/* 边 */}
      {renderEdges()}
      {/* 手动连接 */}
      {renderManualConnections()}
      {/* Switch面板 */}
      {switchPanelWidth > 0 && renderSwitchPanel()}
      {/* 设备节点 */}
      {renderDeviceNodes()}
    </>
  )
}
