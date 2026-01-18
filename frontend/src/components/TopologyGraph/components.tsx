/**
 * TopologyGraph 子组件集合
 * 包含: ManualConnectionLine, ControlPanel, EdgeRenderer, LevelPairSelector
 */
import React from 'react'
import { Segmented, Tooltip, Checkbox, Button, Typography } from 'antd'
import { UndoOutlined, RedoOutlined, ReloadOutlined } from '@ant-design/icons'
import { LayoutType, AdjacentLevelPair, LEVEL_PAIR_NAMES } from '../../types'
import { Node, Edge, LinkDetail, MultiLevelViewOptions, getNodeEdgePoint } from './shared'
import { getTorusGridSize, getTorus3DSize } from './layouts'

const { Text } = Typography

// ==========================================
// ManualConnectionLine - 手动连接线组件
// ==========================================

export interface AnimatedManualConnectionProps {
  conn: { id: string; source: string; target: string }
  sourcePos: { x: number; y: number; zLayer: number } | null
  targetPos: { x: number; y: number; zLayer: number } | null
  isSelected: boolean
  isCrossContainer: boolean
  indexDiff: number
  onClick: (e: React.MouseEvent) => void
  layoutType?: string
  containers?: Array<{
    zLayer: number
    bounds: { x: number; y: number; width: number; height: number }
  }>
  // 节点类型（用于边缘点计算）
  sourceType?: string
  targetType?: string
  isMultiLevel?: boolean
  nodeScale?: number
}

export const ManualConnectionLine: React.FC<AnimatedManualConnectionProps> = ({
  conn,
  sourcePos,
  targetPos,
  isSelected,
  isCrossContainer,
  onClick,
  sourceType: _sourceType = 'default',
  targetType: _targetType = 'default',
  isMultiLevel: _isMultiLevel = false,
  nodeScale: _nodeScale = 1,
}) => {
  if (!sourcePos || !targetPos) return null

  const strokeColor = isSelected ? '#52c41a' : (isCrossContainer ? '#722ed1' : '#b0b0b0')
  const strokeWidth = isSelected ? 3 : 2
  const transitionStyle = { transition: 'all 0.3s ease-out' }

  // 直接使用节点中心坐标
  const sourceEdge = { x: sourcePos.x, y: sourcePos.y }
  const targetEdge = { x: targetPos.x, y: targetPos.y }

  // 提取节点编号（如 pod_0/rack_0/board_0/chip_0 -> 0）
  const getNodeIndex = (nodeId: string): string => {
    const parts = nodeId.split('/')
    const lastPart = parts[parts.length - 1] || ''
    const match = lastPart.match(/_(\d+)$/)
    return match ? match[1] : ''
  }

  if (isCrossContainer) {
    // 汇聚树形：控制点偏移与水平距离成比例，使出发点曲率一致
    const isDownward = targetEdge.y > sourceEdge.y
    const verticalDistance = Math.abs(targetEdge.y - sourceEdge.y)
    const horizontalDistance = targetEdge.x - sourceEdge.x

    // 检查源节点和目标节点的编号是否相同
    const sourceIndex = getNodeIndex(conn.source)
    const targetIndex = getNodeIndex(conn.target)
    const isSameIndex = sourceIndex !== '' && sourceIndex === targetIndex

    // 控制点的水平偏移与水平距离成比例
    // 比例系数决定了曲线的"展开"速度
    const spreadRatio = 0.8
    // 只有编号相同时才增加最小偏移量
    const minOffset = isSameIndex ? 15 : 0
    let ctrl1Offset = horizontalDistance * spreadRatio
    // 如果偏移量太小且编号相同，使用最小偏移
    if (isSameIndex && Math.abs(ctrl1Offset) < minOffset) {
      ctrl1Offset = horizontalDistance >= 0 ? minOffset : -minOffset
    }
    const ctrl1X = sourceEdge.x + ctrl1Offset
    const ctrl2X = isSameIndex ? sourceEdge.x : targetEdge.x

    // 控制点的垂直位置：第一个控制点更靠近起点，让曲线一开始就有角度
    const ctrl1Y = isDownward
      ? sourceEdge.y + verticalDistance * 0.1
      : sourceEdge.y - verticalDistance * 0.1
    const ctrl2Y = isDownward
      ? sourceEdge.y + verticalDistance * 0.90
      : sourceEdge.y - verticalDistance * 0.90

    // 三次贝塞尔曲线
    const pathD = `M ${sourceEdge.x} ${sourceEdge.y} C ${ctrl1X} ${ctrl1Y}, ${ctrl2X} ${ctrl2Y}, ${targetEdge.x} ${targetEdge.y}`

    return (
      <g style={transitionStyle}>
        {/* 点击区域 */}
        <path d={pathD} fill="none" stroke="transparent" strokeWidth={16}
          style={{ cursor: 'pointer' }} onClick={onClick} />
        {/* 可见线条 */}
        <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={strokeWidth}
          strokeOpacity={isSelected ? 1 : 0.6}
          style={{ pointerEvents: 'none', filter: isSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none' }}
        />
      </g>
    )
  } else {
    return (
      <g>
        <line
          x1={sourceEdge.x} y1={sourceEdge.y}
          x2={targetEdge.x} y2={targetEdge.y}
          stroke="transparent" strokeWidth={16}
          style={{ cursor: 'pointer', ...transitionStyle }} onClick={onClick}
        />
        <line
          x1={sourceEdge.x} y1={sourceEdge.y}
          x2={targetEdge.x} y2={targetEdge.y}
          stroke={strokeColor} strokeWidth={strokeWidth}
          strokeOpacity={isSelected ? 1 : 0.6}
          style={{ pointerEvents: 'none', filter: isSelected ? 'drop-shadow(0 0 4px #52c41a)' : 'none', ...transitionStyle }}
        />
      </g>
    )
  }
}

// ==========================================
// ControlPanel - 控制面板组件
// ==========================================

type ViewLevel = 'datacenter' | 'pod' | 'rack' | 'board' | 'chip'

export interface ControlPanelProps {
  multiLevelOptions?: MultiLevelViewOptions
  onMultiLevelOptionsChange?: (options: MultiLevelViewOptions) => void
  currentLevel: ViewLevel
  layoutType: LayoutType
  onLayoutTypeChange?: (type: LayoutType) => void
  isForceMode: boolean
  isForceSimulating: boolean
  isManualMode: boolean
  setIsManualMode: (value: boolean) => void
  manualPositions: Record<string, { x: number; y: number }>
  historyIndex: number
  historyLength: number
  onUndo: () => void
  onRedo: () => void
  onReset: () => void
  onLayoutChange?: () => void
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  multiLevelOptions,
  onMultiLevelOptionsChange,
  currentLevel,
  layoutType,
  onLayoutTypeChange,
  isForceMode,
  isForceSimulating,
  isManualMode,
  setIsManualMode,
  manualPositions,
  historyIndex,
  historyLength,
  onUndo,
  onRedo,
  onReset,
  onLayoutChange,
}) => {
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      right: 16,
      zIndex: 100,
      background: '#fff',
      padding: '10px 14px',
      borderRadius: 10,
      border: '1px solid rgba(0, 0, 0, 0.08)',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.06)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <Segmented
          size="small"
          className="topology-layout-segmented"
          value={multiLevelOptions?.enabled ? 'multi' : 'single'}
          onChange={(value) => {
            if (onMultiLevelOptionsChange) {
              if (value === 'multi') {
                // 多层级视图显示"上一层 + 这一层"
                let levelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = multiLevelOptions?.levelPair || 'datacenter_pod'
                if (currentLevel === 'datacenter') {
                  levelPair = 'datacenter_pod'  // datacenter 没有上层，显示 Pod
                } else if (currentLevel === 'pod') {
                  levelPair = 'datacenter_pod'  // 显示 Pod（上层容器）+ Rack（这一层内容）
                } else if (currentLevel === 'rack') {
                  levelPair = 'pod_rack'  // 显示 Rack（上层容器）+ Board（这一层内容）
                } else if (currentLevel === 'board') {
                  levelPair = 'rack_board'  // 显示 Board（上层容器）+ Chip（这一层内容）
                }
                onMultiLevelOptionsChange({
                  ...multiLevelOptions!,
                  enabled: true,
                  levelPair,
                })
              } else {
                onMultiLevelOptionsChange({
                  ...multiLevelOptions!,
                  enabled: false,
                })
              }
            }
          }}
          options={[
            { label: '单层级', value: 'single' },
            { label: '多层级', value: 'multi' },
          ]}
        />
        <div style={{ borderLeft: '1px solid rgba(0, 0, 0, 0.08)', height: 20 }} />
        <Segmented
          size="small"
          className="topology-layout-segmented"
          value={layoutType}
          onChange={(value) => {
            onLayoutTypeChange?.(value as LayoutType)
            onLayoutChange?.()
          }}
          options={[
            { label: '自动', value: 'auto' },
            { label: '环形', value: 'circle' },
            { label: '网格', value: 'grid' },
            { label: '力导向', value: 'force' },
          ]}
        />
        {isForceMode && (
          <Tooltip title={isForceSimulating ? '物理模拟进行中，可直接拖拽节点' : '物理模拟已稳定'}>
            <span style={{
              fontSize: 11,
              color: isForceSimulating ? '#52c41a' : '#8c8c8c',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                backgroundColor: isForceSimulating ? '#52c41a' : '#d9d9d9',
                animation: isForceSimulating ? 'pulse 1s infinite' : 'none',
              }} />
              {isForceSimulating ? '模拟中' : '已稳定'}
            </span>
          </Tooltip>
        )}
        <div style={{ borderLeft: '1px solid rgba(0, 0, 0, 0.08)', height: 20 }} />
        <Checkbox
          checked={isManualMode}
          onChange={(e) => setIsManualMode(e.target.checked)}
          disabled={multiLevelOptions?.enabled || isForceMode}
        >
          <span style={{ fontSize: 12 }}>手动调整</span>
        </Checkbox>
        {isManualMode && (
          <>
            <Tooltip title="撤销 (Ctrl+Z)">
              <Button
                type="text"
                size="small"
                icon={<UndoOutlined />}
                onClick={onUndo}
                disabled={historyIndex < 0}
              />
            </Tooltip>
            <Tooltip title="重做 (Ctrl+Y)">
              <Button
                type="text"
                size="small"
                icon={<RedoOutlined />}
                onClick={onRedo}
                disabled={historyIndex >= historyLength - 1}
              />
            </Tooltip>
            {Object.keys(manualPositions).length > 0 && (
              <Tooltip title="重置布局">
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={onReset}
                />
              </Tooltip>
            )}
          </>
        )}
      </div>
      {isManualMode && (
        <div style={{
          marginTop: 10,
          padding: '8px 12px',
          background: 'rgba(37, 99, 235, 0.06)',
          borderRadius: 8,
          border: '1px solid rgba(37, 99, 235, 0.12)',
          fontSize: 12,
          color: '#2563eb',
          fontWeight: 500,
        }}>
          Shift+拖动 · 自动吸附对齐 · 自动保存
        </div>
      )}
    </div>
  )
}

// ==========================================
// EdgeRenderer - 边渲染组件
// ==========================================

export interface EdgeRendererProps {
  edges: Edge[]
  nodes: Node[]
  nodePositions: Map<string, { x: number; y: number }>
  zoom: number
  selectedLinkId: string | null
  connectionMode: 'view' | 'select' | 'connect' | 'select_source' | 'select_target'
  isManualMode: boolean
  onLinkClick?: (link: LinkDetail | null) => void
  setTooltip: (tooltip: { x: number; y: number; content: string } | null) => void
  svgRef: React.RefObject<SVGSVGElement>
  getTrafficHeatmapStyle: (source: string, target: string) => { stroke: string; strokeWidth: number; utilization: number; trafficMb: number } | null
  directTopology?: string
  nodeScale?: number  // 节点缩放比例
}

export const renderExternalEdge = (
  edge: Edge,
  index: number,
  props: EdgeRendererProps
): React.ReactNode => {
  const { nodes, nodePositions, zoom, selectedLinkId, connectionMode, onLinkClick, nodeScale = 1 } = props

  let sourcePos = nodePositions.get(edge.source)
  if (!sourcePos) {
    const viewBoxWidth = 800 / zoom
    const viewBoxHeight = 600 / zoom
    sourcePos = { x: viewBoxWidth / 2, y: viewBoxHeight / 2 }
  }
  const sourceNode = nodes.find(n => n.id === edge.source)

  const viewBoxHeight = 600 / zoom
  const anchorX = sourcePos.x
  const anchorY = edge.externalDirection === 'upper' ? -20 : viewBoxHeight + 20

  const midX = (sourcePos.x + anchorX) / 2
  const midY = (sourcePos.y + anchorY) / 2
  const bulgeDir = edge.externalDirection === 'upper' ? -1 : 1
  const bulge = Math.abs(sourcePos.y - anchorY) * 0.3
  const ctrlX = midX
  const ctrlY = midY + bulgeDir * bulge

  // 计算源节点边缘点（使用控制点方向和nodeScale）
  const sourceType = sourceNode?.isSwitch ? 'switch' : (sourceNode?.type || 'default')
  const sourceEdge = getNodeEdgePoint(sourcePos.x, sourcePos.y, ctrlX, ctrlY, sourceType, false, nodeScale)

  const pathD = `M ${sourceEdge.x} ${sourceEdge.y} Q ${ctrlX} ${ctrlY}, ${anchorX} ${anchorY}`
  const shadowPathD = `M ${sourceEdge.x + 2} ${sourceEdge.y + 3} Q ${ctrlX + 2} ${ctrlY + 3}, ${anchorX + 2} ${anchorY + 3}`

  const edgeId = `${edge.source}-external-${edge.externalNodeId}`
  const isLinkSelected = selectedLinkId === edgeId

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectionMode !== 'view') return
    const externalId = edge.externalNodeId || ''
    const externalParts = externalId.split('/')
    const lastPart = externalParts[externalParts.length - 1] || ''
    let externalType = 'unknown'
    if (lastPart.startsWith('board')) externalType = 'board'
    else if (lastPart.startsWith('rack')) externalType = 'rack'
    else if (lastPart.startsWith('pod')) externalType = 'pod'
    else if (lastPart.includes('switch')) externalType = 'switch'
    const dirArrow = edge.externalDirection === 'upper' ? '↑' : '↓'
    onLinkClick?.({
      id: edgeId,
      sourceId: edge.source,
      sourceLabel: sourceNode?.label || edge.source,
      sourceType: sourceNode?.type || 'unknown',
      targetId: externalId,
      targetLabel: `${dirArrow} ${edge.externalNodeLabel || '外部节点'}`,
      targetType: externalType,
      bandwidth: edge.bandwidth,
      latency: edge.latency,
      isManual: false
    })
  }

  return (
    <g key={`external-edge-${index}`} style={{ transition: 'all 0.3s ease-out' }}>
      <path d={shadowPathD} fill="none" stroke="#000" strokeWidth={2}
        strokeOpacity={0.1} style={{ pointerEvents: 'none' }} />
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={12}
        style={{ cursor: 'pointer' }} onClick={handleClick} />
      <path d={pathD} fill="none" stroke={isLinkSelected ? '#52c41a' : '#faad14'}
        strokeWidth={isLinkSelected ? 2.5 : 2}
        strokeOpacity={isLinkSelected ? 1 : 0.8}
        strokeDasharray="8,4,2,4"
        style={{ pointerEvents: 'none' }} />
      <g transform={`translate(${anchorX}, ${anchorY})`}>
        <circle r={6} fill={isLinkSelected ? '#52c41a' : '#faad14'} opacity={0.9} />
        <text y={edge.externalDirection === 'upper' ? -10 : 16} textAnchor="middle"
          fontSize={11} fill="#666" fontWeight={500}>
          {edge.externalNodeLabel || '上层'}
        </text>
      </g>
      {isLinkSelected && (
        <path d={pathD} fill="none" stroke="#52c41a" strokeWidth={4}
          strokeOpacity={0.2} style={{ pointerEvents: 'none', filter: 'blur(3px)' }} />
      )}
    </g>
  )
}

export const renderIndirectEdge = (
  edge: Edge,
  index: number,
  props: EdgeRendererProps
): React.ReactNode => {
  const { nodes, nodePositions, selectedLinkId, connectionMode, onLinkClick, nodeScale = 1 } = props

  const sourcePos = nodePositions.get(edge.source)
  const targetPos = nodePositions.get(edge.target)
  if (!sourcePos || !targetPos) return null

  const sourceNode = nodes.find(n => n.id === edge.source)
  const targetNode = nodes.find(n => n.id === edge.target)

  const dist = Math.sqrt(
    Math.pow(targetPos.x - sourcePos.x, 2) + Math.pow(targetPos.y - sourcePos.y, 2)
  )
  const midX = (sourcePos.x + targetPos.x) / 2
  const midY = (sourcePos.y + targetPos.y) / 2
  const bulge = Math.max(dist * 0.4, 60)
  const ctrlX = midX
  const ctrlY = midY - bulge

  // 计算边缘点（使用控制点方向和nodeScale）
  const sourceType = sourceNode?.isSwitch ? 'switch' : (sourceNode?.type || 'default')
  const targetType = targetNode?.isSwitch ? 'switch' : (targetNode?.type || 'default')
  const sourceEdge = getNodeEdgePoint(sourcePos.x, sourcePos.y, ctrlX, ctrlY, sourceType, false, nodeScale)
  const targetEdge = getNodeEdgePoint(targetPos.x, targetPos.y, ctrlX, ctrlY, targetType, false, nodeScale)

  const pathD = `M ${sourceEdge.x} ${sourceEdge.y} Q ${ctrlX} ${ctrlY}, ${targetEdge.x} ${targetEdge.y}`
  const shadowPathD = `M ${sourceEdge.x + 2} ${sourceEdge.y + 4} Q ${ctrlX + 2} ${ctrlY + 4}, ${targetEdge.x + 2} ${targetEdge.y + 4}`

  const edgeId = `${edge.source}-indirect-${edge.target}`
  const isLinkSelected = selectedLinkId === edgeId

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectionMode !== 'view') return
    onLinkClick?.({
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

  return (
    <g key={`indirect-edge-${index}`} style={{ transition: 'all 0.3s ease-out' }}>
      <path d={shadowPathD} fill="none" stroke="#000" strokeWidth={1.5}
        strokeOpacity={0.08} style={{ pointerEvents: 'none' }} />
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={12}
        style={{ cursor: 'pointer' }} onClick={handleClick} />
      <path d={pathD} fill="none" stroke={isLinkSelected ? '#52c41a' : '#722ed1'}
        strokeWidth={isLinkSelected ? 2 : 1.5}
        strokeOpacity={isLinkSelected ? 0.9 : 0.5}
        strokeDasharray="3,3"
        style={{ pointerEvents: 'none' }} />
      <g transform={`translate(${ctrlX}, ${ctrlY})`}>
        <circle r={5} fill={isLinkSelected ? '#52c41a' : '#722ed1'} opacity={0.7} />
        <text y={-8} textAnchor="middle" fontSize={11} fill="#666" fontWeight={500}>
          via {edge.viaNodeLabel || '上层'}
        </text>
      </g>
      {isLinkSelected && (
        <path d={pathD} fill="none" stroke="#52c41a" strokeWidth={3}
          strokeOpacity={0.2} style={{ pointerEvents: 'none', filter: 'blur(3px)' }} />
      )}
    </g>
  )
}

export const getEdgeStyle = (
  edge: Edge,
  isLinkSelected: boolean,
  getTrafficHeatmapStyle: EdgeRendererProps['getTrafficHeatmapStyle']
): { stroke: string; strokeWidth: number; strokeDasharray?: string } => {
  const trafficStyle = getTrafficHeatmapStyle(edge.source, edge.target)
  if (trafficStyle && !isLinkSelected) {
    return {
      stroke: trafficStyle.stroke,
      strokeWidth: trafficStyle.strokeWidth,
    }
  }
  if (!edge.connectionType) {
    return {
      stroke: isLinkSelected ? '#52c41a' : (edge.isSwitch ? '#1890ff' : '#b0b0b0'),
      strokeWidth: isLinkSelected ? 3 : (edge.isSwitch ? 2 : 1.5),
    }
  }
  switch (edge.connectionType) {
    case 'intra_upper':
      return { stroke: isLinkSelected ? '#52c41a' : '#1890ff', strokeWidth: isLinkSelected ? 3 : 2 }
    case 'intra_lower':
      return { stroke: isLinkSelected ? '#52c41a' : '#52c41a', strokeWidth: isLinkSelected ? 3 : 1.5 }
    case 'inter_level':
      return { stroke: isLinkSelected ? '#52c41a' : '#faad14', strokeWidth: isLinkSelected ? 3 : 1.5, strokeDasharray: '6,3' }
    default:
      return { stroke: '#b0b0b0', strokeWidth: 1.5 }
  }
}

// ==========================================
// LevelPairSelector - 层级选择组件
// ==========================================

interface LevelPairSelectorProps {
  value: AdjacentLevelPair | null
  onChange: (pair: AdjacentLevelPair | null) => void
  disabled?: boolean
  currentLevel: string
  hasCurrentPod: boolean
  hasCurrentRack: boolean
  hasCurrentBoard: boolean
}

function getAvailableOptions(
  currentLevel: string,
  hasCurrentPod: boolean,
  hasCurrentRack: boolean,
  hasCurrentBoard: boolean
): { label: string; value: string; disabled?: boolean }[] {
  const options: { label: string; value: string; disabled?: boolean }[] = [
    { label: '单层级', value: 'single' },
  ]

  if (currentLevel === 'datacenter') {
    options.push({ label: LEVEL_PAIR_NAMES.datacenter_pod, value: 'datacenter_pod' })
  }

  if (currentLevel === 'pod' || (currentLevel === 'datacenter' && hasCurrentPod)) {
    options.push({
      label: LEVEL_PAIR_NAMES.pod_rack,
      value: 'pod_rack',
      disabled: currentLevel === 'datacenter' && !hasCurrentPod,
    })
  }

  if (currentLevel === 'rack' || (currentLevel === 'pod' && hasCurrentRack)) {
    options.push({
      label: LEVEL_PAIR_NAMES.rack_board,
      value: 'rack_board',
      disabled: currentLevel === 'pod' && !hasCurrentRack,
    })
  }

  if (currentLevel === 'rack' && hasCurrentBoard) {
    options.push({
      label: LEVEL_PAIR_NAMES.board_chip,
      value: 'board_chip',
    })
  }

  if (currentLevel === 'board' && hasCurrentRack) {
    if (!options.some(o => o.value === 'rack_board')) {
      options.push({
        label: LEVEL_PAIR_NAMES.rack_board,
        value: 'rack_board',
      })
    }
  }

  return options
}

export const LevelPairSelector: React.FC<LevelPairSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  currentLevel,
  hasCurrentPod,
  hasCurrentRack,
  hasCurrentBoard,
}) => {
  const options = getAvailableOptions(currentLevel, hasCurrentPod, hasCurrentRack, hasCurrentBoard)

  const handleChange = (val: string | number) => {
    if (val === 'single') {
      onChange(null)
    } else {
      onChange(val as AdjacentLevelPair)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <Text style={{ fontSize: 12, color: '#666', whiteSpace: 'nowrap' }}>视图模式:</Text>
      <Segmented
        size="small"
        options={options}
        value={value || 'single'}
        onChange={handleChange}
        disabled={disabled}
      />
    </div>
  )
}

// ==========================================
// TorusArcs - Torus拓扑环绕弧线组件
// ==========================================

export interface TorusArcsProps {
  nodes: Array<{ id: string; x: number; y: number; gridRow?: number; gridCol?: number; gridZ?: number }>
  directTopology: string
  opacity?: number
  /** 获取节点位置的函数（用于支持手动拖拽位置） */
  getNodePosition?: (node: { id: string; x: number; y: number }) => { x: number; y: number }
  selectedLinkId?: string | null
  onLinkClick?: (link: LinkDetail) => void
  connectionMode?: string
  isManualMode?: boolean
}

/**
 * 渲染弧线的通用函数
 */
const renderArc = (
  x1: number, y1: number, x2: number, y2: number,
  key: string, offset: number, opacity: number,
  sourceId: string, targetId: string,
  selectedLinkId: string | null | undefined,
  onLinkClick: ((link: LinkDetail) => void) | undefined,
  connectionMode: string | undefined,
  isManualMode: boolean | undefined
): JSX.Element | null => {
  const midX = (x1 + x2) / 2
  const midY = (y1 + y2) / 2
  const dx = x2 - x1
  const dy = y2 - y1
  const dist = Math.sqrt(dx * dx + dy * dy)
  if (dist < 1) return null
  const bulge = dist * 0.25 + offset * 8
  const perpX = -dy / dist
  const perpY = dx / dist
  const ctrlX = midX + perpX * bulge
  const ctrlY = midY + perpY * bulge
  const pathD = `M ${x1} ${y1} Q ${ctrlX} ${ctrlY}, ${x2} ${y2}`

  const edgeId = `${sourceId}-${targetId}`
  const isSelected = selectedLinkId === edgeId || selectedLinkId === `${targetId}-${sourceId}`

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (connectionMode !== 'view' || isManualMode) return
    onLinkClick?.({
      id: edgeId,
      sourceId,
      sourceLabel: sourceId.split('/').pop() || sourceId,
      sourceType: 'chip',
      targetId,
      targetLabel: targetId.split('/').pop() || targetId,
      targetType: 'chip',
      isManual: false
    })
  }

  return (
    <g key={key} style={{ cursor: connectionMode === 'view' && !isManualMode ? 'pointer' : 'default' }}>
      {/* 透明点击层 */}
      <path d={pathD} fill="none" stroke="transparent" strokeWidth={16} onClick={handleClick} />
      {/* 可见曲线 */}
      <path
        d={pathD}
        fill="none"
        stroke={isSelected ? '#2563eb' : '#999'}
        strokeWidth={isSelected ? 2.5 : 1.5}
        strokeOpacity={isSelected ? 1 : opacity}
        style={{
          pointerEvents: 'none',
          transition: 'stroke-opacity 0.2s ease',
          filter: isSelected ? 'drop-shadow(0 0 4px #2563eb)' : 'none'
        }}
      />
    </g>
  )
}

export const TorusArcs: React.FC<TorusArcsProps> = ({
  nodes,
  directTopology,
  opacity = 0.6,
  getNodePosition,
  selectedLinkId,
  onLinkClick,
  connectionMode,
  isManualMode,
}) => {
  const getPos = (node: { id: string; x: number; y: number }) => {
    if (getNodePosition) return getNodePosition(node)
    return { x: node.x, y: node.y }
  }

  if (directTopology === 'torus_2d') {
    const { cols, rows } = getTorusGridSize(nodes.length)
    if (cols < 2 && rows < 2) return null

    const rowArcs: { x1: number; y1: number; x2: number; y2: number; firstId: string; lastId: string }[] = []
    const colArcs: { x1: number; y1: number; x2: number; y2: number; firstId: string; lastId: string }[] = []

    // Torus: 只画首尾环绕弧
    for (let r = 0; r < rows; r++) {
      const nodesInRow = nodes.filter(n => n.gridRow === r).sort((a, b) => (a.gridCol || 0) - (b.gridCol || 0))
      if (nodesInRow.length >= 3) {
        const first = nodesInRow[0]
        const last = nodesInRow[nodesInRow.length - 1]
        const firstPos = getPos(first)
        const lastPos = getPos(last)
        rowArcs.push({ x1: firstPos.x, y1: firstPos.y, x2: lastPos.x, y2: lastPos.y, firstId: first.id, lastId: last.id })
      }
    }

    for (let c = 0; c < cols; c++) {
      const nodesInCol = nodes.filter(n => n.gridCol === c).sort((a, b) => (a.gridRow || 0) - (b.gridRow || 0))
      if (nodesInCol.length >= 3) {
        const first = nodesInCol[0]
        const last = nodesInCol[nodesInCol.length - 1]
        const firstPos = getPos(first)
        const lastPos = getPos(last)
        colArcs.push({ x1: firstPos.x, y1: firstPos.y, x2: lastPos.x, y2: lastPos.y, firstId: first.id, lastId: last.id })
      }
    }

    return (
      <g>
        {rowArcs.map((arc, i) => renderArc(arc.x1, arc.y1, arc.x2, arc.y2, `row-arc-${i}`, i, opacity, arc.firstId, arc.lastId, selectedLinkId, onLinkClick, connectionMode, isManualMode))}
        {colArcs.map((arc, i) => renderArc(arc.x1, arc.y1, arc.x2, arc.y2, `col-arc-${i}`, i, opacity, arc.firstId, arc.lastId, selectedLinkId, onLinkClick, connectionMode, isManualMode))}
      </g>
    )
  }

  if (directTopology === 'full_mesh_2d') {
    const { cols, rows } = getTorusGridSize(nodes.length)
    if (cols < 2 && rows < 2) return null

    const arcs: JSX.Element[] = []

    // 2D FullMesh: 行内全连接，非相邻节点用曲线
    for (let r = 0; r < rows; r++) {
      const nodesInRow = nodes.filter(n => n.gridRow === r).sort((a, b) => (a.gridCol || 0) - (b.gridCol || 0))
      // 对于行内所有非相邻的节点对，画曲线
      for (let i = 0; i < nodesInRow.length; i++) {
        for (let j = i + 2; j < nodesInRow.length; j++) {
          const n1 = nodesInRow[i]
          const n2 = nodesInRow[j]
          const pos1 = getPos(n1)
          const pos2 = getPos(n2)
          const arc = renderArc(pos1.x, pos1.y, pos2.x, pos2.y, `row-${r}-${i}-${j}`, j - i - 1, opacity * 0.8, n1.id, n2.id, selectedLinkId, onLinkClick, connectionMode, isManualMode)
          if (arc) arcs.push(arc)
        }
      }
    }

    // 列内全连接，非相邻节点用曲线
    for (let c = 0; c < cols; c++) {
      const nodesInCol = nodes.filter(n => n.gridCol === c).sort((a, b) => (a.gridRow || 0) - (b.gridRow || 0))
      for (let i = 0; i < nodesInCol.length; i++) {
        for (let j = i + 2; j < nodesInCol.length; j++) {
          const n1 = nodesInCol[i]
          const n2 = nodesInCol[j]
          const pos1 = getPos(n1)
          const pos2 = getPos(n2)
          const arc = renderArc(pos1.x, pos1.y, pos2.x, pos2.y, `col-${c}-${i}-${j}`, j - i - 1, opacity * 0.8, n1.id, n2.id, selectedLinkId, onLinkClick, connectionMode, isManualMode)
          if (arc) arcs.push(arc)
        }
      }
    }

    return <g>{arcs}</g>
  }

  if (directTopology === 'torus_3d') {
    const { dim, layers } = getTorus3DSize(nodes.length)
    if (dim < 2) return null

    const arcs: JSX.Element[] = []

    // X方向环绕弧
    for (let z = 0; z < layers; z++) {
      for (let r = 0; r < dim; r++) {
        const rowNodes = nodes.filter(n => n.gridZ === z && n.gridRow === r).sort((a, b) => (a.gridCol || 0) - (b.gridCol || 0))
        if (rowNodes.length >= 3) {
          const first = rowNodes[0]
          const last = rowNodes[rowNodes.length - 1]
          const firstPos = getPos(first)
          const lastPos = getPos(last)
          const arc = renderArc(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `x-arc-z${z}-r${r}`, z + r, opacity * 0.8, first.id, last.id, selectedLinkId, onLinkClick, connectionMode, isManualMode)
          if (arc) arcs.push(arc)
        }
      }
    }

    // Y方向环绕弧
    for (let z = 0; z < layers; z++) {
      for (let c = 0; c < dim; c++) {
        const colNodes = nodes.filter(n => n.gridZ === z && n.gridCol === c).sort((a, b) => (a.gridRow || 0) - (b.gridRow || 0))
        if (colNodes.length >= 3) {
          const first = colNodes[0]
          const last = colNodes[colNodes.length - 1]
          const firstPos = getPos(first)
          const lastPos = getPos(last)
          const arc = renderArc(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `y-arc-z${z}-c${c}`, z + c, opacity * 0.8, first.id, last.id, selectedLinkId, onLinkClick, connectionMode, isManualMode)
          if (arc) arcs.push(arc)
        }
      }
    }

    // Z方向环绕弧
    for (let r = 0; r < dim; r++) {
      for (let c = 0; c < dim; c++) {
        const layerNodes = nodes.filter(n => n.gridRow === r && n.gridCol === c).sort((a, b) => (a.gridZ || 0) - (b.gridZ || 0))
        if (layerNodes.length >= 3) {
          const first = layerNodes[0]
          const last = layerNodes[layerNodes.length - 1]
          const firstPos = getPos(first)
          const lastPos = getPos(last)
          const arc = renderArc(firstPos.x, firstPos.y, lastPos.x, lastPos.y, `z-arc-r${r}-c${c}`, r + c, opacity * 0.8, first.id, last.id, selectedLinkId, onLinkClick, connectionMode, isManualMode)
          if (arc) arcs.push(arc)
        }
      }
    }

    return <g>{arcs}</g>
  }

  return null
}
