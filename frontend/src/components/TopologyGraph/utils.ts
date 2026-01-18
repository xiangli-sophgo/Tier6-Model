/**
 * TopologyGraph 工具函数集合
 * 包含: 常量配置, 节点构建, 边构建, 节点形状渲染
 */
import React from 'react'
import { HierarchyLevel, SWITCH_LAYER_COLORS, ManualConnection, HierarchicalTopology } from '../../types'
import { Node, Edge } from './shared'

// ==========================================
// 常量和配置
// ==========================================

export interface LevelConfig {
  switchHierarchy: 'inter_pod' | 'inter_rack' | 'inter_board' | 'inter_chip'
  childKey: 'pods' | 'racks' | 'boards' | 'chips'
  nodeType: 'pod' | 'rack' | 'board' | 'chip'
  upperLevel: HierarchyLevel | null
  lowerLevel: HierarchyLevel | null
}

export const LEVEL_CONFIGS: Record<string, LevelConfig> = {
  datacenter: {
    switchHierarchy: 'inter_pod',
    childKey: 'pods',
    nodeType: 'pod',
    upperLevel: null,
    lowerLevel: 'pod',
  },
  pod: {
    switchHierarchy: 'inter_rack',
    childKey: 'racks',
    nodeType: 'rack',
    upperLevel: 'datacenter',
    lowerLevel: 'rack',
  },
  rack: {
    switchHierarchy: 'inter_board',
    childKey: 'boards',
    nodeType: 'board',
    upperLevel: 'pod',
    lowerLevel: 'board',
  },
  board: {
    switchHierarchy: 'inter_chip',
    childKey: 'chips',
    nodeType: 'chip',
    upperLevel: 'rack',
    lowerLevel: null,
  },
}

export const LEVEL_PAIR_CONFIGS = {
  datacenter_pod: {
    upperLevel: 'datacenter' as HierarchyLevel,
    lowerLevel: 'pod' as HierarchyLevel,
    upperChildKey: 'pods' as const,
    lowerChildKey: 'racks' as const,
    upperSwitchHierarchy: 'inter_pod',
    lowerSwitchHierarchy: 'inter_rack',
  },
  pod_rack: {
    upperLevel: 'pod' as HierarchyLevel,
    lowerLevel: 'rack' as HierarchyLevel,
    upperChildKey: 'racks' as const,
    lowerChildKey: 'boards' as const,
    upperSwitchHierarchy: 'inter_rack',
    lowerSwitchHierarchy: 'inter_board',
  },
  rack_board: {
    upperLevel: 'rack' as HierarchyLevel,
    lowerLevel: 'board' as HierarchyLevel,
    upperChildKey: 'boards' as const,
    lowerChildKey: 'chips' as const,
    upperSwitchHierarchy: 'inter_board',
    lowerSwitchHierarchy: 'inter_chip',
  },
  board_chip: {
    upperLevel: 'board' as HierarchyLevel,
    lowerLevel: 'chip' as HierarchyLevel,
    upperChildKey: 'chips' as const,
    lowerChildKey: null,
    upperSwitchHierarchy: 'inter_chip',
    lowerSwitchHierarchy: null,
  },
}

export const DEFAULT_NODE_COLORS: Record<string, string> = {
  pod: '#1890ff',
  rack: '#52c41a',
  board: '#722ed1',
  chip: '#eb2f96',
  switch: '#666',
}

export const ZOOM_LIMITS = {
  min: 0.2,
  max: 3,
  step: 0.2,
}

export const ALIGNMENT_THRESHOLD = 8

// ==========================================
// 节点构建工具函数
// ==========================================

type Connection = HierarchicalTopology['connections'][0]

export function convertSwitchesToNodes(
  switches: HierarchicalTopology['switches'],
  hierarchyLevel: string,
  parentId?: string
): Node[] {
  if (!switches) return []

  return switches
    .filter(s => {
      if (s.hierarchy_level !== hierarchyLevel) return false
      if (parentId !== undefined && s.parent_id !== parentId) return false
      return true
    })
    .map(sw => ({
      id: sw.id,
      label: sw.label,
      type: 'switch',
      subType: sw.layer,
      isSwitch: true,
      x: 0,
      y: 0,
      color: SWITCH_LAYER_COLORS[sw.layer] || '#666',
      portInfo: {
        uplink: sw.uplink_ports_used,
        downlink: sw.downlink_ports_used,
        inter: sw.inter_ports_used,
      },
    }))
}

export function getSwitchIds(
  switches: HierarchicalTopology['switches'],
  hierarchyLevel: string,
  parentId?: string
): Set<string> {
  if (!switches) return new Set()

  return new Set(
    switches
      .filter(s => {
        if (s.hierarchy_level !== hierarchyLevel) return false
        if (parentId !== undefined && s.parent_id !== parentId) return false
        return true
      })
      .map(s => s.id)
  )
}

export function buildSwitchLabelsMap(
  switches: HierarchicalTopology['switches'],
  hierarchyLevel: string,
  parentId?: string
): Record<string, string> {
  if (!switches) return {}

  const labels: Record<string, string> = {}
  switches
    .filter(s => {
      if (s.hierarchy_level !== hierarchyLevel) return false
      if (parentId !== undefined && s.parent_id !== parentId) return false
      return true
    })
    .forEach(s => {
      labels[s.id] = s.label
    })
  return labels
}

export function buildChildSwitchToParentMap(
  switches: HierarchicalTopology['switches'],
  childSwitchHierarchy: string,
  parentIds: Set<string>
): Record<string, string> {
  if (!switches) return {}

  const mapping: Record<string, string> = {}
  switches
    .filter((s): s is typeof s & { parent_id: string } =>
      s.hierarchy_level === childSwitchHierarchy && !!s.parent_id && parentIds.has(s.parent_id)
    )
    .forEach(s => {
      mapping[s.id] = s.parent_id
    })
  return mapping
}

// ==========================================
// 边构建工具函数
// ==========================================

export function mergeManualConnections(
  connections: Connection[],
  manualConnections: ManualConnection[]
): Connection[] {
  return [
    ...connections,
    ...manualConnections.map(mc => ({
      source: mc.source,
      target: mc.target,
      type: 'manual' as const,
      bandwidth: mc.bandwidth,
      latency: mc.latency,
      is_manual: true,
    }))
  ]
}

export function buildSwitchToNodesMap(
  connections: Connection[],
  internalSwitchIds: Set<string>,
  externalSwitchIds: Set<string>
): Record<string, string[]> {
  const switchToNodes: Record<string, string[]> = {}

  connections
    .filter(c => {
      const internalToExternal = internalSwitchIds.has(c.source) && externalSwitchIds.has(c.target)
      const externalToInternal = externalSwitchIds.has(c.source) && internalSwitchIds.has(c.target)
      return internalToExternal || externalToInternal
    })
    .forEach(c => {
      const externalSwitch = externalSwitchIds.has(c.source) ? c.source : c.target
      const internalNode = externalSwitchIds.has(c.source) ? c.target : c.source
      if (!switchToNodes[externalSwitch]) switchToNodes[externalSwitch] = []
      if (!switchToNodes[externalSwitch].includes(internalNode)) {
        switchToNodes[externalSwitch].push(internalNode)
      }
    })

  return switchToNodes
}

export function generateIndirectEdges(
  switchToNodesMap: Record<string, string[]>,
  switchLabels: Record<string, string>
): Edge[] {
  const edges: Edge[] = []
  const addedPairs = new Set<string>()

  Object.entries(switchToNodesMap).forEach(([switchId, nodes]) => {
    if (nodes.length < 2) return
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pairKey = [nodes[i], nodes[j]].sort().join('-')
        if (addedPairs.has(pairKey)) continue
        addedPairs.add(pairKey)
        edges.push({
          source: nodes[i],
          target: nodes[j],
          isSwitch: true,
          isIndirect: true,
          viaNodeId: switchId,
          viaNodeLabel: switchLabels[switchId] || switchId,
        })
      }
    }
  })

  return edges
}

export function processExternalSwitchConnections(
  connections: Connection[],
  internalSwitchIds: Set<string>,
  externalSwitchIds: Set<string>,
  externalLabels: Record<string, string>
): Edge[] {
  const edges: Edge[] = []

  connections
    .filter(c => {
      const internalToExternal = internalSwitchIds.has(c.source) && externalSwitchIds.has(c.target)
      const externalToInternal = externalSwitchIds.has(c.source) && internalSwitchIds.has(c.target)
      return internalToExternal || externalToInternal
    })
    .forEach(c => {
      const isSourceInternal = internalSwitchIds.has(c.source)
      const internalNode = isSourceInternal ? c.source : c.target
      const externalNode = isSourceInternal ? c.target : c.source
      edges.push({
        source: internalNode,
        target: internalNode,
        bandwidth: c.bandwidth,
        latency: c.latency,
        isSwitch: true,
        isExternal: true,
        externalDirection: 'upper',
        externalNodeId: externalNode,
        externalNodeLabel: externalLabels[externalNode] || externalNode,
      })
    })

  return edges
}

export function determineExternalDirection(
  externalNodeId: string,
  currentId: string
): 'upper' | 'lower' {
  const externalParts = externalNodeId.split('/')
  const currentParts = currentId.split('/')
  return externalParts.length > currentParts.length ? 'lower' : 'upper'
}

export function generateExternalLabel(nodeId: string): string {
  const parts = nodeId.split('/')
  if (parts.length >= 2) {
    return parts.slice(-2).join('/').replace(/_/g, ' ')
  }
  return nodeId
}

export function processCrossLevelConnections(
  connections: Connection[],
  internalNodeIds: Set<string>,
  currentId: string,
  upperSwitchIds: Set<string>,
  layerSwitchIds: Set<string>
): Edge[] {
  const edges: Edge[] = []

  connections
    .filter(c => {
      const sourceInternal = internalNodeIds.has(c.source)
      const targetInternal = internalNodeIds.has(c.target)
      if (!((sourceInternal && !targetInternal) || (!sourceInternal && targetInternal))) return false

      const externalNode = sourceInternal ? c.target : c.source
      if (upperSwitchIds.has(externalNode) || layerSwitchIds.has(externalNode)) return false
      return true
    })
    .forEach(c => {
      const isSourceInternal = internalNodeIds.has(c.source)
      const internalNode = isSourceInternal ? c.source : c.target
      const externalNode = isSourceInternal ? c.target : c.source

      const externalDirection = determineExternalDirection(externalNode, currentId)
      const externalLabel = generateExternalLabel(externalNode)

      edges.push({
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

  return edges
}

export function filterDirectConnections(
  connections: Connection[],
  nodeIds: Set<string>,
  switchIds: Set<string>
): Edge[] {
  return connections
    .filter(c => {
      const sourceValid = nodeIds.has(c.source) || switchIds.has(c.source)
      const targetValid = nodeIds.has(c.target) || switchIds.has(c.target)
      return sourceValid && targetValid
    })
    .map(c => ({
      source: c.source,
      target: c.target,
      bandwidth: c.bandwidth,
      latency: c.latency,
      isSwitch: c.type === 'switch',
    }))
}

// ==========================================
// 节点形状渲染
// ==========================================

export function renderNodeShape(node: Node): React.ReactNode {
  const isSwitch = node.isSwitch
  if (isSwitch) {
    return (
      React.createElement('g', null,
        React.createElement('rect', { x: -36, y: -14, width: 72, height: 28, rx: 3, fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('rect', { x: -32, y: -10, width: 64, height: 16, rx: 2, fill: 'rgba(0,0,0,0.15)' }),
        React.createElement('rect', { x: -28, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: -20, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: -12, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: -4, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: 6, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: 14, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('rect', { x: 22, y: -6, width: 6, height: 8, rx: 1, fill: 'rgba(255,255,255,0.5)' }),
        React.createElement('circle', { cx: -28, cy: 8, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: -22, cy: 8, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: -16, cy: 8, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: -10, cy: 8, r: 2, fill: '#fbbf24' }),
        React.createElement('rect', { x: 10, y: 4, width: 18, height: 6, rx: 1, fill: 'rgba(255,255,255,0.2)' })
      )
    )
  }
  if (node.type === 'pod') {
    return (
      React.createElement('g', null,
        React.createElement('rect', { x: -28, y: -12, width: 56, height: 32, rx: 3, fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('polygon', { points: '-32,-12 0,-24 32,-12', fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('rect', { x: -20, y: -4, width: 8, height: 8, rx: 1, fill: 'rgba(255,255,255,0.3)' }),
        React.createElement('rect', { x: -6, y: -4, width: 8, height: 8, rx: 1, fill: 'rgba(255,255,255,0.3)' }),
        React.createElement('rect', { x: 8, y: -4, width: 8, height: 8, rx: 1, fill: 'rgba(255,255,255,0.3)' }),
        React.createElement('rect', { x: -5, y: 8, width: 10, height: 12, rx: 1, fill: 'rgba(255,255,255,0.4)' })
      )
    )
  }
  if (node.type === 'rack') {
    return (
      React.createElement('g', null,
        React.createElement('rect', { x: -18, y: -28, width: 36, height: 56, rx: 3, fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('line', { x1: -14, y1: -16, x2: 14, y2: -16, stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }),
        React.createElement('line', { x1: -14, y1: -4, x2: 14, y2: -4, stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }),
        React.createElement('line', { x1: -14, y1: 8, x2: 14, y2: 8, stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }),
        React.createElement('line', { x1: -14, y1: 20, x2: 14, y2: 20, stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }),
        React.createElement('circle', { cx: 10, cy: -22, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: 10, cy: -10, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: 10, cy: 2, r: 2, fill: '#4ade80' }),
        React.createElement('circle', { cx: 10, cy: 14, r: 2, fill: '#fbbf24' })
      )
    )
  }
  if (node.type === 'board') {
    return (
      React.createElement('g', null,
        React.createElement('rect', { x: -32, y: -18, width: 64, height: 36, rx: 2, fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('path', { d: 'M-24,-10 L-24,-2 L-16,-2 L-16,6 L-8,6', stroke: 'rgba(255,255,255,0.25)', strokeWidth: 1.5, fill: 'none' }),
        React.createElement('path', { d: 'M8,-10 L8,0 L16,0 L16,8 L24,8', stroke: 'rgba(255,255,255,0.25)', strokeWidth: 1.5, fill: 'none' }),
        React.createElement('rect', { x: -8, y: -8, width: 16, height: 16, rx: 1, fill: 'rgba(0,0,0,0.2)', stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }),
        React.createElement('circle', { cx: -26, cy: 0, r: 3, fill: 'rgba(255,255,255,0.4)' }),
        React.createElement('circle', { cx: 26, cy: 0, r: 3, fill: 'rgba(255,255,255,0.4)' })
      )
    )
  }
  if (node.type === 'npu' || node.type === 'cpu') {
    return (
      React.createElement('g', null,
        React.createElement('rect', { x: -20, y: -20, width: 40, height: 40, rx: 2, fill: node.color, stroke: '#fff', strokeWidth: 2 }),
        React.createElement('rect', { x: -12, y: -26, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: -2, y: -26, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: 8, y: -26, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: -12, y: 20, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: -2, y: 20, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: 8, y: 20, width: 4, height: 6, fill: node.color }),
        React.createElement('rect', { x: -26, y: -12, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: -26, y: -2, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: -26, y: 8, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: 20, y: -12, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: 20, y: -2, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: 20, y: 8, width: 6, height: 4, fill: node.color }),
        React.createElement('rect', { x: -10, y: -10, width: 20, height: 20, rx: 1, fill: 'rgba(255,255,255,0.15)' })
      )
    )
  }
  return React.createElement('rect', { x: -25, y: -18, width: 50, height: 36, rx: 6, fill: node.color, stroke: '#fff', strokeWidth: 2 })
}
