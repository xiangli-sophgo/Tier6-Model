/**
 * 拓扑设置页面
 * 包含 3D 视图、拓扑图视图、配置面板等
 * 3D 视图使用懒加载优化初始包体积
 */

import React, { useCallback, useState, useRef, useEffect, lazy, Suspense } from 'react'
import { Loader2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { TopologyGraph, NodeDetail } from '@/components/TopologyGraph'
import { ConfigPanel } from '@/components/ConfigPanel'
import { PageHeader } from '@/components/ui/page-header'
import { useWorkbench } from '@/contexts/WorkbenchContext'

// 懒加载 Scene3D 组件（只在需要时加载 Three.js 相关代码）
const Scene3D = lazy(() => import('@/components/Scene3D').then(module => ({ default: module.Scene3D })))

// 侧边栏宽度常量
const SIDER_WIDTH_KEY = 'tier6_topology_sider_width'
const DEFAULT_SIDER_WIDTH = 520
const MIN_SIDER_WIDTH = 380
const MAX_SIDER_WIDTH = 900

export const TopologySetup: React.FC = () => {
  const { topology, connection, navigation, ui, analysis, getCurrentLevel, currentViewConnections } = useWorkbench()
  // 使用全局状态管理视图模式，切换页面后会保持
  const viewMode = ui.topologyPageViewMode
  const setViewMode = ui.setTopologyPageViewMode

  // 侧边栏宽度
  const [siderWidth, setSiderWidth] = useState(() => {
    const cached = localStorage.getItem(SIDER_WIDTH_KEY)
    return cached
      ? Math.max(MIN_SIDER_WIDTH, Math.min(MAX_SIDER_WIDTH, parseInt(cached, 10)))
      : DEFAULT_SIDER_WIDTH
  })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)

  // 选中的芯片 ID（用于从视图点击芯片时跳转到配置）
  const [selectedChipId, setSelectedChipId] = useState<string | undefined>()

  // 拖拽处理
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setIsDragging(true)
      dragStartX.current = e.clientX
      dragStartWidth.current = siderWidth
      e.preventDefault()
    },
    [siderWidth]
  )

  useEffect(() => {
    if (!isDragging) return
    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - dragStartX.current
      const newWidth = Math.max(
        MIN_SIDER_WIDTH,
        Math.min(MAX_SIDER_WIDTH, dragStartWidth.current + delta)
      )
      setSiderWidth(newWidth)
    }
    const handleMouseUp = () => {
      setIsDragging(false)
      localStorage.setItem(SIDER_WIDTH_KEY, siderWidth.toString())
    }
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, siderWidth])

  // 3D视图节点选择处理
  const handleScene3DNodeSelect = useCallback(
    (
      nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch',
      nodeId: string,
      label: string,
      _info: Record<string, string | number>,
      subType?: string
    ) => {
      const connections: { id: string; label: string; bandwidth?: number }[] = []
      if (topology.topology?.connections) {
        topology.topology.connections.forEach((conn) => {
          if (conn.source === nodeId) {
            connections.push({
              id: conn.target,
              label: `→ ${conn.target}`,
              bandwidth: conn.bandwidth,
            })
          } else if (conn.target === nodeId) {
            connections.push({
              id: conn.source,
              label: `← ${conn.source}`,
              bandwidth: conn.bandwidth,
            })
          }
        })
      }
      ui.setSelectedNode({ id: nodeId, label, type: nodeType, subType, connections })

      // 如果点击的是芯片，设置选中的芯片 ID 并切换到 Chip Tab
      if (nodeType === 'chip') {
        setSelectedChipId(nodeId)
      }
    },
    [topology.topology, ui]
  )

  // 节点双击导航处理
  const handleNodeDoubleClick = useCallback(
    (nodeId: string, nodeType: string) => {
      if (connection.multiLevelOptions.enabled) {
        let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' =
          connection.multiLevelOptions.levelPair || 'datacenter_pod'
        if (nodeType === 'pod') newLevelPair = 'datacenter_pod'
        else if (nodeType === 'rack') newLevelPair = 'pod_rack'
        else if (nodeType === 'board') newLevelPair = 'rack_board'
        connection.setMultiLevelOptions({
          ...connection.multiLevelOptions,
          levelPair: newLevelPair,
        })
      }
      const pathParts = nodeId.split('/')
      if (nodeType === 'pod') {
        navigation.navigateToPod(nodeId)
      } else if (nodeType === 'rack') {
        if (pathParts.length >= 2) {
          navigation.navigateToRack(pathParts[0], nodeId)
        } else if (navigation.currentPod) {
          navigation.navigateToRack(navigation.currentPod.id, nodeId)
        }
      } else if (nodeType === 'board') {
        if (pathParts.length >= 3) {
          navigation.navigateToBoard(pathParts[0], `${pathParts[0]}/${pathParts[1]}`, nodeId)
        } else {
          navigation.navigateTo(nodeId)
        }
      }
    },
    [connection, navigation]
  )

  // 导航返回处理
  const handleNavigateBack = useCallback(() => {
    if (connection.multiLevelOptions.enabled) {
      const newPathLength = navigation.viewState.path.length - 1
      let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' =
        'datacenter_pod'
      if (newPathLength <= 1) newLevelPair = 'datacenter_pod'
      else if (newPathLength === 2) newLevelPair = 'pod_rack'
      else if (newPathLength >= 3) newLevelPair = 'rack_board'
      connection.setMultiLevelOptions({
        ...connection.multiLevelOptions,
        levelPair: newLevelPair,
      })
    }
    navigation.navigateBack()
  }, [connection, navigation])

  // 键盘快捷键处理 - ESC返回上一级
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 如果正在输入框中则忽略
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      // ESC - 返回上一级
      if (e.key === 'Escape' && navigation.canGoBack) {
        e.preventDefault()
        handleNavigateBack()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [navigation.canGoBack, handleNavigateBack])

  // 面包屑导航处理
  const handleBreadcrumbClick = useCallback(
    (index: number) => {
      if (connection.multiLevelOptions.enabled) {
        let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' =
          'datacenter_pod'
        if (index <= 1) newLevelPair = 'datacenter_pod'
        else if (index === 2) newLevelPair = 'pod_rack'
        else if (index === 3) newLevelPair = 'rack_board'
        connection.setMultiLevelOptions({
          ...connection.multiLevelOptions,
          levelPair: newLevelPair,
        })
      }
      navigation.navigateToBreadcrumb(index)
    },
    [connection, navigation]
  )

  // 节点点击处理
  const handleNodeClick = useCallback(
    (node: NodeDetail | null) => {
      ui.setSelectedNode(node)
      if (node) {
        ui.setSelectedLink(null)
        const levelTypes = ['datacenter', 'pod', 'rack', 'board']
        if (node.subType && levelTypes.includes(node.subType)) {
          ui.setFocusedLevel(node.subType as 'datacenter' | 'pod' | 'rack' | 'board')
        } else {
          ui.setFocusedLevel(null)
        }

        // 如果点击的是芯片，设置选中的芯片 ID 并切换到 Chip Tab
        if (node.type === 'chip') {
          setSelectedChipId(node.id)
        }
      } else {
        ui.setFocusedLevel(null)
      }
    },
    [ui]
  )

  return (
    <div className="h-full w-full bg-gradient-to-b from-gray-50 to-white flex flex-col">
      {/* 标题栏 */}
      <PageHeader title="互联拓扑" />

      {/* 主内容区 */}
      <div className="flex-1 flex" style={{ height: 'calc(100% - 73px)' }}>
        {/* 左侧配置面板 */}
        <div
          style={{ width: siderWidth }}
          className="bg-gradient-to-b from-blue-50/50 to-white p-4 overflow-auto relative border-r border-blue-100 flex flex-col"
        >
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          <ConfigPanel
            topology={topology.topology}
            onGenerate={topology.handleGenerate}
            loading={topology.loading}
            currentLevel={getCurrentLevel()}
            selectedChipId={selectedChipId}
            onChipTabActivate={() => setSelectedChipId(undefined)}
            manualConnectionConfig={connection.manualConnectionConfig}
            onManualConnectionConfigChange={connection.setManualConnectionConfig}
            connectionMode={connection.connectionMode}
            onConnectionModeChange={connection.setConnectionMode}
            selectedNodes={connection.selectedNodes}
            onSelectedNodesChange={connection.setSelectedNodes}
            targetNodes={connection.targetNodes}
            onTargetNodesChange={connection.setTargetNodes}
            onBatchConnect={connection.handleBatchConnect}
            onDeleteManualConnection={connection.handleDeleteManualConnection}
            currentViewConnections={currentViewConnections}
            onDeleteConnection={connection.handleDeleteConnection}
            onUpdateConnectionParams={connection.handleUpdateConnectionParams}
            layoutType={connection.layoutType}
            onLayoutTypeChange={connection.setLayoutType}
            viewMode="3d"
            focusedLevel={ui.focusedLevel}
            onTrafficResultChange={analysis.setTrafficResult}
            onAnalysisDataChange={analysis.setDeploymentAnalysisData}
            analysisHistory={analysis.analysisHistory}
            onAddToHistory={analysis.handleAddToHistory}
            onDeleteHistory={analysis.handleDeleteHistory}
            onClearHistory={analysis.handleClearHistory}
          />
        </div>

        {/* 节点详情卡片 */}
        {ui.selectedNode && (
          <div className="mt-4 bg-white rounded-2xl border border-blue-100 p-4 shadow-md card">
            <div className="flex justify-between items-center mb-3">
              <span className="font-semibold text-sm text-blue-900">节点详情: {ui.selectedNode.label}</span>
              <button className="text-blue-600 text-sm hover:text-blue-700 hover:underline" onClick={() => ui.setSelectedNode(null)}>关闭</button>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex">
                <span className="text-text-muted w-16">ID</span>
                <span className="text-text-primary">{ui.selectedNode.id}</span>
              </div>
              <div className="flex items-center">
                <span className="text-text-muted w-16">类型</span>
                <Badge variant="outline" className={ui.selectedNode.type === 'switch' ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-cyan-100 text-cyan-700 border-cyan-200'}>
                  {ui.selectedNode.subType?.toUpperCase() || ui.selectedNode.type.toUpperCase()}
                </Badge>
              </div>
              {ui.selectedNode.portInfo && (
                <div className="flex">
                  <span className="text-text-muted w-16">端口</span>
                  <span className="text-text-primary">上行: {ui.selectedNode.portInfo.uplink} | 下行: {ui.selectedNode.portInfo.downlink} | 互联: {ui.selectedNode.portInfo.inter}</span>
                </div>
              )}
              <div className="flex">
                <span className="text-text-muted w-16">连接数</span>
                <span className="text-text-primary">{ui.selectedNode.connections.length}</span>
              </div>
            </div>
            {ui.selectedNode.connections.length > 0 && (
              <Collapsible className="mt-2">
                <CollapsibleTrigger className="text-sm text-blue-600 hover:text-blue-700">
                  连接列表 ({ui.selectedNode.connections.length})
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="max-h-[150px] overflow-auto mt-2">
                    {ui.selectedNode.connections.map((conn, idx) => (
                      <div key={idx} className="text-xs py-0.5 border-b border-blue-100 text-text-secondary">
                        {conn.label}
                        {conn.bandwidth && (
                          <span className="text-text-muted ml-2">{conn.bandwidth} GB/s</span>
                        )}
                      </div>
                    ))}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}
          </div>
        )}

        {/* 连接详情卡片 */}
        {ui.selectedLink && (
          <div className="mt-4 bg-white rounded-2xl border border-blue-100 p-4 shadow-md card">
            <div className="flex justify-between items-center mb-3">
              <span className="font-semibold text-sm text-blue-900">连接详情</span>
              <button className="text-blue-600 text-sm hover:text-blue-700 hover:underline" onClick={() => ui.setSelectedLink(null)}>关闭</button>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex items-center">
                <span className="text-text-muted w-20">源节点</span>
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">{ui.selectedLink.sourceLabel}</Badge>
                <span className="text-text-muted ml-1 text-xs">({ui.selectedLink.sourceType.toUpperCase()})</span>
              </div>
              <div className="flex items-center">
                <span className="text-text-muted w-20">目标节点</span>
                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">{ui.selectedLink.targetLabel}</Badge>
                <span className="text-text-muted ml-1 text-xs">({ui.selectedLink.targetType.toUpperCase()})</span>
              </div>
              {ui.selectedLink.bandwidth && (
                <div className="flex">
                  <span className="text-text-muted w-20">带宽</span>
                  <span className="text-text-primary">{ui.selectedLink.bandwidth} GB/s</span>
                </div>
              )}
              {ui.selectedLink.latency && (
                <div className="flex">
                  <span className="text-text-muted w-20">延迟</span>
                  <span className="text-text-primary">{ui.selectedLink.latency} us</span>
                </div>
              )}
              <div className="flex items-center">
                <span className="text-text-muted w-20">类型</span>
                <Badge variant="outline" className={ui.selectedLink.isManual ? 'bg-orange-100 text-orange-700 border-orange-200' : 'bg-blue-100 text-blue-700 border-blue-200'}>
                  {ui.selectedLink.isManual ? '手动连接' : '自动连接'}
                </Badge>
              </div>
            </div>
          </div>
        )}

        {/* 拖拽手柄 */}
        <div
          onMouseDown={handleMouseDown}
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize z-10 transition-colors"
          style={{ background: isDragging ? '#2563EB' : 'transparent' }}
          onMouseEnter={(e) => {
            if (!isDragging) (e.target as HTMLElement).style.background = '#BFDBFE'
          }}
          onMouseLeave={(e) => {
            if (!isDragging) (e.target as HTMLElement).style.background = 'transparent'
          }}
        />
      </div>

      {/* 右侧内容区域 */}
      <div className="flex-1 flex flex-col bg-gradient-to-b from-gray-50 to-white">
        {/* 内容区域 */}
        <div className="relative flex-1 bg-transparent flex flex-col overflow-hidden">
          {topology.loading && !topology.topology ? (
            <div className="flex justify-center items-center h-full flex-col gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              <div className="text-text-muted text-sm">加载中...</div>
            </div>
          ) : viewMode === '3d' ? (
            <Suspense
              fallback={
                <div className="flex justify-center items-center h-full flex-col gap-3">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                  <div className="text-gray-500 text-sm">正在加载 3D 视图...</div>
                </div>
              }
            >
              <Scene3D
                topology={topology.topology}
                viewState={navigation.viewState}
                breadcrumbs={navigation.breadcrumbs}
                currentPod={navigation.currentPod}
                currentRack={navigation.currentRack}
                currentBoard={navigation.currentBoard}
                onNavigate={navigation.navigateTo}
                onNavigateToPod={navigation.navigateToPod}
                onNavigateToRack={navigation.navigateToRack}
                onNavigateBack={navigation.navigateBack}
                onBreadcrumbClick={navigation.navigateToBreadcrumb}
                canGoBack={navigation.canGoBack}
                onNodeSelect={handleScene3DNodeSelect}
                viewMode={viewMode}
                onViewModeChange={setViewMode}
              />
            </Suspense>
          ) : (
            <TopologyGraph
              visible={true}
              onClose={() => setViewMode('3d')}
              topology={topology.topology}
              currentLevel={getCurrentLevel()}
              currentPod={navigation.currentPod}
              currentRack={navigation.currentRack}
              currentBoard={navigation.currentBoard}
              onNodeDoubleClick={handleNodeDoubleClick}
              onNodeClick={handleNodeClick}
              onLinkClick={(link) => {
                ui.setSelectedLink(link)
                if (link) ui.setSelectedNode(null)
              }}
              selectedNodeId={ui.selectedNode?.id || null}
              selectedLinkId={ui.selectedLink?.id || null}
              onNavigateBack={handleNavigateBack}
              onBreadcrumbClick={handleBreadcrumbClick}
              breadcrumbs={navigation.breadcrumbs}
              canGoBack={navigation.canGoBack}
              embedded={true}
              connectionMode={connection.connectionMode}
              selectedNodes={connection.selectedNodes}
              onSelectedNodesChange={connection.setSelectedNodes}
              targetNodes={connection.targetNodes}
              onTargetNodesChange={connection.setTargetNodes}
              sourceNode={connection.sourceNode}
              onSourceNodeChange={connection.setSourceNode}
              onManualConnect={connection.handleManualConnect}
              manualConnections={connection.manualConnectionConfig.connections}
              onDeleteManualConnection={connection.handleDeleteManualConnection}
              onDeleteConnection={connection.handleDeleteConnection}
              layoutType={connection.layoutType}
              onLayoutTypeChange={connection.setLayoutType}
              multiLevelOptions={connection.multiLevelOptions}
              onMultiLevelOptionsChange={connection.setMultiLevelOptions}
              trafficResult={null}
              viewMode={viewMode}
              onViewModeChange={setViewMode}
            />
          )}
        </div>
      </div>
      </div>
    </div>
  )
}
