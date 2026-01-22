/**
 * 拓扑设置页面
 * 包含 3D 视图、拓扑图视图、配置面板等
 */

import React, { useCallback, useState, useRef, useEffect } from 'react'
import { Layout, Spin, Card, Descriptions, Tag, Collapse } from 'antd'
import { Scene3D } from '@/components/Scene3D'
import { TopologyGraph, NodeDetail } from '@/components/TopologyGraph'
import { ConfigPanel } from '@/components/ConfigPanel'
import { useWorkbench } from '@/contexts/WorkbenchContext'

const { Sider, Content } = Layout

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
      } else {
        ui.setFocusedLevel(null)
      }
    },
    [ui]
  )

  return (
    <Layout style={{ height: '100%', background: '#fff' }}>
      {/* 标题栏 */}
      <div
        style={{
          padding: '16px 24px',
          borderBottom: '1px solid #f0f0f0',
          background: '#fff',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 20, fontWeight: 600, color: '#1a1a1a' }}>
            互联拓扑
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c' }}>
            配置Tier6+互联拓扑
          </span>
        </div>
      </div>

      {/* 主内容区 */}
      <Layout style={{ height: 'calc(100% - 73px)' }}>
        {/* 左侧配置面板 */}
      <Sider
        width={siderWidth}
        style={{
          background: '#EFEFEF',
          padding: 16,
          overflow: 'auto',
          position: 'relative',
          borderRight: '1px solid #E5E5E5',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          <ConfigPanel
            topology={topology.topology}
            onGenerate={topology.handleGenerate}
            loading={topology.loading}
            currentLevel={getCurrentLevel()}
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
          <Card
            title={`节点详情: ${ui.selectedNode.label}`}
            size="small"
            style={{ marginTop: 16 }}
            extra={<a onClick={() => ui.setSelectedNode(null)}>关闭</a>}
          >
            <Descriptions column={1} size="small">
              <Descriptions.Item label="ID">{ui.selectedNode.id}</Descriptions.Item>
              <Descriptions.Item label="类型">
                <Tag color={ui.selectedNode.type === 'switch' ? 'blue' : 'green'}>
                  {ui.selectedNode.subType?.toUpperCase() || ui.selectedNode.type.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              {ui.selectedNode.portInfo && (
                <Descriptions.Item label="端口">
                  上行: {ui.selectedNode.portInfo.uplink} | 下行:{' '}
                  {ui.selectedNode.portInfo.downlink} | 互联: {ui.selectedNode.portInfo.inter}
                </Descriptions.Item>
              )}
              <Descriptions.Item label="连接数">
                {ui.selectedNode.connections.length}
              </Descriptions.Item>
            </Descriptions>
            {ui.selectedNode.connections.length > 0 && (
              <Collapse
                size="small"
                style={{ marginTop: 8 }}
                items={[
                  {
                    key: 'connections',
                    label: `连接列表 (${ui.selectedNode.connections.length})`,
                    children: (
                      <div style={{ maxHeight: 150, overflow: 'auto' }}>
                        {ui.selectedNode.connections.map((conn, idx) => (
                          <div
                            key={idx}
                            style={{
                              fontSize: 12,
                              padding: '2px 0',
                              borderBottom: '1px solid #f0f0f0',
                            }}
                          >
                            {conn.label}
                            {conn.bandwidth && (
                              <span style={{ color: '#999', marginLeft: 8 }}>
                                {conn.bandwidth} GB/s
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ),
                  },
                ]}
              />
            )}
          </Card>
        )}

        {/* 连接详情卡片 */}
        {ui.selectedLink && (
          <Card
            title="连接详情"
            size="small"
            style={{ marginTop: 16 }}
            extra={<a onClick={() => ui.setSelectedLink(null)}>关闭</a>}
          >
            <Descriptions column={1} size="small">
              <Descriptions.Item label="源节点">
                <Tag color="green">{ui.selectedLink.sourceLabel}</Tag>
                <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>
                  ({ui.selectedLink.sourceType.toUpperCase()})
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="目标节点">
                <Tag color="blue">{ui.selectedLink.targetLabel}</Tag>
                <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>
                  ({ui.selectedLink.targetType.toUpperCase()})
                </span>
              </Descriptions.Item>
              {ui.selectedLink.bandwidth && (
                <Descriptions.Item label="带宽">
                  {ui.selectedLink.bandwidth} GB/s
                </Descriptions.Item>
              )}
              {ui.selectedLink.latency && (
                <Descriptions.Item label="延迟">{ui.selectedLink.latency} us</Descriptions.Item>
              )}
              <Descriptions.Item label="类型">
                <Tag color={ui.selectedLink.isManual ? 'orange' : 'default'}>
                  {ui.selectedLink.isManual ? '手动连接' : '自动连接'}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        )}

        {/* 拖拽手柄 */}
        <div
          onMouseDown={handleMouseDown}
          style={{
            position: 'absolute',
            top: 0,
            right: 0,
            width: 4,
            height: '100%',
            cursor: 'col-resize',
            background: isDragging ? '#4f46e5' : 'transparent',
            transition: 'background 0.15s',
            zIndex: 10,
          }}
          onMouseEnter={(e) => {
            if (!isDragging) (e.target as HTMLElement).style.background = '#e2e8f0'
          }}
          onMouseLeave={(e) => {
            if (!isDragging) (e.target as HTMLElement).style.background = 'transparent'
          }}
        />
      </Sider>

      {/* 右侧内容区域 */}
      <Layout>
        {/* 内容区域 */}
        <Content
          style={{ position: 'relative', background: '#ffffff', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
        >
          {topology.loading && !topology.topology ? (
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
              }}
            >
              <Spin size="large" tip="加载中..." />
            </div>
          ) : viewMode === '3d' ? (
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
        </Content>
      </Layout>
      </Layout>
    </Layout>
  )
}
