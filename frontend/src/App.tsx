import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Layout, Typography, Spin, Segmented, Card, Descriptions, Tag, Collapse, Button } from 'antd'
import { ArrowLeftOutlined, BookOutlined, LinkOutlined, FileTextOutlined } from '@ant-design/icons'
import { Scene3D } from './components/Scene3D'
import { ConfigPanel } from './components/ConfigPanel'
import { TopologyGraph, NodeDetail } from './components/TopologyGraph'
import { ChartsPanel } from './components/ConfigPanel/DeploymentAnalysis/charts'
import { AnalysisResultDisplay } from './components/ConfigPanel/DeploymentAnalysis'
import { WorkbenchProvider, useWorkbench } from './contexts/WorkbenchContext'
import { KnowledgeGraph, CATEGORY_COLORS, CATEGORY_NAMES, ForceKnowledgeNode } from './components/KnowledgeGraph'
import knowledgeData from './data/knowledge-graph'

const { Header, Sider, Content } = Layout
const { Title } = Typography

// 侧边栏宽度常量
const SIDER_WIDTH_KEY = 'tier6_sider_width_cache'
const DEFAULT_SIDER_WIDTH = 520
const MIN_SIDER_WIDTH = 380
const MAX_SIDER_WIDTH = 900

/**
 * 知识节点详情卡片组件 - 单卡片全高显示
 */
interface KnowledgeNodeCardsProps {
  nodes: ForceKnowledgeNode[]
  onClose: (nodeId: string) => void
  onNodeClick: (node: { id: string; category: string }) => void
}

const KnowledgeNodeCards: React.FC<KnowledgeNodeCardsProps> = ({ nodes, onClose, onNodeClick }) => {
  // 只显示最近选中的一个节点
  const node = nodes[0]
  if (!node) return null

  // 获取相关节点
  const getRelatedNodes = (nodeId: string) => {
    const relatedIds = new Set<string>()
    knowledgeData.relations.forEach(r => {
      if (r.source === nodeId) relatedIds.add(r.target)
      if (r.target === nodeId) relatedIds.add(r.source)
    })
    return knowledgeData.nodes.filter(n => relatedIds.has(n.id))
  }

  // 渲染定义文本（支持 Markdown 格式：**加粗**、\n换行）
  const renderDefinition = (text: string) => {
    // 1. 按换行符拆分
    const lines = text.split('\n')

    return lines.map((line, lineIndex) => {
      // 2. 处理每行中的加粗标记 **text**
      const segments: React.ReactNode[] = []
      const boldRegex = /\*\*([^*]+)\*\*/g
      let lastIndex = 0
      let match

      while ((match = boldRegex.exec(line)) !== null) {
        if (match.index > lastIndex) {
          segments.push(line.slice(lastIndex, match.index))
        }
        segments.push(
          <strong key={`${lineIndex}-${match.index}`} style={{ color: '#4f46e5' }}>
            {match[1]}
          </strong>
        )
        lastIndex = boldRegex.lastIndex
      }
      if (lastIndex < line.length) {
        segments.push(line.slice(lastIndex))
      }

      // 3. 判断是否是分点项（以数字+标点开头）
      const isListItem = /^\d+[）\.\)]\s*/.test(line)

      return (
        <span
          key={lineIndex}
          style={{
            display: lineIndex > 0 || isListItem ? 'block' : undefined,
            marginTop: lineIndex > 0 ? 4 : 0,
            paddingLeft: isListItem ? 12 : 0,
          }}
        >
          {segments.length > 0 ? segments : line}
        </span>
      )
    })
  }

  // 分区样式
  const sectionStyle: React.CSSProperties = {
    background: '#f9fafb',
    border: '1px solid #f3f4f6',
    borderRadius: 6,
    padding: '10px 12px',
    marginBottom: 10,
  }
  const sectionTitleStyle: React.CSSProperties = {
    fontSize: 14,
    fontWeight: 600,
    color: '#374151',
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  }

  const relatedNodes = getRelatedNodes(node.id)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, marginTop: 16, minHeight: 0 }}>
      <Card
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag color={CATEGORY_COLORS[node.category]} style={{ margin: 0, fontSize: 12, flexShrink: 0 }}>
              {CATEGORY_NAMES[node.category]}
            </Tag>
            <span style={{ fontSize: 16, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {node.fullName || node.name}
            </span>
          </div>
        }
        size="small"
        style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
        bodyStyle={{ flex: 1, overflow: 'auto', minHeight: 0 }}
        extra={<a onClick={() => onClose(node.id)}>关闭</a>}
      >
        {/* 定义区域 */}
        <div style={sectionStyle}>
          <div style={sectionTitleStyle}>
            <BookOutlined style={{ color: '#6366f1' }} />
            <span>定义</span>
          </div>
          <div style={{
            fontSize: 15,
            lineHeight: 1.8,
            color: '#1f2937',
          }}>
            {renderDefinition(node.definition)}
          </div>
        </div>

        {/* 相关概念区域 */}
        {relatedNodes.length > 0 && (
          <div style={sectionStyle}>
            <div style={sectionTitleStyle}>
              <LinkOutlined style={{ color: '#10b981' }} />
              <span>相关概念 ({relatedNodes.length})</span>
            </div>
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 6,
            }}>
              {relatedNodes.map(n => (
                <Tag
                  key={n.id}
                  color={CATEGORY_COLORS[n.category as keyof typeof CATEGORY_COLORS]}
                  style={{ cursor: 'pointer', margin: 0, fontSize: 13 }}
                  onClick={() => onNodeClick(n)}
                >
                  {n.name}
                </Tag>
              ))}
            </div>
          </div>
        )}

        {/* 参考资料区域（仅当有 source 时显示）*/}
        {node.source && (
          <div style={{ ...sectionStyle, marginBottom: 0 }}>
            <div style={sectionTitleStyle}>
              <FileTextOutlined style={{ color: '#8b5cf6' }} />
              <span>参考资料</span>
            </div>
            <div style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
              {node.source}
              {(node as any).url && (
                <a
                  href={(node as any).url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ marginLeft: 8, color: '#6366f1' }}
                >
                  <LinkOutlined /> 链接
                </a>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

/**
 * 主工作台内容组件（使用 Context）
 */
const WorkbenchContent: React.FC = () => {
  const { topology, connection, analysis, ui, navigation, currentViewConnections, getCurrentLevel } = useWorkbench()

  // 侧边栏宽度
  const [siderWidth, setSiderWidth] = useState(() => {
    const cached = localStorage.getItem(SIDER_WIDTH_KEY)
    return cached ? Math.max(MIN_SIDER_WIDTH, Math.min(MAX_SIDER_WIDTH, parseInt(cached, 10))) : DEFAULT_SIDER_WIDTH
  })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)

  // 全局键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'Escape' || e.key === 'Backspace') {
        e.preventDefault()
        if (navigation.canGoBack) navigation.navigateBack()
        return
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        if (navigation.canGoHistoryBack) navigation.navigateHistoryBack()
        return
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        if (navigation.canGoHistoryForward) navigation.navigateHistoryForward()
        return
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [navigation])

  // 拖拽处理
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true)
    dragStartX.current = e.clientX
    dragStartWidth.current = siderWidth
    e.preventDefault()
  }, [siderWidth])

  useEffect(() => {
    if (!isDragging) return
    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - dragStartX.current
      const newWidth = Math.max(MIN_SIDER_WIDTH, Math.min(MAX_SIDER_WIDTH, dragStartWidth.current + delta))
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
  const handleScene3DNodeSelect = useCallback((
    nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch',
    nodeId: string,
    label: string,
    _info: Record<string, string | number>,
    subType?: string
  ) => {
    const connections: { id: string; label: string; bandwidth?: number }[] = []
    if (topology.topology?.connections) {
      topology.topology.connections.forEach(conn => {
        if (conn.source === nodeId) {
          connections.push({ id: conn.target, label: `→ ${conn.target}`, bandwidth: conn.bandwidth })
        } else if (conn.target === nodeId) {
          connections.push({ id: conn.source, label: `← ${conn.source}`, bandwidth: conn.bandwidth })
        }
      })
    }
    ui.setSelectedNode({ id: nodeId, label, type: nodeType, subType, connections })
  }, [topology.topology, ui])

  // 节点双击导航处理
  const handleNodeDoubleClick = useCallback((nodeId: string, nodeType: string) => {
    if (connection.multiLevelOptions.enabled) {
      let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = connection.multiLevelOptions.levelPair || 'datacenter_pod'
      if (nodeType === 'pod') newLevelPair = 'datacenter_pod'
      else if (nodeType === 'rack') newLevelPair = 'pod_rack'
      else if (nodeType === 'board') newLevelPair = 'rack_board'
      connection.setMultiLevelOptions({ ...connection.multiLevelOptions, levelPair: newLevelPair })
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
  }, [connection, navigation])

  // 导航返回处理
  const handleNavigateBack = useCallback(() => {
    if (connection.multiLevelOptions.enabled) {
      const newPathLength = navigation.viewState.path.length - 1
      let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = 'datacenter_pod'
      if (newPathLength <= 1) newLevelPair = 'datacenter_pod'
      else if (newPathLength === 2) newLevelPair = 'pod_rack'
      else if (newPathLength >= 3) newLevelPair = 'rack_board'
      connection.setMultiLevelOptions({ ...connection.multiLevelOptions, levelPair: newLevelPair })
    }
    navigation.navigateBack()
  }, [connection, navigation])

  // 面包屑导航处理
  const handleBreadcrumbClick = useCallback((index: number) => {
    if (connection.multiLevelOptions.enabled) {
      let newLevelPair: 'datacenter_pod' | 'pod_rack' | 'rack_board' | 'board_chip' = 'datacenter_pod'
      if (index <= 1) newLevelPair = 'datacenter_pod'
      else if (index === 2) newLevelPair = 'pod_rack'
      else if (index === 3) newLevelPair = 'rack_board'
      connection.setMultiLevelOptions({ ...connection.multiLevelOptions, levelPair: newLevelPair })
    }
    navigation.navigateToBreadcrumb(index)
  }, [connection, navigation])

  // 节点点击处理
  const handleNodeClick = useCallback((node: NodeDetail | null) => {
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
  }, [ui])

  return (
    <Layout style={{ height: '100vh' }}>
      {/* Header */}
      <Header style={{
        background: '#FFFFFF',
        borderBottom: '1px solid #E5E5E5',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 56,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
        position: 'relative',
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg, #5E6AD2 0%, #7C3AED 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 6px rgba(94, 106, 210, 0.3)',
          }}>
            <span style={{ color: '#fff', fontSize: 13, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>T6+</span>
          </div>
          <Title level={4} style={{ color: '#1A1A1A', margin: 0, fontSize: 16, fontWeight: 600 }}>
            Tier6+ 互联建模
          </Title>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Segmented
            value={ui.viewMode}
            onChange={(v) => ui.setViewMode(v as '3d' | 'topology' | 'analysis' | 'knowledge')}
            options={[
              { value: '3d', label: '3D视图' },
              { value: 'topology', label: '拓扑图' },
              { value: 'analysis', label: '部署分析' },
              { value: 'knowledge', label: '知识网络' },
            ]}
          />
          <span style={{ color: '#999999', fontSize: 12 }}>v{__APP_VERSION__}</span>
        </div>
      </Header>

      <Layout>
        {/* Sider */}
        <Sider
          width={siderWidth}
          style={{
            background: '#EFEFEF',
            padding: 16,
            overflow: 'auto',
            position: 'relative',
            borderRight: '1px solid #E5E5E5',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            {/* 知识网络模式且选中节点时，隐藏配置面板 */}
            {ui.viewMode === 'knowledge' && ui.knowledgeSelectedNodes.length > 0 ? null : ui.viewMode === 'knowledge' ? (
              <Collapse
                defaultActiveKey={['config']}
                size="small"
                items={[{
                  key: 'config',
                  label: '配置面板',
                  children: (
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
                      viewMode={ui.viewMode}
                      focusedLevel={ui.focusedLevel}
                      onTrafficResultChange={analysis.setTrafficResult}
                      onAnalysisDataChange={analysis.setDeploymentAnalysisData}
                      analysisHistory={analysis.analysisHistory}
                      onAddToHistory={analysis.handleAddToHistory}
                      onDeleteHistory={analysis.handleDeleteHistory}
                      onClearHistory={analysis.handleClearHistory}
                    />
                  ),
                }]}
              />
            ) : (
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
                viewMode={ui.viewMode}
                focusedLevel={ui.focusedLevel}
                onTrafficResultChange={analysis.setTrafficResult}
                onAnalysisDataChange={analysis.setDeploymentAnalysisData}
                analysisHistory={analysis.analysisHistory}
                onAddToHistory={analysis.handleAddToHistory}
                onDeleteHistory={analysis.handleDeleteHistory}
                onClearHistory={analysis.handleClearHistory}
              />
            )}
          </div>

          {/* 知识图谱节点详情 - 支持多卡片堆叠 */}
          {ui.viewMode === 'knowledge' && ui.knowledgeSelectedNodes.length > 0 && (
            <KnowledgeNodeCards
              nodes={ui.knowledgeSelectedNodes}
              onClose={(nodeId) => ui.removeKnowledgeSelectedNode(nodeId)}
              onNodeClick={(node) => ui.addKnowledgeSelectedNode(node as ForceKnowledgeNode)}
            />
          )}

          {/* 节点详情卡片 */}
          {ui.viewMode !== 'knowledge' && ui.selectedNode && (
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
                    上行: {ui.selectedNode.portInfo.uplink} | 下行: {ui.selectedNode.portInfo.downlink} | 互联: {ui.selectedNode.portInfo.inter}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="连接数">{ui.selectedNode.connections.length}</Descriptions.Item>
              </Descriptions>
              {ui.selectedNode.connections.length > 0 && (
                <Collapse
                  size="small"
                  style={{ marginTop: 8 }}
                  items={[{
                    key: 'connections',
                    label: `连接列表 (${ui.selectedNode.connections.length})`,
                    children: (
                      <div style={{ maxHeight: 150, overflow: 'auto' }}>
                        {ui.selectedNode.connections.map((conn, idx) => (
                          <div key={idx} style={{ fontSize: 12, padding: '2px 0', borderBottom: '1px solid #f0f0f0' }}>
                            {conn.label}
                            {conn.bandwidth && <span style={{ color: '#999', marginLeft: 8 }}>{conn.bandwidth} GB/s</span>}
                          </div>
                        ))}
                      </div>
                    ),
                  }]}
                />
              )}
            </Card>
          )}

          {/* 连接详情卡片 */}
          {ui.viewMode !== 'knowledge' && ui.selectedLink && (
            <Card
              title="连接详情"
              size="small"
              style={{ marginTop: 16 }}
              extra={<a onClick={() => ui.setSelectedLink(null)}>关闭</a>}
            >
              <Descriptions column={1} size="small">
                <Descriptions.Item label="源节点">
                  <Tag color="green">{ui.selectedLink.sourceLabel}</Tag>
                  <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>({ui.selectedLink.sourceType.toUpperCase()})</span>
                </Descriptions.Item>
                <Descriptions.Item label="目标节点">
                  <Tag color="blue">{ui.selectedLink.targetLabel}</Tag>
                  <span style={{ color: '#999', marginLeft: 4, fontSize: 12 }}>({ui.selectedLink.targetType.toUpperCase()})</span>
                </Descriptions.Item>
                {ui.selectedLink.bandwidth && (
                  <Descriptions.Item label="带宽">{ui.selectedLink.bandwidth} GB/s</Descriptions.Item>
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
              position: 'absolute', top: 0, right: 0, width: 4, height: '100%',
              cursor: 'col-resize',
              background: isDragging ? '#4f46e5' : 'transparent',
              transition: 'background 0.15s',
              zIndex: 10,
            }}
            onMouseEnter={(e) => { if (!isDragging) (e.target as HTMLElement).style.background = '#e2e8f0' }}
            onMouseLeave={(e) => { if (!isDragging) (e.target as HTMLElement).style.background = 'transparent' }}
          />
        </Sider>

        {/* Content */}
        <Content style={{ position: 'relative', background: '#ffffff', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {topology.loading && !topology.topology ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <Spin size="large" tip="加载中..." />
            </div>
          ) : ui.viewMode === 'analysis' ? (
            <div style={{ flex: 1, overflow: 'auto', padding: 24, background: '#fafafa' }}>
              <div style={{ maxWidth: 1600, margin: '0 auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                  <div style={{ fontSize: 20, fontWeight: 600, color: '#1a1a1a' }}>LLM 部署分析结果</div>
                  {analysis.analysisViewMode === 'detail' && (
                    <Button type="primary" size="small" icon={<ArrowLeftOutlined />} onClick={() => analysis.setAnalysisViewMode('history')} style={{ fontSize: 13 }}>
                      历史记录
                    </Button>
                  )}
                </div>
                <AnalysisResultDisplay
                  result={analysis.deploymentAnalysisData?.result ?? null}
                  topKPlans={analysis.deploymentAnalysisData?.topKPlans ?? []}
                  loading={analysis.deploymentAnalysisData?.loading ?? false}
                  onSelectPlan={analysis.deploymentAnalysisData?.onSelectPlan}
                  searchStats={analysis.deploymentAnalysisData?.searchStats ?? null}
                  errorMsg={analysis.deploymentAnalysisData?.errorMsg ?? null}
                  viewMode={analysis.analysisViewMode}
                  onViewModeChange={analysis.setAnalysisViewMode}
                  history={analysis.analysisHistory}
                  onLoadFromHistory={analysis.handleLoadFromHistory}
                  onDeleteHistory={analysis.handleDeleteHistory}
                  onClearHistory={analysis.handleClearHistory}
                  canMapToTopology={analysis.deploymentAnalysisData?.canMapToTopology}
                  onMapToTopology={analysis.deploymentAnalysisData?.onMapToTopology}
                  onClearTraffic={analysis.deploymentAnalysisData?.onClearTraffic}
                  hardware={analysis.deploymentAnalysisData?.hardware}
                  model={analysis.deploymentAnalysisData?.model}
                  inference={analysis.deploymentAnalysisData?.inference}
                />
                {analysis.deploymentAnalysisData?.result && analysis.analysisViewMode === 'detail' && (
                  <ChartsPanel
                    result={analysis.deploymentAnalysisData.result}
                    topKPlans={analysis.deploymentAnalysisData.topKPlans}
                    hardware={analysis.deploymentAnalysisData.hardware}
                    model={analysis.deploymentAnalysisData.model}
                    inference={analysis.deploymentAnalysisData.inference}
                    topology={topology.topology}
                  />
                )}
              </div>
            </div>
          ) : ui.viewMode === 'knowledge' ? (
            <KnowledgeGraph />
          ) : ui.viewMode === '3d' ? (
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
            />
          ) : (
            <TopologyGraph
              visible={true}
              onClose={() => ui.setViewMode('3d')}
              topology={topology.topology}
              currentLevel={getCurrentLevel()}
              currentPod={navigation.currentPod}
              currentRack={navigation.currentRack}
              currentBoard={navigation.currentBoard}
              onNodeDoubleClick={handleNodeDoubleClick}
              onNodeClick={handleNodeClick}
              onLinkClick={(link) => { ui.setSelectedLink(link); if (link) ui.setSelectedNode(null) }}
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
              trafficResult={analysis.trafficResult}
            />
          )}
        </Content>
      </Layout>
    </Layout>
  )
}

/**
 * App 根组件 - 提供 Context
 */
const App: React.FC = () => {
  return (
    <WorkbenchProvider>
      <WorkbenchContent />
    </WorkbenchProvider>
  )
}

export default App
