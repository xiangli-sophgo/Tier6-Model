import React, { useState, useEffect, useRef } from 'react'
import {
  Typography,
  Button,
  Space,
  Card,
  Statistic,
  Row,
  Col,
  InputNumber,
  Modal,
  Input,
  message,
  Popconfirm,
  Switch,
  Tabs,
  Select,
  Tooltip,
} from 'antd'
import {
  ClusterOutlined,
  DatabaseOutlined,
  SaveOutlined,
  FolderOpenOutlined,
  DeleteOutlined,
  PlusOutlined,
  MinusCircleOutlined,
  ApartmentOutlined,
} from '@ant-design/icons'
import { GlobalSwitchConfig } from '../../types'
import { listConfigs, saveConfig, deleteConfig, SavedConfig } from '../../api/topology'
import { clearAllCache } from '../../utils/storage'
import {
  ChipIcon,
  BoardIcon,
  BoardConfigs,
  FlexBoardConfig,
  RackConfig,
  ConfigPanelProps,
  DEFAULT_BOARD_CONFIGS,
  DEFAULT_RACK_CONFIG,
  DEFAULT_SWITCH_CONFIG,
  loadCachedConfig,
  saveCachedConfig,
} from './shared'
import { SwitchLevelConfig, ConnectionEditPanel } from './components'
import { DeploymentAnalysisPanel } from './DeploymentAnalysis'
import { BaseCard } from '../common/BaseCard'
import { getChipList, getChipConfig, saveCustomChipPreset, deleteCustomChipPreset, getChipInterconnectConfig } from '../../utils/llmDeployment/presets'
import { ChipHardwareConfig } from '../../utils/llmDeployment/types'

const { Text } = Typography

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  topology,
  onGenerate,
  currentLevel = 'datacenter',
  // 手动连线相关
  manualConnectionConfig,
  onManualConnectionConfigChange,
  connectionMode = 'view',
  onConnectionModeChange,
  selectedNodes = new Set<string>(),
  onSelectedNodesChange,
  targetNodes = new Set<string>(),
  onTargetNodesChange,
  onBatchConnect,
  onDeleteManualConnection,
  currentViewConnections = [],
  onDeleteConnection,
  onUpdateConnectionParams,
  layoutType: _layoutType = 'auto',
  onLayoutTypeChange: _onLayoutTypeChange,
  viewMode = 'topology',
  focusedLevel,
  // 流量热力图
  onTrafficResultChange,
  // 部署分析结果
  onAnalysisDataChange,
  // 历史记录
  analysisHistory,
  onAddToHistory,
  onDeleteHistory,
  onClearHistory,
}) => {
  void _layoutType
  void _onLayoutTypeChange
  // 从缓存加载初始配置
  const cachedConfig = loadCachedConfig()

  // Pod层级配置
  const [podCount, setPodCount] = useState(cachedConfig?.podCount ?? 1)

  // Rack层级配置
  const [racksPerPod, setRacksPerPod] = useState(cachedConfig?.racksPerPod ?? 4)

  // Board配置（按U高度分类，每种类型有独立的chip配置）- 旧格式，保持兼容
  const [boardConfigs, setBoardConfigs] = useState<BoardConfigs>(
    cachedConfig?.boardConfigs ?? DEFAULT_BOARD_CONFIGS
  )

  // 新的灵活Rack配置
  const [rackConfig, setRackConfig] = useState<RackConfig>(
    cachedConfig?.rackConfig ?? DEFAULT_RACK_CONFIG
  )

  // Rack配置编辑模式
  const [rackEditMode, setRackEditMode] = useState(false)

  // Switch配置（深度合并默认值以兼容旧缓存）
  const [switchConfig, setSwitchConfig] = useState<GlobalSwitchConfig>(() => {
    if (cachedConfig?.switchConfig) {
      // 深度合并各层级配置，确保新字段有默认值
      const merged = { ...DEFAULT_SWITCH_CONFIG }
      if (cachedConfig.switchConfig.switch_types) {
        merged.switch_types = cachedConfig.switchConfig.switch_types
      }
      // 合并各层级配置，过滤掉无效字段
      const mergeLevel = (defaultLevel: any, cachedLevel: any) => {
        if (!cachedLevel) return defaultLevel
        return {
          enabled: cachedLevel.enabled ?? defaultLevel.enabled,
          layers: cachedLevel.layers ?? defaultLevel.layers,
          downlink_redundancy: cachedLevel.downlink_redundancy ?? defaultLevel.downlink_redundancy,
          connect_to_upper_level: cachedLevel.connect_to_upper_level ?? defaultLevel.connect_to_upper_level,
          direct_topology: cachedLevel.direct_topology ?? defaultLevel.direct_topology,
          keep_direct_topology: cachedLevel.keep_direct_topology ?? defaultLevel.keep_direct_topology,
          connection_mode: cachedLevel.connection_mode ?? defaultLevel.connection_mode,
          group_config: cachedLevel.group_config ?? defaultLevel.group_config,
          custom_connections: cachedLevel.custom_connections ?? defaultLevel.custom_connections,
          // 只保留正确的字段名
          switch_position: cachedLevel.switch_position ?? defaultLevel.switch_position,
          switch_u_height: cachedLevel.switch_u_height ?? defaultLevel.switch_u_height,
        }
      }
      merged.inter_pod = mergeLevel(DEFAULT_SWITCH_CONFIG.inter_pod, cachedConfig.switchConfig.inter_pod)
      merged.inter_rack = mergeLevel(DEFAULT_SWITCH_CONFIG.inter_rack, cachedConfig.switchConfig.inter_rack)
      merged.inter_board = mergeLevel(DEFAULT_SWITCH_CONFIG.inter_board, cachedConfig.switchConfig.inter_board)
      merged.inter_chip = mergeLevel(DEFAULT_SWITCH_CONFIG.inter_chip, cachedConfig.switchConfig.inter_chip)
      return merged
    }
    return DEFAULT_SWITCH_CONFIG
  })

  // 保存/加载配置状态
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([])
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [loadModalOpen, setLoadModalOpen] = useState(false)
  const [configName, setConfigName] = useState('')
  const [configDesc, setConfigDesc] = useState('')

  // 加载配置列表
  const loadConfigList = async () => {
    try {
      const configs = await listConfigs()
      setSavedConfigs(configs)
    } catch (error) {
      console.error('加载配置列表失败:', error)
    }
  }

  useEffect(() => {
    loadConfigList()
  }, [])

  // 配置变化时自动保存到localStorage
  useEffect(() => {
    saveCachedConfig({ podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig })
  }, [podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig])

  // 配置变化时自动生成拓扑（防抖500ms）
  const isFirstRender = useRef(true)
  useEffect(() => {
    // 跳过首次渲染（避免页面加载时重复生成）
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }

    const timer = setTimeout(() => {
      onGenerate({
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        board_configs: boardConfigs,
        rack_config: rackConfig,
        switch_config: switchConfig,
        manual_connections: manualConnectionConfig,
      })
    }, 500)

    return () => clearTimeout(timer)
  }, [podCount, racksPerPod, boardConfigs, rackConfig, switchConfig, manualConnectionConfig, onGenerate])

  // 保存当前配置
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      message.error('请输入配置名称')
      return
    }
    try {
      await saveConfig({
        name: configName.trim(),
        description: configDesc.trim() || undefined,
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        board_configs: boardConfigs,
      })
      message.success('配置保存成功')
      setSaveModalOpen(false)
      setConfigName('')
      setConfigDesc('')
      loadConfigList()
    } catch (error) {
      console.error('保存配置失败:', error)
      message.error('保存配置失败')
    }
  }

  // 加载指定配置
  const handleLoadConfig = (config: SavedConfig) => {
    setPodCount(config.pod_count)
    setRacksPerPod(config.racks_per_pod)
    setBoardConfigs(config.board_configs)
    setLoadModalOpen(false)
    message.success(`已加载配置: ${config.name}`)
  }

  // 删除配置
  const handleDeleteConfig = async (name: string) => {
    try {
      await deleteConfig(name)
      message.success('配置已删除')
      loadConfigList()
    } catch (error) {
      console.error('删除配置失败:', error)
      message.error('删除配置失败')
    }
  }

  // 计算统计数据
  const stats = {
    pods: topology?.pods.length || 0,
    racks: topology?.pods.reduce((sum, p) => sum + p.racks.length, 0) || 0,
    boards: topology?.pods.reduce((sum, p) =>
      sum + p.racks.reduce((s, r) => s + r.boards.length, 0), 0) || 0,
    chips: topology?.pods.reduce((sum, p) =>
      sum + p.racks.reduce((s, r) =>
        s + r.boards.reduce((b, board) => b + board.chips.length, 0), 0), 0) || 0,
    switches: topology?.switches?.length || 0,
  }

  // 配置项样式
  const configRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  }

  // 根据芯片配置更新连接参数（层级默认参数和手动连接）
  // 注意：不直接更新当前连接，因为拓扑重新生成时会使用层级默认参数
  const updateConnectionDefaultsFromChips = React.useCallback((boards: typeof rackConfig.boards) => {
    // 收集所有芯片的互联配置，找到数量最多的芯片类型
    let maxCount = 0
    let primaryInterconnect: ReturnType<typeof getChipInterconnectConfig> = null

    for (const board of boards) {
      const boardCount = board.count || 1
      for (const chip of board.chips) {
        const totalChips = chip.count * boardCount
        if (chip.preset_id && totalChips > maxCount) {
          const interconnect = getChipInterconnectConfig(chip.preset_id)
          if (interconnect) {
            maxCount = totalChips
            primaryInterconnect = interconnect
          }
        }
      }
    }

    if (!primaryInterconnect) return

    const newBandwidth = primaryInterconnect.intra_node_bandwidth_gbps
    const newLatency = primaryInterconnect.intra_node_latency_us // us

    // 更新层级默认参数和手动连接
    if (onManualConnectionConfigChange) {
      const existingConnections = manualConnectionConfig?.connections || []
      // 更新 Board 层的手动连接参数
      const updatedConnections = existingConnections.map(conn => {
        if (conn.hierarchy_level === 'board') {
          return { ...conn, bandwidth: newBandwidth, latency: newLatency }
        }
        return conn
      })

      const newConfig = {
        ...(manualConnectionConfig || { enabled: true, mode: 'append' as const, connections: [] }),
        connections: updatedConnections,
        level_defaults: {
          ...(manualConnectionConfig?.level_defaults || {}),
          board: {
            ...(manualConnectionConfig?.level_defaults?.board || {}),
            bandwidth: newBandwidth,
            latency: newLatency,
          },
        },
      }
      onManualConnectionConfigChange(newConfig)
    }

    // 提示用户
    message.info(`已根据 ${primaryInterconnect.interconnect_type} 更新 Board 层连接参数: ${newBandwidth} GB/s, ${newLatency} us`)
  }, [manualConnectionConfig, onManualConnectionConfigChange])

  // 层级配置Tab key
  const [layerTabKey, setLayerTabKey] = useState<string>(currentLevel === 'datacenter' ? 'datacenter' : currentLevel)

  // 当右边层级变化时，同步层级配置Tab
  useEffect(() => {
    setLayerTabKey(currentLevel === 'datacenter' ? 'datacenter' : currentLevel)
  }, [currentLevel])

  // 外部指定聚焦层级时切换Tab（点击容器时）
  useEffect(() => {
    if (focusedLevel) {
      setLayerTabKey(focusedLevel)
    }
  }, [focusedLevel])

  // 汇总信息
  const summaryText = topology
    ? `${stats.pods}Pod ${stats.racks}Rack ${stats.boards}Board ${stats.chips}Chip`
    : '未生成'

  // 拓扑配置内容（统计信息）
  const topologyConfigContent = (
    <Row gutter={[8, 8]}>
      <Col span={12}>
        <Statistic
          title="Pods"
          value={stats.pods}
          prefix={<ClusterOutlined />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Racks"
          value={stats.racks}
          prefix={<DatabaseOutlined />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Boards"
          value={stats.boards}
          prefix={<BoardIcon />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      <Col span={12}>
        <Statistic
          title="Chips"
          value={stats.chips}
          prefix={<ChipIcon />}
          valueStyle={{ fontSize: 16 }}
        />
      </Col>
      {stats.switches > 0 && (
        <Col span={24}>
          <Statistic
            title="Switches"
            value={stats.switches}
            prefix={<ApartmentOutlined />}
            valueStyle={{ fontSize: 16 }}
          />
        </Col>
      )}
    </Row>
  )


  // 层级配置内容（节点配置 + Switch连接配置）
  const layerConfigContent = (
    <Tabs
      size="small"
      type="card"
      activeKey={layerTabKey}
      onChange={setLayerTabKey}
      items={[
        {
          key: 'datacenter',
          label: '数据中心层',
          children: (
            <div>
              {/* Pod数量配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>节点配置</Text>
                <div style={configRowStyle}>
                  <Text>Pod 数量</Text>
                  <InputNumber
                    min={1}
                    max={10}
                    value={podCount}
                    onChange={(v) => setPodCount(v || 1)}
                    size="small"
                    style={{ width: 80 }}
                  />
                </div>
              </div>
              {/* Pod间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="inter_pod"
                  config={switchConfig.inter_pod}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_pod: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级或聚焦层级时显示） */}
              {(currentLevel === 'datacenter' || focusedLevel === 'datacenter') && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    onUpdateConnectionParams={onUpdateConnectionParams}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'pod',
          label: 'Pod层',
          children: (
            <div>
              {/* Rack数量配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>节点配置</Text>
                <div style={configRowStyle}>
                  <Text>每Pod机柜数</Text>
                  <InputNumber
                    min={1}
                    max={64}
                    value={racksPerPod}
                    onChange={(v) => setRacksPerPod(v || 1)}
                    size="small"
                    style={{ width: 80 }}
                  />
                </div>
              </div>
              {/* Rack间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="inter_rack"
                  config={switchConfig.inter_rack}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_rack: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级或聚焦层级时显示） */}
              {(currentLevel === 'pod' || focusedLevel === 'pod') && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    onUpdateConnectionParams={onUpdateConnectionParams}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'rack',
          label: 'Rack层',
          children: (
            <div>
              {/* Board配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: 'linear-gradient(135deg, rgba(248, 250, 252, 0.8) 0%, rgba(241, 245, 249, 0.8) 100%)',
                borderRadius: 12,
                border: '1px solid rgba(0, 0, 0, 0.04)',
              }}>
                {/* 标题和编辑开关 */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <Text strong>节点配置</Text>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Text type="secondary" style={{ fontSize: 11 }}>编辑</Text>
                    <Switch
                      size="small"
                      checked={rackEditMode}
                      onChange={setRackEditMode}
                    />
                  </div>
                </div>

                {/* 汇总信息 */}
                {(() => {
                  const usedU = rackConfig.boards.reduce((sum, b) => sum + b.u_height * (b.count || 1), 0)
                  const totalBoards = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1), 0)
                  const totalChips = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1) * b.chips.reduce((s, c) => s + c.count, 0), 0)
                  const isOverflow = usedU > rackConfig.total_u
                  return (
                    <div style={{ marginBottom: 8, fontSize: 12, color: '#666' }}>
                      <span>容量: <Text strong>{rackConfig.total_u}U</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>已用: <Text strong type={isOverflow ? 'danger' : undefined}>{usedU}U</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>板卡: <Text strong>{totalBoards}</Text></span>
                      <span style={{ margin: '0 8px', color: '#d9d9d9' }}>|</span>
                      <span>芯片: <Text strong>{totalChips}</Text></span>
                    </div>
                  )
                })()}

                {/* 编辑模式：Rack容量 */}
                {rackEditMode && (
                  <div style={configRowStyle}>
                    <Text>Rack容量</Text>
                    <InputNumber
                      min={10}
                      max={60}
                      value={rackConfig.total_u}
                      onChange={(v) => setRackConfig(prev => ({ ...prev, total_u: v || 42 }))}
                      size="small"
                      style={{ width: 70 }}
                      suffix="U"
                    />
                  </div>
                )}

                {/* 板卡列表 */}
                <div style={{ marginTop: 8 }}>
                  {rackConfig.boards.map((board, boardIndex) => (
                    <div key={board.id} style={{ marginBottom: 6, padding: '6px 10px', background: '#fff', borderRadius: 8, border: '1px solid rgba(0,0,0,0.06)' }}>
                      {rackEditMode ? (
                        /* 编辑模式 */
                        <>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Text style={{ fontSize: 12, whiteSpace: 'nowrap' }}>名称:</Text>
                              <Input
                                size="small"
                                value={board.name}
                                onChange={(e) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], name: e.target.value }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 120 }}
                              />
                              <Text style={{ fontSize: 12, marginLeft: 8, whiteSpace: 'nowrap' }}>高度:</Text>
                              <InputNumber
                                size="small"
                                min={1}
                                max={10}
                                value={board.u_height}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], u_height: v || 1 }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 70 }}
                                suffix="U"
                              />
                              <Text style={{ fontSize: 12, marginLeft: 8, whiteSpace: 'nowrap' }}>数量:</Text>
                              <InputNumber
                                size="small"
                                min={0}
                                max={42}
                                value={board.count || 1}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], count: v || 0 }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ width: 60 }}
                              />
                            </div>
                            <Button
                              type="text"
                              danger
                              size="small"
                              icon={<MinusCircleOutlined />}
                              onClick={() => {
                                const newBoards = rackConfig.boards.filter((_, i) => i !== boardIndex)
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              disabled={rackConfig.boards.length <= 1}
                            />
                          </div>
                        </>
                      ) : (
                        /* 展示模式 */
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text style={{ fontSize: 13 }}>{board.name} ×{board.count || 1}</Text>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>{board.u_height}U</Text>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {board.chips.map(c => `${c.name}×${c.count}`).join(' ')}
                            </Text>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* 编辑模式：添加板卡按钮 */}
                {rackEditMode && (
                  <Button
                    type="dashed"
                    size="small"
                    icon={<PlusOutlined />}
                    onClick={() => {
                      const newBoard: FlexBoardConfig = {
                        id: `board_${Date.now()}`,
                        name: 'Board',
                        u_height: 2,
                        count: 1,
                        chips: [{ name: 'Chip', count: 8 }],
                      }
                      setRackConfig(prev => ({ ...prev, boards: [...prev.boards, newBoard] }))
                    }}
                    style={{ width: '100%', marginTop: 4 }}
                  >
                    添加板卡类型
                  </Button>
                )}
              </div>

              {/* Board间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="inter_board"
                  config={switchConfig.inter_board}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_board: newConfig }))}
                  configRowStyle={configRowStyle}
                  viewMode={viewMode}
                />
              </div>
              {/* 连接编辑（当前层级或聚焦层级时显示） */}
              {(currentLevel === 'rack' || focusedLevel === 'rack') && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    onUpdateConnectionParams={onUpdateConnectionParams}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
        {
          key: 'board',
          label: 'Board层',
          children: (
            <div>
              {/* 芯片配置 */}
              <div style={{
                marginBottom: 12,
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>芯片配置</Text>
                <Text type="secondary" style={{ fontSize: 11, marginBottom: 10, display: 'block' }}>
                  为每种板卡类型配置芯片
                </Text>
                {rackConfig.boards.map((board, boardIndex) => (
                  <div key={board.id} style={{ marginBottom: 10, padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e8e8e8' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <Text strong style={{ fontSize: 12 }}>{board.name}</Text>
                      <Button
                        type="dashed"
                        size="small"
                        icon={<PlusOutlined />}
                        onClick={() => {
                          const newBoards = [...rackConfig.boards]
                          const newChips = [...newBoards[boardIndex].chips, { name: 'H100-SXM', count: 8, preset_id: 'h100-sxm' }]
                          newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                          setRackConfig(prev => ({ ...prev, boards: newBoards }))
                        }}
                      >
                        添加芯片
                      </Button>
                    </div>
                    {board.chips.map((chip, chipIndex) => {
                      const chipPresetList = getChipList()
                      const presetConfig = chip.preset_id ? getChipConfig(chip.preset_id) : null
                      // 当前使用的参数值（预设值或自定义值）
                      const currentTflops = chip.compute_tflops_fp16 ?? presetConfig?.compute_tflops_fp16 ?? 100
                      const currentMemory = chip.memory_gb ?? presetConfig?.memory_gb ?? 32
                      const currentBandwidth = chip.memory_bandwidth_gbps ?? presetConfig?.memory_bandwidth_gbps ?? 1000
                      const currentBwUtil = chip.memory_bandwidth_utilization ?? presetConfig?.memory_bandwidth_utilization ?? 0.9
                      // 检查参数是否被修改过
                      const isModified = presetConfig && (
                        (chip.compute_tflops_fp16 !== undefined && chip.compute_tflops_fp16 !== presetConfig.compute_tflops_fp16) ||
                        (chip.memory_gb !== undefined && chip.memory_gb !== presetConfig.memory_gb) ||
                        (chip.memory_bandwidth_gbps !== undefined && chip.memory_bandwidth_gbps !== presetConfig.memory_bandwidth_gbps) ||
                        (chip.memory_bandwidth_utilization !== undefined && chip.memory_bandwidth_utilization !== presetConfig.memory_bandwidth_utilization)
                      )
                      const isCustomPreset = chipPresetList.find(c => c.id === chip.preset_id)?.isCustom
                      return (
                        <div key={chipIndex} style={{ marginBottom: 8, padding: '8px 10px', background: '#fafafa', borderRadius: 6, border: isModified ? '1px solid #faad14' : '1px solid transparent' }}>
                          {/* 类型选择 */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                            <Text style={{ fontSize: 12, width: 60, flexShrink: 0 }}>类型:</Text>
                            <Select
                              size="small"
                              value={chip.preset_id || 'custom'}
                              onChange={(value) => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = [...newBoards[boardIndex].chips]
                                if (value === 'custom') {
                                  newChips[chipIndex] = {
                                    ...newChips[chipIndex],
                                    name: '自定义芯片',
                                    preset_id: undefined,
                                    compute_tflops_fp16: 100,
                                    memory_gb: 32,
                                    memory_bandwidth_gbps: 1000,
                                  }
                                } else {
                                  const preset = getChipConfig(value)
                                  if (preset) {
                                    newChips[chipIndex] = {
                                      ...newChips[chipIndex],
                                      name: preset.chip_type,
                                      preset_id: value,
                                      compute_tflops_fp16: undefined,
                                      memory_gb: undefined,
                                      memory_bandwidth_gbps: undefined,
                                    }
                                  }
                                }
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                // 根据新选择的芯片类型更新连接默认参数
                                updateConnectionDefaultsFromChips(newBoards)
                              }}
                              style={{ flex: 1 }}
                              options={[
                                ...chipPresetList.map(c => ({
                                  value: c.id,
                                  label: c.name,
                                })),
                                { value: 'custom', label: '自定义...' },
                              ]}
                            />
                            <Button
                              type="text"
                              danger
                              size="small"
                              icon={<MinusCircleOutlined />}
                              onClick={() => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = newBoards[boardIndex].chips.filter((_, i) => i !== chipIndex)
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              disabled={board.chips.length <= 1}
                            />
                          </div>
                          {/* 自定义类型时显示名称输入 */}
                          {!chip.preset_id && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                              <Text style={{ fontSize: 12, width: 60, flexShrink: 0 }}>名称:</Text>
                              <Input
                                size="small"
                                placeholder="芯片名称"
                                value={chip.name}
                                onChange={(e) => {
                                  const newBoards = [...rackConfig.boards]
                                  const newChips = [...newBoards[boardIndex].chips]
                                  newChips[chipIndex] = { ...newChips[chipIndex], name: e.target.value }
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ flex: 1 }}
                              />
                            </div>
                          )}
                          {/* 数量 */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                            <Text style={{ fontSize: 12, width: 60, flexShrink: 0 }}>数量:</Text>
                            <InputNumber
                              size="small"
                              min={1}
                              max={64}
                              value={chip.count}
                              onChange={(v) => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = [...newBoards[boardIndex].chips]
                                newChips[chipIndex] = { ...newChips[chipIndex], count: v || 1 }
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              style={{ flex: 1 }}
                              addonAfter="个"
                            />
                          </div>
                          {/* 第二行：芯片参数（可编辑） */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Tooltip title="FP16 精度的理论峰值算力">
                                <Text style={{ fontSize: 12, width: 60, flexShrink: 0, cursor: 'help' }}>算力:</Text>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={1}
                                value={currentTflops}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  const newChips = [...newBoards[boardIndex].chips]
                                  newChips[chipIndex] = { ...newChips[chipIndex], compute_tflops_fp16: v || undefined }
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ flex: 1 }}
                                addonAfter="TFLOPs"
                              />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Tooltip title="DRAM 存储容量">
                                <Text style={{ fontSize: 12, width: 60, flexShrink: 0, cursor: 'help' }}>显存:</Text>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={1}
                                value={currentMemory}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  const newChips = [...newBoards[boardIndex].chips]
                                  newChips[chipIndex] = { ...newChips[chipIndex], memory_gb: v || undefined }
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ flex: 1 }}
                                addonAfter="GB"
                              />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Tooltip title="DRAM 理论带宽">
                                <Text style={{ fontSize: 12, width: 60, flexShrink: 0, cursor: 'help' }}>带宽:</Text>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={1}
                                value={currentBandwidth}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  const newChips = [...newBoards[boardIndex].chips]
                                  newChips[chipIndex] = { ...newChips[chipIndex], memory_bandwidth_gbps: v || undefined }
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ flex: 1 }}
                                addonAfter="GB/s"
                              />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <Tooltip title="显存带宽的实际利用率，通常为0.8-0.95">
                                <Text style={{ fontSize: 12, width: 70, flexShrink: 0, cursor: 'help' }}>带宽利用率:</Text>
                              </Tooltip>
                              <InputNumber
                                size="small"
                                min={0.1}
                                max={1}
                                step={0.01}
                                value={currentBwUtil}
                                onChange={(v) => {
                                  const newBoards = [...rackConfig.boards]
                                  const newChips = [...newBoards[boardIndex].chips]
                                  newChips[chipIndex] = { ...newChips[chipIndex], memory_bandwidth_utilization: v || undefined }
                                  newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                  setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                }}
                                style={{ flex: 1 }}
                                formatter={(value) => value ? `${(value * 100).toFixed(0)}%` : ''}
                                parser={(value) => value ? parseFloat(value.replace('%', '')) / 100 : 0.9}
                              />
                            </div>
                          </div>
                          {/* 第三行：操作按钮 */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                            {isModified && (
                              <Tooltip title="重置为预设值">
                                <Button
                                  type="link"
                                  size="small"
                                  style={{ padding: 0, fontSize: 11 }}
                                  onClick={() => {
                                    const newBoards = [...rackConfig.boards]
                                    const newChips = [...newBoards[boardIndex].chips]
                                    newChips[chipIndex] = {
                                      ...newChips[chipIndex],
                                      compute_tflops_fp16: undefined,
                                      memory_gb: undefined,
                                      memory_bandwidth_gbps: undefined,
                                      memory_bandwidth_utilization: undefined,
                                    }
                                    newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                    setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                  }}
                                >
                                  重置
                                </Button>
                              </Tooltip>
                            )}
                            {(isModified || !chip.preset_id) && (
                              <Tooltip title="保存为新预设">
                                <Button
                                  type="link"
                                  size="small"
                                  icon={<SaveOutlined />}
                                  style={{ padding: 0, fontSize: 11 }}
                                  onClick={() => {
                                    const newName = prompt('输入预设名称:', chip.name || '自定义芯片')
                                    if (newName) {
                                      const presetId = `custom-${Date.now()}`
                                      const config: ChipHardwareConfig = {
                                        chip_type: newName,
                                        compute_tflops_fp16: currentTflops,
                                        memory_gb: currentMemory,
                                        memory_bandwidth_gbps: currentBandwidth,
                                        memory_bandwidth_utilization: currentBwUtil,
                                      }
                                      saveCustomChipPreset(presetId, config)
                                      // 更新当前芯片使用新预设
                                      const newBoards = [...rackConfig.boards]
                                      const newChips = [...newBoards[boardIndex].chips]
                                      newChips[chipIndex] = {
                                        ...newChips[chipIndex],
                                        name: newName,
                                        preset_id: presetId,
                                        compute_tflops_fp16: undefined,
                                        memory_gb: undefined,
                                        memory_bandwidth_gbps: undefined,
                                        memory_bandwidth_utilization: undefined,
                                      }
                                      newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                      setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                      message.success(`已保存预设: ${newName}`)
                                    }
                                  }}
                                >
                                  保存预设
                                </Button>
                              </Tooltip>
                            )}
                            {isCustomPreset && chip.preset_id && (
                              <Popconfirm
                                title="确定删除此预设？"
                                onConfirm={() => {
                                  if (chip.preset_id) {
                                    deleteCustomChipPreset(chip.preset_id)
                                    // 将当前芯片改为自定义
                                    const newBoards = [...rackConfig.boards]
                                    const newChips = [...newBoards[boardIndex].chips]
                                    newChips[chipIndex] = {
                                      ...newChips[chipIndex],
                                      preset_id: undefined,
                                      compute_tflops_fp16: currentTflops,
                                      memory_gb: currentMemory,
                                      memory_bandwidth_gbps: currentBandwidth,
                                      memory_bandwidth_utilization: currentBwUtil,
                                    }
                                    newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                    setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                    message.success('已删除预设')
                                  }
                                }}
                              >
                                <Button
                                  type="link"
                                  danger
                                  size="small"
                                  icon={<DeleteOutlined />}
                                  style={{ padding: 0, fontSize: 11 }}
                                >
                                  删除预设
                                </Button>
                              </Popconfirm>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ))}
              </div>

              {/* Chip间连接配置 */}
              <div style={{
                padding: 14,
                background: '#f5f5f5',
                borderRadius: 10,
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接配置</Text>
                <SwitchLevelConfig
                  levelKey="inter_chip"
                  config={switchConfig.inter_chip}
                  switchTypes={switchConfig.switch_types}
                  onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_chip: newConfig }))}
                  configRowStyle={configRowStyle}
                />
              </div>
              {/* 连接编辑（当前层级或聚焦层级时显示） */}
              {(currentLevel === 'board' || focusedLevel === 'board') && (
                <div style={{ marginTop: 12 }}>
                  <ConnectionEditPanel
                    manualConnectionConfig={manualConnectionConfig}
                    onManualConnectionConfigChange={onManualConnectionConfigChange}
                    connectionMode={connectionMode}
                    onConnectionModeChange={onConnectionModeChange}
                    selectedNodes={selectedNodes}
                    onSelectedNodesChange={onSelectedNodesChange}
                    targetNodes={targetNodes}
                    onTargetNodesChange={onTargetNodesChange}
                    onBatchConnect={onBatchConnect}
                    onDeleteManualConnection={onDeleteManualConnection}
                    currentViewConnections={currentViewConnections}
                    onDeleteConnection={onDeleteConnection}
                    onUpdateConnectionParams={onUpdateConnectionParams}
                    configRowStyle={configRowStyle}
                    currentLevel={currentLevel}
                  />
                </div>
              )}
            </div>
          ),
        },
      ]}
    />
  )

  // Switch配置内容（只有Switch类型定义）
  const switchConfigContent = (
    <div>
      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 8 }}>
        定义可用的Switch型号，在各层级的连接配置中使用
      </Text>
      {switchConfig.switch_types.map((swType, index) => (
        <div key={swType.id} style={{ marginBottom: 8, padding: 8, background: '#f5f5f5', borderRadius: 8 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <Input
              size="small"
              placeholder="名称"
              value={swType.name}
              onChange={(e) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], name: e.target.value }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              style={{ flex: 1 }}
            />
            <InputNumber
              size="small"
              min={8}
              max={1024}
              controls={false}
              value={swType.port_count}
              onChange={(v) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], port_count: v || 48 }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              style={{ width: 80 }}
            />
            <Text style={{ fontSize: 12, color: '#666' }}>端口</Text>
            <Button
              type="text"
              danger
              size="small"
              icon={<MinusCircleOutlined />}
              disabled={switchConfig.switch_types.length <= 1}
              onClick={() => {
                const newTypes = switchConfig.switch_types.filter((_, i) => i !== index)
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
            />
          </div>
        </div>
      ))}
      <Button
        type="dashed"
        size="small"
        icon={<PlusOutlined />}
        onClick={() => {
          const newId = `switch_${Date.now()}`
          const newTypes = [...switchConfig.switch_types, { id: newId, name: '新Switch', port_count: 48 }]
          setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
        }}
        style={{ width: '100%' }}
      >
        添加Switch类型
      </Button>
    </div>
  )

  // 顶层页面Tab状态
  const [activePageTab, setActivePageTab] = useState<'topology' | 'deployment'>('topology')

  // 自定义Tab样式
  const tabButtonStyle = (isActive: boolean): React.CSSProperties => ({
    flex: 1,
    padding: '12px 16px',
    border: 'none',
    background: isActive ? '#fff' : 'transparent',
    color: isActive ? '#5E6AD2' : '#666',
    fontWeight: isActive ? 600 : 400,
    fontSize: 14,
    cursor: 'pointer',
    borderRadius: isActive ? '8px' : '8px',
    boxShadow: isActive ? '0 2px 8px rgba(94, 106, 210, 0.15)' : 'none',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {/* 顶层页面切换 - 自定义样式 */}
      <div style={{
        display: 'flex',
        gap: 8,
        padding: 4,
        background: '#E8E8E8',
        borderRadius: 10,
        marginBottom: 16,
      }}>
        <button
          style={tabButtonStyle(activePageTab === 'topology')}
          onClick={() => setActivePageTab('topology')}
        >
          拓扑设置
        </button>
        <button
          style={tabButtonStyle(activePageTab === 'deployment')}
          onClick={() => setActivePageTab('deployment')}
        >
          部署设置
        </button>
      </div>

      {/* 内容区域 - 使用 display 控制显示，避免组件卸载导致状态丢失 */}
      <div style={{ display: activePageTab === 'topology' ? 'block' : 'none' }}>
        <>
          {/* 拓扑汇总 */}
          <div style={{ marginBottom: 12 }}>
            <BaseCard
              title={<>拓扑汇总 <span style={{ fontSize: 12, fontWeight: 400, color: '#9ca3af', marginLeft: 8 }}>{summaryText}</span></>}
              accentColor="#5E6AD2"
              collapsible
              defaultExpanded={false}
            >
              {topologyConfigContent}
            </BaseCard>
          </div>

          {/* 层级配置 */}
          <div style={{ marginBottom: 12 }}>
            <BaseCard
              title="层级配置"
              accentColor="#13c2c2"
              collapsible
              defaultExpanded
            >
              {layerConfigContent}
            </BaseCard>
          </div>

          {/* Switch配置 */}
          <div style={{ marginBottom: 12 }}>
            <BaseCard
              title="Switch配置"
              accentColor="#52c41a"
              collapsible
              defaultExpanded={false}
            >
              {switchConfigContent}
            </BaseCard>
          </div>
          <style>{`
            .custom-collapse.ant-collapse {
              background: transparent !important;
              border: none !important;
            }
            .custom-collapse .ant-collapse-item {
              background: #fff !important;
              border-radius: 10px !important;
              margin-bottom: 12px !important;
              border: 1px solid #E5E5E5 !important;
              overflow: hidden;
            }
            .custom-collapse .ant-collapse-item:last-child {
              border-radius: 10px !important;
            }
            .custom-collapse .ant-collapse-header {
              padding: 12px 16px !important;
              font-weight: 500;
              background: #fff !important;
            }
            .custom-collapse .ant-collapse-content {
              border-top: 1px solid #F0F0F0 !important;
              background: #fff !important;
            }
            .custom-collapse .ant-collapse-content-box {
              padding: 12px 16px !important;
            }
          `}</style>
          {/* 保存/加载/清除配置按钮 */}
          <Row gutter={8} style={{ marginTop: 16 }}>
            <Col span={8}>
              <Popconfirm
                title="清除所有缓存"
                description="确定要清除所有缓存数据吗？清除后页面将刷新。"
                onConfirm={async () => {
                  try {
                    await clearAllCache()
                    message.success('缓存已清除，即将刷新页面')
                    setTimeout(() => window.location.reload(), 500)
                  } catch (error) {
                    message.error('清除缓存失败')
                  }
                }}
                okText="确定"
                cancelText="取消"
              >
                <Button block icon={<DeleteOutlined />} danger>
                  清除缓存
                </Button>
              </Popconfirm>
            </Col>
            <Col span={8}>
              <Button
                block
                icon={<SaveOutlined />}
                onClick={() => setSaveModalOpen(true)}
              >
                保存配置
              </Button>
            </Col>
            <Col span={8}>
              <Button
                block
                icon={<FolderOpenOutlined />}
                onClick={() => {
                  loadConfigList()
                  setLoadModalOpen(true)
                }}
              >
                加载配置
              </Button>
            </Col>
          </Row>
        </>
      </div>
      <div style={{ display: activePageTab === 'deployment' ? 'block' : 'none' }}>
        <DeploymentAnalysisPanel
          topology={topology}
          onTrafficResultChange={onTrafficResultChange}
          onAnalysisDataChange={onAnalysisDataChange}
          rackConfig={rackConfig}
          podCount={podCount}
          racksPerPod={racksPerPod}
          history={analysisHistory}
          onAddToHistory={onAddToHistory}
          onDeleteHistory={onDeleteHistory}
          onClearHistory={onClearHistory}
        />
      </div>

      {/* 保存配置模态框 */}
      <Modal
        title="保存配置"
        open={saveModalOpen}
        onOk={handleSaveConfig}
        onCancel={() => {
          setSaveModalOpen(false)
          setConfigName('')
          setConfigDesc('')
        }}
        okText="保存"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text>配置名称 *</Text>
            <Input
              placeholder="输入配置名称"
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              style={{ marginTop: 4 }}
            />
          </div>
          <div>
            <Text>描述 (可选)</Text>
            <Input.TextArea
              placeholder="输入配置描述"
              value={configDesc}
              onChange={(e) => setConfigDesc(e.target.value)}
              rows={2}
              style={{ marginTop: 4 }}
            />
          </div>
          {savedConfigs.some(c => c.name === configName.trim()) && (
            <Text type="warning" style={{ fontSize: 12 }}>
              同名配置已存在，保存将覆盖原配置
            </Text>
          )}
        </Space>
      </Modal>

      {/* 加载配置模态框 */}
      <Modal
        title="加载配置"
        open={loadModalOpen}
        onCancel={() => setLoadModalOpen(false)}
        footer={null}
        width={480}
      >
        {savedConfigs.length === 0 ? (
          <Text type="secondary">暂无保存的配置</Text>
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            {savedConfigs.map(config => (
              <Card
                key={config.name}
                size="small"
                style={{ cursor: 'pointer' }}
                hoverable
                onClick={() => handleLoadConfig(config)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <Text strong>{config.name}</Text>
                    {config.description && (
                      <div><Text type="secondary" style={{ fontSize: 12 }}>{config.description}</Text></div>
                    )}
                    <div style={{ marginTop: 4 }}>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        Pod:{config.pod_count} | Rack:{config.racks_per_pod} |
                        1U:{config.board_configs.u1.count} 2U:{config.board_configs.u2.count} 4U:{config.board_configs.u4.count}
                      </Text>
                    </div>
                  </div>
                  <Popconfirm
                    title="确定删除此配置？"
                    onConfirm={(e) => {
                      e?.stopPropagation()
                      handleDeleteConfig(config.name)
                    }}
                    onCancel={(e) => e?.stopPropagation()}
                    okText="删除"
                    cancelText="取消"
                  >
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>
                </div>
              </Card>
            ))}
          </Space>
        )}
      </Modal>
    </div>
  )
}
