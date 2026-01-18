import React from 'react'
import {
  Typography,
  Button,
  Space,
  InputNumber,
  Collapse,
  Input,
  Switch,
  Select,
  Checkbox,
  Divider,
  Radio,
} from 'antd'
import {
  DeleteOutlined,
  PlusOutlined,
  MinusCircleOutlined,
  UndoOutlined,
} from '@ant-design/icons'
import {
  HierarchyLevelSwitchConfig, SwitchTypeConfig, SwitchLayerConfig,
  ManualConnectionConfig, ConnectionMode, SwitchConnectionMode, HierarchyLevel,
  LevelConnectionDefaults,
} from '../../types'

const { Text } = Typography

// ============================================
// Switch层级配置子组件
// ============================================

interface SwitchLevelConfigProps {
  levelKey: string
  config: HierarchyLevelSwitchConfig
  switchTypes: SwitchTypeConfig[]
  onChange: (config: HierarchyLevelSwitchConfig) => void
  configRowStyle: React.CSSProperties
  viewMode?: '3d' | 'topology' | 'analysis' | 'knowledge'
}

export const SwitchLevelConfig: React.FC<SwitchLevelConfigProps> = ({
  levelKey,
  config,
  switchTypes,
  onChange,
  configRowStyle,
  viewMode,
}) => {
  const updateLayer = (index: number, field: keyof SwitchLayerConfig, value: any) => {
    const newLayers = [...config.layers]
    newLayers[index] = { ...newLayers[index], [field]: value }
    onChange({ ...config, layers: newLayers })
  }

  const addLayer = () => {
    const newLayer: SwitchLayerConfig = {
      layer_name: config.layers.length === 0 ? 'leaf' : 'spine',
      switch_type_id: switchTypes[0]?.id || '',
      count: 2,
      inter_connect: false,
    }
    onChange({ ...config, layers: [...config.layers, newLayer] })
  }

  const removeLayer = (index: number) => {
    const newLayers = config.layers.filter((_, i) => i !== index)
    onChange({ ...config, layers: newLayers })
  }

  // 直连拓扑类型选项
  const directTopologyOptions = [
    { value: 'none', label: '无连接' },
    { value: 'full_mesh', label: '全连接 (Full Mesh)' },
    { value: 'full_mesh_2d', label: '2D FullMesh (行列全连接)' },
    { value: 'ring', label: '环形 (Ring)' },
    { value: 'torus_2d', label: '2D Torus' },
    { value: 'torus_3d', label: '3D Torus' },
  ]

  return (
    <div>
      {/* 启用开关 */}
      <div style={configRowStyle}>
        <Text>启用Switch</Text>
        <Switch
          size="small"
          checked={config.enabled}
          onChange={(checked) => {
            if (checked && config.layers.length === 0) {
              // 启用时如果没有层，自动添加一个默认Switch层
              const defaultLayer: SwitchLayerConfig = {
                layer_name: 'leaf',
                switch_type_id: switchTypes[0]?.id || '',
                count: 1,
                inter_connect: false,
              }
              onChange({ ...config, enabled: checked, layers: [defaultLayer] })
            } else {
              onChange({ ...config, enabled: checked })
            }
          }}
        />
      </div>

      {/* 不启用Switch时显示直连拓扑选项 */}
      {!config.enabled && (
        <div style={configRowStyle}>
          <Text>直连拓扑</Text>
          <Select
            size="small"
            value={config.direct_topology || 'none'}
            onChange={(v) => onChange({ ...config, direct_topology: v })}
            style={{ width: 150 }}
            options={directTopologyOptions}
          />
        </div>
      )}

      {config.enabled && (
        <>
          {/* 保留节点直连 */}
          <div style={configRowStyle}>
            <Text>保留节点直连</Text>
            <Switch
              size="small"
              checked={config.keep_direct_topology || false}
              onChange={(checked) => onChange({ ...config, keep_direct_topology: checked })}
            />
          </div>

          {/* 保留直连时选择拓扑类型 */}
          {config.keep_direct_topology && (
            <div style={configRowStyle}>
              <Text>直连拓扑</Text>
              <Select
                size="small"
                value={config.direct_topology || 'full_mesh'}
                onChange={(v) => onChange({ ...config, direct_topology: v })}
                style={{ width: 150 }}
                options={directTopologyOptions}
              />
            </div>
          )}

          {/* 连接模式 */}
          <div style={configRowStyle}>
            <Text>Switch连接模式</Text>
            <Select
              size="small"
              value={config.connection_mode || 'full_mesh'}
              onChange={(v: SwitchConnectionMode) => onChange({ ...config, connection_mode: v })}
              style={{ width: 120 }}
              options={[
                { value: 'full_mesh', label: '全连接' },
                { value: 'custom', label: '自定义' },
              ]}
            />
          </div>

          {/* 自定义模式：节点连接Switch数 */}
          {config.connection_mode === 'custom' && (
            <div style={configRowStyle}>
              <Text>节点连接Switch数</Text>
              <InputNumber
                min={1}
                max={config.layers[0]?.count || 1}
                size="small"
                value={Math.min(config.downlink_redundancy || 1, config.layers[0]?.count || 1)}
                onChange={(v) => onChange({ ...config, downlink_redundancy: v || 1 })}
                style={{ width: 60 }}
              />
            </div>
          )}

          {/* Switch 3D显示配置（仅rack层级且3D视图时显示） */}
          {levelKey === 'inter_board' && viewMode === '3d' && (
            <>
              <div style={configRowStyle}>
                <Text>Switch位置</Text>
                <Radio.Group
                  size="small"
                  value={config.switch_position || 'top'}
                  onChange={(e) => onChange({ ...config, switch_position: e.target.value })}
                >
                  <Radio.Button value="top">顶部</Radio.Button>
                  <Radio.Button value="middle">中间</Radio.Button>
                  <Radio.Button value="bottom">底部</Radio.Button>
                </Radio.Group>
              </div>
              <div style={configRowStyle}>
                <Text>Switch高度</Text>
                <Radio.Group
                  size="small"
                  value={config.switch_u_height || 1}
                  onChange={(e) => onChange({ ...config, switch_u_height: e.target.value })}
                >
                  <Radio.Button value={1}>1U</Radio.Button>
                  <Radio.Button value={2}>2U</Radio.Button>
                  <Radio.Button value={4}>4U</Radio.Button>
                </Radio.Group>
              </div>
            </>
          )}

          <Divider style={{ margin: '8px 0' }} />

          {/* Switch层列表 */}
          <Text type="secondary" style={{ fontSize: 11 }}>Switch层配置 (从下到上)</Text>
          {config.layers.map((layer, index) => (
            <div key={index} style={{ marginTop: 8, padding: 8, background: '#f5f5f5', borderRadius: 8 }}>
              {/* 第一行：层名称和删除按钮 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Text style={{ fontSize: 11, color: '#666' }}>层名称</Text>
                  <Input
                    size="small"
                    placeholder="如 leaf, spine"
                    value={layer.layer_name}
                    onChange={(e) => updateLayer(index, 'layer_name', e.target.value)}
                    style={{ width: 80 }}
                  />
                </div>
                <Button
                  type="text"
                  danger
                  size="small"
                  icon={<MinusCircleOutlined />}
                  onClick={() => removeLayer(index)}
                />
              </div>
              {/* 第二行：Switch类型和数量 */}
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                <Select
                  size="small"
                  value={layer.switch_type_id}
                  onChange={(v) => updateLayer(index, 'switch_type_id', v)}
                  style={{ flex: 1 }}
                  options={switchTypes.map(t => ({ value: t.id, label: `${t.name} (${t.port_count}口)` }))}
                />
                <Text style={{ fontSize: 11, color: '#666' }}>×</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={16}
                  value={layer.count}
                  onChange={(v) => updateLayer(index, 'count', v || 1)}
                  style={{ width: 60 }}
                />
                <Text style={{ fontSize: 11, color: '#666' }}>台</Text>
              </div>
              {/* 第三行：同层互联选项 */}
              <Checkbox
                checked={layer.inter_connect}
                onChange={(e) => updateLayer(index, 'inter_connect', e.target.checked)}
              >
                <Text style={{ fontSize: 11 }}>同层互联</Text>
              </Checkbox>
            </div>
          ))}

          <Button
            type="dashed"
            size="small"
            icon={<PlusOutlined />}
            onClick={addLayer}
            style={{ marginTop: 8, width: '100%' }}
          >
            添加Switch层
          </Button>
        </>
      )}

    </div>
  )
}

// ============================================
// 连接编辑子组件
// ============================================

interface ConnectionEditPanelProps {
  manualConnectionConfig?: ManualConnectionConfig
  onManualConnectionConfigChange?: (config: ManualConnectionConfig) => void
  connectionMode?: ConnectionMode
  onConnectionModeChange?: (mode: ConnectionMode) => void
  selectedNodes?: Set<string>
  onSelectedNodesChange?: (nodes: Set<string>) => void
  targetNodes?: Set<string>
  onTargetNodesChange?: (nodes: Set<string>) => void
  onBatchConnect?: (level: HierarchyLevel) => void
  onDeleteManualConnection?: (id: string) => void
  currentViewConnections?: Array<{ source: string; target: string; type?: string; bandwidth?: number; latency?: number }>
  onDeleteConnection?: (source: string, target: string) => void
  onUpdateConnectionParams?: (source: string, target: string, bandwidth?: number, latency?: number) => void
  configRowStyle: React.CSSProperties
  currentLevel?: string
}

export const ConnectionEditPanel: React.FC<ConnectionEditPanelProps> = ({
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
  configRowStyle: _configRowStyle,
  currentLevel = 'datacenter',
}) => {
  void _configRowStyle
  // 获取当前层级
  const getCurrentHierarchyLevel = (): HierarchyLevel => {
    switch (currentLevel) {
      case 'datacenter': return 'datacenter'
      case 'pod': return 'pod'
      case 'rack': return 'rack'
      case 'board': return 'board'
      default: return 'datacenter'
    }
  }
  // 获取当前层级的默认参数
  const levelKey = currentLevel as 'datacenter' | 'pod' | 'rack' | 'board'
  const currentDefaults = manualConnectionConfig?.level_defaults?.[levelKey] || {}

  // 更新层级默认参数
  const updateLevelDefaults = (defaults: LevelConnectionDefaults) => {
    if (!onManualConnectionConfigChange) return
    const newConfig: ManualConnectionConfig = {
      ...(manualConnectionConfig || { enabled: true, mode: 'append', connections: [] }),
      level_defaults: {
        ...(manualConnectionConfig?.level_defaults || {}),
        [levelKey]: defaults,
      },
    }
    onManualConnectionConfigChange(newConfig)
  }

  // 更新手动连接的参数
  const updateManualConnectionParams = (connId: string, bandwidth?: number, latency?: number) => {
    if (!onManualConnectionConfigChange || !manualConnectionConfig) return
    const newConnections = manualConnectionConfig.connections.map(conn => {
      if (conn.id === connId) {
        return { ...conn, bandwidth, latency }
      }
      return conn
    })
    onManualConnectionConfigChange({
      ...manualConnectionConfig,
      connections: newConnections,
    })
  }

  return (
    <div style={{
      padding: 14,
      background: '#f5f5f5',
      borderRadius: 10,
      border: '1px solid rgba(0, 0, 0, 0.06)',
    }}>
      <Text strong style={{ display: 'block', marginBottom: 10, color: '#171717' }}>连接编辑</Text>

      {/* 层级默认带宽/延迟配置 */}
      <div style={{
        marginBottom: 12,
        padding: 10,
        background: '#fff',
        borderRadius: 8,
        border: '1px solid rgba(0,0,0,0.06)',
      }}>
        <div style={{ marginBottom: 8 }}>
          <Text style={{ fontSize: 12, color: '#333', fontWeight: 500 }}>层级默认参数</Text>
          <Text style={{ fontSize: 11, color: '#999', marginLeft: 8 }}>新建连接时自动应用</Text>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ fontSize: 12 }}>带宽:</Text>
            <InputNumber
              size="small"
              min={0}
              value={currentDefaults.bandwidth}
              onChange={(v) => updateLevelDefaults({ ...currentDefaults, bandwidth: v || undefined })}
              style={{ width: 80 }}
              placeholder="未设置"
            />
            <Text style={{ fontSize: 11, color: '#999' }}>GB/s</Text>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ fontSize: 12 }}>延迟:</Text>
            <InputNumber
              size="small"
              min={0}
              value={currentDefaults.latency}
              onChange={(v) => updateLevelDefaults({ ...currentDefaults, latency: v || undefined })}
              style={{ width: 80 }}
              placeholder="未设置"
            />
            <Text style={{ fontSize: 11, color: '#999' }}>us</Text>
          </div>
        </div>
      </div>

      {/* 编辑模式按钮 */}
      <div style={{ marginBottom: 12 }}>
        {connectionMode === 'view' ? (
          <Button
            type="default"
            onClick={() => onConnectionModeChange?.('select_source')}
          >
            编辑连接
          </Button>
        ) : (
          <Space>
            <Button
              type={connectionMode === 'select_source' ? 'primary' : 'default'}
              onClick={() => onConnectionModeChange?.('select_source')}
            >
              选源节点
            </Button>
            <Button
              type={connectionMode === 'select_target' ? 'primary' : 'default'}
              onClick={() => onConnectionModeChange?.('select_target')}
              disabled={selectedNodes.size === 0}
            >
              选目标节点
            </Button>
            <Button onClick={() => onConnectionModeChange?.('view')}>
              退出
            </Button>
          </Space>
        )}
      </div>

      {connectionMode !== 'view' && (
        <div style={{
          padding: 12,
          background: 'rgba(37, 99, 235, 0.04)',
          borderRadius: 8,
          marginBottom: 12,
          border: '1px solid rgba(37, 99, 235, 0.1)',
        }}>
          <Text style={{ fontSize: 12, display: 'block', marginBottom: 6, color: '#525252' }}>
            <strong>操作说明：</strong>
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: connectionMode === 'select_source' ? '#2563eb' : '#525252' }}>
            1. 点击图中节点选为源节点（绿色框）
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: connectionMode === 'select_target' ? '#2563eb' : '#525252' }}>
            2. 切换到"选目标节点"，点击选择目标（蓝色框）
          </Text>
          <Text style={{ fontSize: 12, display: 'block', color: '#525252' }}>
            3. 点击下方"确认连接"按钮完成
          </Text>
        </div>
      )}

      {/* 选中状态显示 */}
      {(selectedNodes.size > 0 || targetNodes.size > 0) && (
        <div style={{
          marginBottom: 12,
          padding: 12,
          background: 'rgba(5, 150, 105, 0.04)',
          borderRadius: 8,
          border: '1px solid rgba(5, 150, 105, 0.1)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <Text style={{ fontSize: 14 }}>
              <strong>源节点: {selectedNodes.size} 个</strong>
              {selectedNodes.size > 0 && (
                <span style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>
                  ({Array.from(selectedNodes).slice(0, 3).join(', ')}{selectedNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </Text>
            {selectedNodes.size > 0 && (
              <Button size="small" type="link" onClick={() => onSelectedNodesChange?.(new Set())}>清空</Button>
            )}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontSize: 14 }}>
              <strong>目标节点: {targetNodes.size} 个</strong>
              {targetNodes.size > 0 && (
                <span style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>
                  ({Array.from(targetNodes).slice(0, 3).join(', ')}{targetNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </Text>
            {targetNodes.size > 0 && (
              <Button size="small" type="link" onClick={() => onTargetNodesChange?.(new Set())}>清空</Button>
            )}
          </div>
          {selectedNodes.size > 0 && targetNodes.size > 0 && (() => {
            // 计算实际会创建的连接数（排除已存在的连接）
            let newCount = 0
            let existCount = 0
            selectedNodes.forEach(sourceId => {
              targetNodes.forEach(targetId => {
                if (sourceId === targetId) return
                // 检查是否已存在于当前视图连接中
                const existsInView = currentViewConnections.some(c =>
                  (c.source === sourceId && c.target === targetId) ||
                  (c.source === targetId && c.target === sourceId)
                )
                // 检查是否已存在于手动连接中
                const existsManual = manualConnectionConfig?.connections?.some(c =>
                  (c.source === sourceId && c.target === targetId) ||
                  (c.source === targetId && c.target === sourceId)
                )
                if (existsInView || existsManual) {
                  existCount++
                } else {
                  newCount++
                }
              })
            })
            return (
              <Button
                type="primary"
                style={{ marginTop: 12, width: '100%' }}
                onClick={() => onBatchConnect?.(getCurrentHierarchyLevel())}
                disabled={newCount === 0}
              >
                确认连接（{newCount} 条新连接{existCount > 0 ? `，${existCount} 条已存在` : ''}）
              </Button>
            )
          })()}
        </div>
      )}

      {/* 手动添加的连接列表 */}
      {(() => {
        // 过滤当前层级的手动连接
        const currentLevelConnections = manualConnectionConfig?.connections?.filter(
          conn => conn.hierarchy_level === currentLevel
        ) || []
        return (
          <Collapse
            size="small"
            bordered={false}
            style={{
              marginTop: 8,
              background: 'transparent',
            }}
            className="connection-collapse"
            items={[{
              key: 'manual',
              label: <span style={{ fontSize: 14 }}>手动连接 ({currentLevelConnections.length})</span>,
              style: { background: '#fff', borderRadius: 8, marginBottom: 8, border: '1px solid rgba(0,0,0,0.06)', overflow: 'hidden' },
              children: (
                <div style={{ maxHeight: 240, overflow: 'auto' }}>
                  {currentLevelConnections.map((conn) => {
                    // 判断是否使用默认值（值为空）
                    const useDefaultBandwidth = conn.bandwidth === undefined || conn.bandwidth === null
                    const useDefaultLatency = conn.latency === undefined || conn.latency === null
                    const hasCustom = !useDefaultBandwidth || !useDefaultLatency
                    // 显示值：空值时显示默认值
                    const displayBandwidth = useDefaultBandwidth ? currentDefaults.bandwidth : conn.bandwidth
                    const displayLatency = useDefaultLatency ? currentDefaults.latency : conn.latency
                    return (
                  <div
                    key={conn.id}
                    style={{
                      padding: 10,
                      background: 'rgba(5, 150, 105, 0.04)',
                      marginBottom: 8,
                      borderRadius: 8,
                      border: '1px solid rgba(5, 150, 105, 0.1)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 2 }}>
                          <Text style={{ fontSize: 11, color: '#999', width: 20, flexShrink: 0 }}>源:</Text>
                          <Text code style={{ fontSize: 12, wordBreak: 'break-all' }}>{conn.source}</Text>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <Text style={{ fontSize: 11, color: '#999', width: 20, flexShrink: 0 }}>→</Text>
                          <Text code style={{ fontSize: 12, wordBreak: 'break-all' }}>{conn.target}</Text>
                        </div>
                      </div>
                      <Space size={4} style={{ flexShrink: 0, marginLeft: 8 }}>
                        {hasCustom && (
                          <Button
                            type="text"
                            size="small"
                            icon={<UndoOutlined />}
                            title="重置为默认"
                            onClick={() => updateManualConnectionParams(conn.id, undefined, undefined)}
                            style={{ color: '#999' }}
                          />
                        )}
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteManualConnection?.(conn.id)}
                        />
                      </Space>
                    </div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultBandwidth ? '#999' : '#333' }}>带宽:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayBandwidth}
                          onChange={(v) => updateManualConnectionParams(conn.id, v ?? undefined, conn.latency)}
                          style={{ width: 80, color: useDefaultBandwidth ? '#999' : undefined }}
                          placeholder="GB/s"
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultLatency ? '#999' : '#333' }}>延迟:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayLatency}
                          onChange={(v) => updateManualConnectionParams(conn.id, conn.bandwidth, v ?? undefined)}
                          style={{ width: 80, color: useDefaultLatency ? '#999' : undefined }}
                          placeholder="us"
                        />
                      </div>
                    </div>
                  </div>
                )
                  })}
                  {currentLevelConnections.length === 0 && (
                    <Text type="secondary" style={{ fontSize: 13 }}>暂无手动连接</Text>
                  )}
                </div>
              ),
            }, {
          key: 'current',
          label: <span style={{ fontSize: 14 }}>当前连接 ({currentViewConnections.length})</span>,
          style: { background: '#fff', borderRadius: 8, border: '1px solid rgba(0,0,0,0.06)' },
          children: (
            <div style={{ maxHeight: 240, overflow: 'auto' }}>
              {currentViewConnections.map((conn, idx) => {
                // 判断是否使用默认值（值为空）
                const useDefaultBandwidth = conn.bandwidth === undefined || conn.bandwidth === null
                const useDefaultLatency = conn.latency === undefined || conn.latency === null
                const hasCustom = !useDefaultBandwidth || !useDefaultLatency
                // 显示值：空值时显示默认值
                const displayBandwidth = useDefaultBandwidth ? currentDefaults.bandwidth : conn.bandwidth
                const displayLatency = useDefaultLatency ? currentDefaults.latency : conn.latency
                return (
                  <div
                    key={`auto-${idx}`}
                    style={{
                      padding: 10,
                      background: '#fff',
                      marginBottom: 8,
                      borderRadius: 8,
                      border: '1px solid rgba(0, 0, 0, 0.06)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 2 }}>
                          <Text style={{ fontSize: 11, color: '#999', width: 20, flexShrink: 0 }}>源:</Text>
                          <Text code style={{ fontSize: 12, wordBreak: 'break-all' }}>{conn.source}</Text>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <Text style={{ fontSize: 11, color: '#999', width: 20, flexShrink: 0 }}>→</Text>
                          <Text code style={{ fontSize: 12, wordBreak: 'break-all' }}>{conn.target}</Text>
                        </div>
                      </div>
                      <Space size={4} style={{ flexShrink: 0, marginLeft: 8 }}>
                        {hasCustom && (
                          <Button
                            type="text"
                            size="small"
                            icon={<UndoOutlined />}
                            title="重置为默认"
                            onClick={() => onUpdateConnectionParams?.(conn.source, conn.target, undefined, undefined)}
                            style={{ color: '#999' }}
                          />
                        )}
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => onDeleteConnection?.(conn.source, conn.target)}
                        />
                      </Space>
                    </div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultBandwidth ? '#999' : '#333' }}>带宽:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayBandwidth}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, v ?? undefined, conn.latency)}
                          style={{ width: 80, color: useDefaultBandwidth ? '#999' : undefined }}
                          placeholder="GB/s"
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Text style={{ fontSize: 11, color: useDefaultLatency ? '#999' : '#333' }}>延迟:</Text>
                        <InputNumber
                          size="small"
                          min={0}
                          value={displayLatency}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, conn.bandwidth, v ?? undefined)}
                          style={{ width: 80, color: useDefaultLatency ? '#999' : undefined }}
                          placeholder="us"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
              {currentViewConnections.length === 0 && (
                <Text type="secondary" style={{ fontSize: 13 }}>暂无连接</Text>
              )}
            </div>
          ),
        }]}
      />
        )
      })()}
    </div>
  )
}

