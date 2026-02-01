import React from 'react'
import { Trash2, Plus, MinusCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Checkbox } from '@/components/ui/checkbox'
import { NumberInput } from '@/components/ui/number-input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { BaseCard } from '@/components/common/BaseCard'
import {
  HierarchyLevelSwitchConfig, SwitchTypeConfig, SwitchLayerConfig,
  ManualConnectionConfig, ConnectionMode, SwitchConnectionMode, HierarchyLevel,
} from '../../types'

// ============================================
// Switch层级配置子组件
// ============================================

interface SwitchLevelConfigProps {
  levelKey: string
  config: HierarchyLevelSwitchConfig
  switchTypes: SwitchTypeConfig[]
  onChange: (config: HierarchyLevelSwitchConfig) => void
  configRowStyle: React.CSSProperties
  viewMode?: '3d' | 'topology' | 'knowledge'
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
        <span className="text-sm">启用Switch</span>
        <Switch
          checked={config.enabled}
          onCheckedChange={(checked) => {
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
          <span className="text-sm">直连拓扑</span>
          <Select
            value={config.direct_topology || 'none'}
            onValueChange={(v) => onChange({ ...config, direct_topology: v as any })}
          >
            <SelectTrigger className="w-[150px] h-7">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {directTopologyOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {config.enabled && (
        <>
          {/* 保留节点直连 */}
          <div style={configRowStyle}>
            <span className="text-sm">保留节点直连</span>
            <Switch
              checked={config.keep_direct_topology || false}
              onCheckedChange={(checked) => onChange({ ...config, keep_direct_topology: checked })}
            />
          </div>

          {/* 保留直连时选择拓扑类型 */}
          {config.keep_direct_topology && (
            <div style={configRowStyle}>
              <span className="text-sm">直连拓扑</span>
              <Select
                value={config.direct_topology || 'full_mesh'}
                onValueChange={(v) => onChange({ ...config, direct_topology: v as any })}
              >
                <SelectTrigger className="w-[150px] h-7">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {directTopologyOptions.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* 连接模式 */}
          <div style={configRowStyle}>
            <span className="text-sm">Switch连接模式</span>
            <Select
              value={config.connection_mode || 'full_mesh'}
              onValueChange={(v: SwitchConnectionMode) => onChange({ ...config, connection_mode: v })}
            >
              <SelectTrigger className="w-[120px] h-7">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="full_mesh">全连接</SelectItem>
                <SelectItem value="custom">自定义</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* 自定义模式：节点连接Switch数 */}
          {config.connection_mode === 'custom' && (
            <div style={configRowStyle}>
              <span className="text-sm">节点连接Switch数</span>
              <NumberInput
                min={1}
                max={config.layers[0]?.count || 1}
                value={Math.min(config.downlink_redundancy || 1, config.layers[0]?.count || 1)}
                onChange={(v) => onChange({ ...config, downlink_redundancy: v || 1 })}
                className="w-[60px]"
              />
            </div>
          )}

          {/* Switch 3D显示配置（仅rack层级且3D视图时显示） */}
          {levelKey === 'inter_board' && viewMode === '3d' && (
            <>
              <div style={configRowStyle}>
                <span className="text-sm">Switch位置</span>
                <div className="flex rounded-md border border-gray-200 overflow-hidden">
                  {['top', 'middle', 'bottom'].map((pos) => (
                    <button
                      key={pos}
                      onClick={() => onChange({ ...config, switch_position: pos as 'top' | 'middle' | 'bottom' })}
                      className={`px-2 py-1 text-xs transition-colors ${
                        (config.switch_position || 'top') === pos
                          ? 'bg-blue-500 text-white'
                          : 'bg-white text-gray-600 hover:bg-gray-50'
                      } ${pos !== 'top' ? 'border-l border-gray-200' : ''}`}
                    >
                      {pos === 'top' ? '顶部' : pos === 'middle' ? '中间' : '底部'}
                    </button>
                  ))}
                </div>
              </div>
              <div style={configRowStyle}>
                <span className="text-sm">Switch高度</span>
                <div className="flex rounded-md border border-gray-200 overflow-hidden">
                  {[1, 2, 4].map((h) => (
                    <button
                      key={h}
                      onClick={() => onChange({ ...config, switch_u_height: h })}
                      className={`px-2 py-1 text-xs transition-colors ${
                        (config.switch_u_height || 1) === h
                          ? 'bg-blue-500 text-white'
                          : 'bg-white text-gray-600 hover:bg-gray-50'
                      } ${h !== 1 ? 'border-l border-gray-200' : ''}`}
                    >
                      {h}U
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="border-t border-gray-200 my-2" />

          {/* Switch层列表 */}
          <span className="text-xs text-gray-500">Switch层配置 (从下到上)</span>
          {config.layers.map((layer, index) => (
            <div key={index} className="mt-2 p-2 bg-gray-100 rounded-lg">
              {/* 第一行：层名称和删除按钮 */}
              <div className="flex justify-between items-center mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-gray-500">层名称</span>
                  <Input
                    placeholder="如 leaf, spine"
                    value={layer.layer_name}
                    onChange={(e) => updateLayer(index, 'layer_name', e.target.value)}
                    className="w-20 h-7"
                  />
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 text-red-500 hover:text-red-600"
                  onClick={() => removeLayer(index)}
                >
                  <MinusCircle className="h-4 w-4" />
                </Button>
              </div>
              {/* 第二行：Switch类型和数量 */}
              <div className="flex gap-2 items-center mb-1.5">
                <Select
                  value={layer.switch_type_id}
                  onValueChange={(v) => updateLayer(index, 'switch_type_id', v)}
                >
                  <SelectTrigger className="flex-1 h-7">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {switchTypes.map((t) => (
                      <SelectItem key={t.id} value={t.id}>{t.name} ({t.port_count}口)</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <span className="text-[11px] text-gray-500">×</span>
                <NumberInput
                  min={1}
                  max={16}
                  value={layer.count}
                  onChange={(v) => updateLayer(index, 'count', v || 1)}
                  className="w-[60px]"
                />
                <span className="text-[11px] text-gray-500">台</span>
              </div>
              {/* 第三行：同层互联选项 */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`inter-connect-${index}`}
                  checked={layer.inter_connect}
                  onCheckedChange={(checked) => updateLayer(index, 'inter_connect', checked)}
                />
                <label htmlFor={`inter-connect-${index}`} className="text-[11px]">同层互联</label>
              </div>
            </div>
          ))}

          <Button
            variant="outline"
            size="sm"
            onClick={addLayer}
            className="mt-2 w-full"
          >
            <Plus className="h-4 w-4 mr-1" />
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
  const [manualExpanded, setManualExpanded] = React.useState(true)
  const [currentExpanded, setCurrentExpanded] = React.useState(false)

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

  // 过滤当前层级的手动连接
  const currentLevelConnections = manualConnectionConfig?.connections?.filter(
    conn => conn.hierarchy_level === currentLevel
  ) || []

  // 连接编辑面板是否展开
  const [panelExpanded, setPanelExpanded] = React.useState(true)

  return (
    <BaseCard
      title="连接编辑"
      collapsible
      expanded={panelExpanded}
      onExpandChange={setPanelExpanded}
      gradient
    >

      {/* 编辑模式按钮 */}
      <div className="mb-3">
        {connectionMode === 'view' ? (
          <Button
            variant="outline"
            onClick={() => onConnectionModeChange?.('select_source')}
          >
            编辑连接
          </Button>
        ) : (
          <div className="flex gap-2">
            <Button
              variant={connectionMode === 'select_source' ? 'default' : 'outline'}
              onClick={() => onConnectionModeChange?.('select_source')}
            >
              选源节点
            </Button>
            <Button
              variant={connectionMode === 'select_target' ? 'default' : 'outline'}
              onClick={() => onConnectionModeChange?.('select_target')}
              disabled={selectedNodes.size === 0}
            >
              选目标节点
            </Button>
            <Button variant="outline" onClick={() => onConnectionModeChange?.('view')}>
              退出
            </Button>
          </div>
        )}
      </div>

      {connectionMode !== 'view' && (
        <div className="p-3 bg-blue-50 rounded-lg mb-3 border border-blue-100">
          <span className="text-xs block mb-1.5 text-gray-600">
            <strong>操作说明：</strong>
          </span>
          <span className={`text-xs block ${connectionMode === 'select_source' ? 'text-blue-600' : 'text-gray-600'}`}>
            1. 点击图中节点选为源节点（绿色框）
          </span>
          <span className={`text-xs block ${connectionMode === 'select_target' ? 'text-blue-600' : 'text-gray-600'}`}>
            2. 切换到"选目标节点"，点击选择目标（蓝色框）
          </span>
          <span className="text-xs block text-gray-600">
            3. 点击下方"确认连接"按钮完成
          </span>
        </div>
      )}

      {/* 选中状态显示 */}
      {(selectedNodes.size > 0 || targetNodes.size > 0) && (
        <div className="mb-3 p-3 bg-green-50 rounded-lg border border-green-100">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm">
              <strong>源节点: {selectedNodes.size} 个</strong>
              {selectedNodes.size > 0 && (
                <span className="text-xs text-gray-500 ml-2">
                  ({Array.from(selectedNodes).slice(0, 3).join(', ')}{selectedNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </span>
            {selectedNodes.size > 0 && (
              <Button variant="link" size="sm" className="h-auto p-0" onClick={() => onSelectedNodesChange?.(new Set())}>清空</Button>
            )}
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm">
              <strong>目标节点: {targetNodes.size} 个</strong>
              {targetNodes.size > 0 && (
                <span className="text-xs text-gray-500 ml-2">
                  ({Array.from(targetNodes).slice(0, 3).join(', ')}{targetNodes.size > 3 ? '...' : ''})
                </span>
              )}
            </span>
            {targetNodes.size > 0 && (
              <Button variant="link" size="sm" className="h-auto p-0" onClick={() => onTargetNodesChange?.(new Set())}>清空</Button>
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
                className="mt-3 w-full"
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
      <div className="mt-2 space-y-2">
        <BaseCard
          title="手动连接"
          collapsible
          expanded={manualExpanded}
          onExpandChange={setManualExpanded}
          collapsibleCount={currentLevelConnections.length}
          contentClassName="max-h-60 overflow-auto"
          gradient
        >
          {currentLevelConnections.map((conn) => {
                // 显示连接的带宽和延迟
                const displayBandwidth = conn.bandwidth
                const displayLatency = conn.latency
                return (
                  <div
                    key={conn.id}
                    className="p-2.5 bg-green-50 mb-2 rounded-lg border border-green-100"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center mb-0.5">
                          <span className="text-[11px] text-gray-400 w-5 shrink-0">源:</span>
                          <code className="text-xs break-all">{conn.source}</code>
                        </div>
                        <div className="flex items-center">
                          <span className="text-[11px] text-gray-400 w-5 shrink-0">→</span>
                          <code className="text-xs break-all">{conn.target}</code>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0 ml-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 w-7 p-0 text-red-500 hover:text-red-600"
                          onClick={() => onDeleteManualConnection?.(conn.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                    <div className="mt-2 flex gap-3 items-center">
                      <div className="flex items-center gap-1">
                        <span className="text-[11px] text-gray-700">带宽:</span>
                        <NumberInput
                          min={0}
                          value={displayBandwidth}
                          onChange={(v) => updateManualConnectionParams(conn.id, v, conn.latency)}
                          className="w-20"
                          placeholder="GB/s"
                        />
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-[11px] text-gray-700">延迟:</span>
                        <NumberInput
                          min={0}
                          value={displayLatency}
                          onChange={(v) => updateManualConnectionParams(conn.id, conn.bandwidth, v)}
                          className="w-20"
                          placeholder="us"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
              {currentLevelConnections.length === 0 && (
                <span className="text-gray-400 text-[13px]">暂无手动连接</span>
              )}
        </BaseCard>

        <BaseCard
          title="当前连接"
          collapsible
          expanded={currentExpanded}
          onExpandChange={setCurrentExpanded}
          collapsibleCount={currentViewConnections.length}
          contentClassName="max-h-60 overflow-auto"
          gradient
        >
          {currentViewConnections.map((conn, idx) => {
                return (
                  <div
                    key={`auto-${idx}`}
                    className="p-2.5 bg-white mb-2 rounded-lg border border-gray-200/50"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center mb-0.5">
                          <span className="text-[11px] text-gray-400 w-5 shrink-0">源:</span>
                          <code className="text-xs break-all">{conn.source}</code>
                        </div>
                        <div className="flex items-center">
                          <span className="text-[11px] text-gray-400 w-5 shrink-0">→</span>
                          <code className="text-xs break-all">{conn.target}</code>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0 ml-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 w-7 p-0 text-red-500 hover:text-red-600"
                          onClick={() => onDeleteConnection?.(conn.source, conn.target)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                    <div className="mt-2 flex gap-3 items-center">
                      <div className="flex items-center gap-1">
                        <span className="text-[11px] text-gray-700">带宽:</span>
                        <NumberInput
                          min={0}
                          value={conn.bandwidth}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, v, conn.latency)}
                          className="w-20"
                          placeholder="GB/s"
                        />
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-[11px] text-gray-700">延迟:</span>
                        <NumberInput
                          min={0}
                          value={conn.latency}
                          onChange={(v) => onUpdateConnectionParams?.(conn.source, conn.target, conn.bandwidth, v)}
                          className="w-20"
                          placeholder="us"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
              {currentViewConnections.length === 0 && (
                <span className="text-gray-400 text-[13px]">暂无连接</span>
              )}
        </BaseCard>
      </div>
    </BaseCard>
  )
}
