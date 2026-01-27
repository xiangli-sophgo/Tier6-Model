import React, { useState, useEffect, useRef } from 'react'
import {
  Server,
  Database,
  Save,
  FolderOpen,
  Trash2,
  Plus,
  MinusCircle,
  Network,
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Card, CardContent } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Textarea } from '@/components/ui/textarea'
import { GlobalSwitchConfig } from '../../types'
import { listConfigs, saveConfig, deleteConfig, SavedConfig } from '../../api/topology'
import { clearAllCache, NetworkConfig, SavedChipConfig } from '../../utils/storage'
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
import { BaseCard } from '../common/BaseCard'
import { getChipList, getChipConfig, saveCustomChipPreset, deleteCustomChipPreset, getChipInterconnectConfig } from '../../utils/llmDeployment/presets'
import { ChipHardwareConfig } from '../../utils/llmDeployment/types'

// 自定义数字输入组件
const NumberInput: React.FC<{
  value: number | undefined
  onChange: (value: number | undefined) => void
  min?: number
  max?: number
  step?: number
  className?: string
  disabled?: boolean
  suffix?: string
}> = ({ value, onChange, min = 0, max = 9999, step = 1, className = '', disabled = false, suffix }) => (
  <div className="flex items-center">
    <Input
      type="number"
      value={value ?? ''}
      onChange={(e) => {
        const v = e.target.value === '' ? undefined : parseFloat(e.target.value)
        if (v === undefined || (!isNaN(v) && v >= min && v <= max)) {
          onChange(v)
        }
      }}
      min={min}
      max={max}
      step={step}
      className={className}
      disabled={disabled}
    />
    {suffix && <span className="ml-1 text-xs text-gray-500">{suffix}</span>}
  </div>
)

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
  void onTrafficResultChange
  void onAnalysisDataChange
  void analysisHistory
  void onAddToHistory
  void onDeleteHistory
  void onClearHistory
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

  // 从 rackConfig 提取芯片配置列表
  const extractChipConfigs = (): SavedChipConfig[] => {
    const chipConfigMap = new Map<string, SavedChipConfig>()

    for (const board of rackConfig.boards) {
      const boardCount = board.count || 1
      for (const chip of board.chips) {
        const key = chip.preset_id || chip.name

        // 获取硬件参数 (优先使用自定义值，其次预设值)
        const presetConfig = chip.preset_id ? getChipConfig(chip.preset_id) : null
        const hardware = {
          chip_type: chip.name,
          flops_dtype: (presetConfig?.flops_dtype || 'BF16') as 'BF16' | 'FP16' | 'FP8' | 'INT8',
          compute_tflops_fp16: chip.compute_tflops_fp16 ?? presetConfig?.compute_tflops_fp16 ?? 100,
          compute_tops_int8: presetConfig?.compute_tops_int8,
          num_cores: presetConfig?.num_cores,
          memory_gb: chip.memory_gb ?? presetConfig?.memory_gb ?? 32,
          memory_bandwidth_gbps: chip.memory_bandwidth_gbps ?? presetConfig?.memory_bandwidth_gbps ?? 1000,
          memory_bandwidth_utilization: chip.memory_bandwidth_utilization ?? presetConfig?.memory_bandwidth_utilization ?? 0.9,
          l2_cache_mb: presetConfig?.l2_cache_mb,
          l2_bandwidth_gbps: presetConfig?.l2_bandwidth_gbps,
        }

        const existing = chipConfigMap.get(key)
        if (existing) {
          existing.total_count += chip.count * boardCount * podCount * racksPerPod
        } else {
          chipConfigMap.set(key, {
            preset_id: chip.preset_id,
            hardware,
            total_count: chip.count * boardCount * podCount * racksPerPod,
            chips_per_board: chip.count,
          })
        }
      }
    }

    return Array.from(chipConfigMap.values())
  }

  // 从连接配置提取网络参数
  const extractNetworkConfig = (): NetworkConfig => {
    // 默认值
    let intraNodeBandwidth = 900  // NVLink 4.0 GB/s
    let interNodeBandwidth = 50   // InfiniBand NDR GB/s
    let intraNodeLatency = 1      // us
    let interNodeLatency = 2      // us

    // 从 manualConnectionConfig 的 level_defaults 获取
    if (manualConnectionConfig?.level_defaults) {
      if (manualConnectionConfig.level_defaults.board?.bandwidth) {
        intraNodeBandwidth = manualConnectionConfig.level_defaults.board.bandwidth
      }
      if (manualConnectionConfig.level_defaults.board?.latency) {
        intraNodeLatency = manualConnectionConfig.level_defaults.board.latency
      }
      if (manualConnectionConfig.level_defaults.rack?.bandwidth) {
        interNodeBandwidth = manualConnectionConfig.level_defaults.rack.bandwidth
      }
      if (manualConnectionConfig.level_defaults.rack?.latency) {
        interNodeLatency = manualConnectionConfig.level_defaults.rack.latency
      }
    }

    return {
      intra_node_bandwidth_gbps: intraNodeBandwidth,
      inter_node_bandwidth_gbps: interNodeBandwidth,
      intra_node_latency_us: intraNodeLatency,
      inter_node_latency_us: interNodeLatency,
    }
  }

  // 保存当前配置
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      toast.error('请输入配置名称')
      return
    }
    try {
      // 提取芯片配置和网络配置
      const chipConfigs = extractChipConfigs()
      const networkConfig = extractNetworkConfig()

      await saveConfig({
        name: configName.trim(),
        description: configDesc.trim() || undefined,
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        board_configs: boardConfigs,
        // 扩展字段 - 保存完整配置用于部署分析
        rack_config: rackConfig,
        switch_config: switchConfig,
        manual_connections: manualConnectionConfig,
        generated_topology: topology || undefined,
        chip_configs: chipConfigs,
        network_config: networkConfig,
      })
      toast.success('配置保存成功')
      setSaveModalOpen(false)
      setConfigName('')
      setConfigDesc('')
      loadConfigList()
    } catch (error) {
      console.error('保存配置失败:', error)
      toast.error('保存配置失败')
    }
  }

  // 加载指定配置
  const handleLoadConfig = (config: SavedConfig) => {
    setPodCount(config.pod_count)
    setRacksPerPod(config.racks_per_pod)
    setBoardConfigs(config.board_configs)

    // 加载扩展配置 (如果存在)
    if (config.rack_config) {
      setRackConfig(config.rack_config as RackConfig)
    }
    if (config.switch_config) {
      setSwitchConfig(config.switch_config)
    }
    if (config.manual_connections && onManualConnectionConfigChange) {
      onManualConnectionConfigChange(config.manual_connections)
    }

    setLoadModalOpen(false)
    toast.success(`已加载配置: ${config.name}`)
  }

  // 删除配置
  const handleDeleteConfig = async (name: string) => {
    try {
      await deleteConfig(name)
      toast.success('配置已删除')
      loadConfigList()
    } catch (error) {
      console.error('删除配置失败:', error)
      toast.error('删除配置失败')
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
    toast.info(`已根据 ${primaryInterconnect.interconnect_type} 更新 Board 层连接参数: ${newBandwidth} GB/s, ${newLatency} us`)
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

  // 统计项组件
  const StatItem: React.FC<{ icon: React.ReactNode; label: string; value: number }> = ({ icon, label, value }) => (
    <div className="text-center">
      <div className="text-gray-500 text-xs mb-1 flex items-center justify-center gap-1">
        {icon}
        {label}
      </div>
      <div className="text-base font-semibold">{value}</div>
    </div>
  )

  // 拓扑配置内容（统计信息）
  const topologyConfigContent = (
    <div className="grid grid-cols-2 gap-4">
      <StatItem icon={<Server className="h-3.5 w-3.5" />} label="Pods" value={stats.pods} />
      <StatItem icon={<Database className="h-3.5 w-3.5" />} label="Racks" value={stats.racks} />
      <StatItem icon={<BoardIcon />} label="Boards" value={stats.boards} />
      <StatItem icon={<ChipIcon />} label="Chips" value={stats.chips} />
      {stats.switches > 0 && (
        <div className="col-span-2">
          <StatItem icon={<Network className="h-3.5 w-3.5" />} label="Switches" value={stats.switches} />
        </div>
      )}
    </div>
  )


  // 层级配置内容（节点配置 + Switch连接配置）
  const layerConfigContent = (
    <Tabs value={layerTabKey} onValueChange={setLayerTabKey} className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="datacenter">数据中心</TabsTrigger>
        <TabsTrigger value="pod">Pod层</TabsTrigger>
        <TabsTrigger value="rack">Rack层</TabsTrigger>
        <TabsTrigger value="board">Board层</TabsTrigger>
      </TabsList>

      <TabsContent value="datacenter">
        <div>
          {/* Pod数量配置 */}
          <div className="mb-3 p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">节点配置</span>
            <div style={configRowStyle}>
              <span>Pod 数量</span>
              <NumberInput
                min={1}
                max={10}
                value={podCount}
                onChange={(v) => setPodCount(v || 1)}
                className="w-20 h-8"
              />
            </div>
          </div>
          {/* Pod间连接配置 */}
          <div className="p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">连接配置</span>
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
            <div className="mt-3">
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
      </TabsContent>

      <TabsContent value="pod">
        <div>
          {/* Rack数量配置 */}
          <div className="mb-3 p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">节点配置</span>
            <div style={configRowStyle}>
              <span>每Pod机柜数</span>
              <NumberInput
                min={1}
                max={64}
                value={racksPerPod}
                onChange={(v) => setRacksPerPod(v || 1)}
                className="w-20 h-8"
              />
            </div>
          </div>
          {/* Rack间连接配置 */}
          <div className="p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">连接配置</span>
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
            <div className="mt-3">
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
      </TabsContent>

      <TabsContent value="rack">
        <div>
          {/* Board配置 */}
          <div className="mb-3 p-3.5 rounded-xl border border-gray-200/50" style={{ background: 'linear-gradient(135deg, rgba(248, 250, 252, 0.8) 0%, rgba(241, 245, 249, 0.8) 100%)' }}>
            {/* 标题和编辑开关 */}
            <div className="flex justify-between items-center mb-2">
              <span className="font-semibold">节点配置</span>
              <div className="flex items-center gap-1.5">
                <span className="text-gray-500 text-[11px]">编辑</span>
                <Switch
                  checked={rackEditMode}
                  onCheckedChange={setRackEditMode}
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
                <div className="mb-2 text-xs text-gray-600">
                  <span>容量: <strong>{rackConfig.total_u}U</strong></span>
                  <span className="mx-2 text-gray-300">|</span>
                  <span>已用: <strong className={isOverflow ? 'text-red-500' : ''}>{usedU}U</strong></span>
                  <span className="mx-2 text-gray-300">|</span>
                  <span>板卡: <strong>{totalBoards}</strong></span>
                  <span className="mx-2 text-gray-300">|</span>
                  <span>芯片: <strong>{totalChips}</strong></span>
                </div>
              )
            })()}

            {/* 编辑模式：Rack容量 */}
            {rackEditMode && (
              <div style={configRowStyle}>
                <span>Rack容量</span>
                <NumberInput
                  min={10}
                  max={60}
                  value={rackConfig.total_u}
                  onChange={(v) => setRackConfig(prev => ({ ...prev, total_u: v || 42 }))}
                  className="w-[70px] h-8"
                  suffix="U"
                />
              </div>
            )}

            {/* 板卡列表 */}
            <div className="mt-2">
              {rackConfig.boards.map((board, boardIndex) => (
                <div key={board.id} className="mb-1.5 p-1.5 px-2.5 bg-white rounded-lg border border-gray-200/50">
                  {rackEditMode ? (
                    /* 编辑模式 */
                    <>
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs whitespace-nowrap">名称:</span>
                          <Input
                            value={board.name}
                            onChange={(e) => {
                              const newBoards = [...rackConfig.boards]
                              newBoards[boardIndex] = { ...newBoards[boardIndex], name: e.target.value }
                              setRackConfig(prev => ({ ...prev, boards: newBoards }))
                            }}
                            className="w-[120px] h-7"
                          />
                          <span className="text-xs ml-2 whitespace-nowrap">高度:</span>
                          <NumberInput
                            min={1}
                            max={10}
                            value={board.u_height}
                            onChange={(v) => {
                              const newBoards = [...rackConfig.boards]
                              newBoards[boardIndex] = { ...newBoards[boardIndex], u_height: v || 1 }
                              setRackConfig(prev => ({ ...prev, boards: newBoards }))
                            }}
                            className="w-[70px] h-7"
                            suffix="U"
                          />
                          <span className="text-xs ml-2 whitespace-nowrap">数量:</span>
                          <NumberInput
                            min={0}
                            max={42}
                            value={board.count || 1}
                            onChange={(v) => {
                              const newBoards = [...rackConfig.boards]
                              newBoards[boardIndex] = { ...newBoards[boardIndex], count: v || 0 }
                              setRackConfig(prev => ({ ...prev, boards: newBoards }))
                            }}
                            className="w-[60px] h-7"
                          />
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-500 hover:text-red-600 h-7 w-7 p-0"
                          onClick={() => {
                            const newBoards = rackConfig.boards.filter((_, i) => i !== boardIndex)
                            setRackConfig(prev => ({ ...prev, boards: newBoards }))
                          }}
                          disabled={rackConfig.boards.length <= 1}
                        >
                          <MinusCircle className="h-4 w-4" />
                        </Button>
                      </div>
                    </>
                  ) : (
                    /* 展示模式 */
                    <div className="flex justify-between items-center">
                      <span className="text-[13px]">{board.name} ×{board.count || 1}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-gray-500 text-xs">{board.u_height}U</span>
                        <span className="text-gray-500 text-xs">
                          {board.chips.map(c => `${c.name}×${c.count}`).join(' ')}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* 编辑模式：添加板卡按钮 */}
            {rackEditMode && (
              <Button
                variant="outline"
                size="sm"
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
                className="w-full mt-1 border-dashed"
              >
                <Plus className="h-4 w-4 mr-1" />
                添加板卡类型
              </Button>
            )}
          </div>

          {/* Board间连接配置 */}
          <div className="p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">连接配置</span>
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
            <div className="mt-3">
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
      </TabsContent>

      <TabsContent value="board">
        <div>
          {/* 芯片配置 */}
          <div className="mb-3 p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">芯片配置</span>
            <span className="text-gray-500 text-[11px] mb-2.5 block">
              为每种板卡类型配置芯片
            </span>
            {rackConfig.boards.map((board, boardIndex) => (
              <div key={board.id} className="mb-2.5 p-2 px-2.5 bg-white rounded-md border border-gray-200">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="font-semibold text-xs">{board.name}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 border-dashed"
                    onClick={() => {
                      const newBoards = [...rackConfig.boards]
                      const newChips = [...newBoards[boardIndex].chips, { name: 'SG2262', count: 8, preset_id: 'sg2262' }]
                      newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                      setRackConfig(prev => ({ ...prev, boards: newBoards }))
                    }}
                  >
                    <Plus className="h-3.5 w-3.5 mr-1" />
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
                  const currentFlopsDtype = presetConfig?.flops_dtype ?? 'BF16'
                  // 检查参数是否被修改过
                  const isModified = presetConfig && (
                    (chip.compute_tflops_fp16 !== undefined && chip.compute_tflops_fp16 !== presetConfig.compute_tflops_fp16) ||
                    (chip.memory_gb !== undefined && chip.memory_gb !== presetConfig.memory_gb) ||
                    (chip.memory_bandwidth_gbps !== undefined && chip.memory_bandwidth_gbps !== presetConfig.memory_bandwidth_gbps) ||
                    (chip.memory_bandwidth_utilization !== undefined && chip.memory_bandwidth_utilization !== presetConfig.memory_bandwidth_utilization)
                  )
                  const isCustomPreset = chipPresetList.find(c => c.id === chip.preset_id)?.isCustom
                  return (
                    <div key={chipIndex} className="mb-2 p-2 px-2.5 rounded-md" style={{ background: '#fafafa', border: isModified ? '1px solid #faad14' : '1px solid transparent' }}>
                      {/* 类型选择 */}
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-xs w-[60px] flex-shrink-0">类型:</span>
                        <Select
                          value={chip.preset_id || 'custom'}
                          onValueChange={(value) => {
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
                        >
                          <SelectTrigger className="flex-1 h-7">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {chipPresetList.map(c => (
                              <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
                            ))}
                            <SelectItem value="custom">自定义...</SelectItem>
                          </SelectContent>
                        </Select>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-500 hover:text-red-600 h-7 w-7 p-0"
                          onClick={() => {
                            const newBoards = [...rackConfig.boards]
                            const newChips = newBoards[boardIndex].chips.filter((_, i) => i !== chipIndex)
                            newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                            setRackConfig(prev => ({ ...prev, boards: newBoards }))
                          }}
                          disabled={board.chips.length <= 1}
                        >
                          <MinusCircle className="h-4 w-4" />
                        </Button>
                      </div>
                      {/* 自定义类型时显示名称输入 */}
                      {!chip.preset_id && (
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-xs w-[60px] flex-shrink-0">名称:</span>
                          <Input
                            placeholder="芯片名称"
                            value={chip.name}
                            onChange={(e) => {
                              const newBoards = [...rackConfig.boards]
                              const newChips = [...newBoards[boardIndex].chips]
                              newChips[chipIndex] = { ...newChips[chipIndex], name: e.target.value }
                              newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                              setRackConfig(prev => ({ ...prev, boards: newBoards }))
                            }}
                            className="flex-1 h-7"
                          />
                        </div>
                      )}
                      {/* 数量 */}
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-xs w-[60px] flex-shrink-0">数量:</span>
                        <div className="flex flex-1 items-center">
                          <NumberInput
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
                            className="flex-1 h-7"
                          />
                          <span className="ml-1 text-xs text-gray-500">个</span>
                        </div>
                      </div>
                      {/* 第二行：芯片参数（可编辑） */}
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="text-xs w-[60px] flex-shrink-0 cursor-help">算力:</span>
                              </TooltipTrigger>
                              <TooltipContent>{currentFlopsDtype} 精度的理论峰值算力</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <div className="flex flex-1 items-center">
                            <NumberInput
                              min={1}
                              value={currentTflops}
                              onChange={(v) => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = [...newBoards[boardIndex].chips]
                                newChips[chipIndex] = { ...newChips[chipIndex], compute_tflops_fp16: v || undefined }
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              className="flex-1 h-7"
                            />
                            <span className="ml-1 text-xs text-gray-500 whitespace-nowrap">{currentFlopsDtype} TFLOPs</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="text-xs w-[60px] flex-shrink-0 cursor-help">显存:</span>
                              </TooltipTrigger>
                              <TooltipContent>DRAM 存储容量</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <div className="flex flex-1 items-center">
                            <NumberInput
                              min={1}
                              value={currentMemory}
                              onChange={(v) => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = [...newBoards[boardIndex].chips]
                                newChips[chipIndex] = { ...newChips[chipIndex], memory_gb: v || undefined }
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              className="flex-1 h-7"
                            />
                            <span className="ml-1 text-xs text-gray-500">GB</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="text-xs w-[60px] flex-shrink-0 cursor-help">带宽:</span>
                              </TooltipTrigger>
                              <TooltipContent>DRAM 理论带宽</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <div className="flex flex-1 items-center">
                            <NumberInput
                              min={1}
                              value={currentBandwidth}
                              onChange={(v) => {
                                const newBoards = [...rackConfig.boards]
                                const newChips = [...newBoards[boardIndex].chips]
                                newChips[chipIndex] = { ...newChips[chipIndex], memory_bandwidth_gbps: v || undefined }
                                newBoards[boardIndex] = { ...newBoards[boardIndex], chips: newChips }
                                setRackConfig(prev => ({ ...prev, boards: newBoards }))
                              }}
                              className="flex-1 h-7"
                            />
                            <span className="ml-1 text-xs text-gray-500">GB/s</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="text-xs w-[70px] flex-shrink-0 cursor-help">带宽利用率:</span>
                              </TooltipTrigger>
                              <TooltipContent>显存带宽的实际利用率，通常为0.8-0.95</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <NumberInput
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
                            className="flex-1 h-7"
                          />
                        </div>
                      </div>
                      {/* 第三行：操作按钮 */}
                      <div className="flex items-center gap-1.5 mt-1.5">
                        {isModified && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="link"
                                  size="sm"
                                  className="p-0 h-auto text-[11px]"
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
                              </TooltipTrigger>
                              <TooltipContent>重置为预设值</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                        {(isModified || !chip.preset_id) && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="link"
                                  size="sm"
                                  className="p-0 h-auto text-[11px]"
                                  onClick={() => {
                                    const newName = prompt('输入预设名称:', chip.name || '自定义芯片')
                                    if (newName) {
                                      const presetId = `custom-${Date.now()}`
                                      const config: ChipHardwareConfig = {
                                        chip_type: newName,
                                        flops_dtype: currentFlopsDtype,
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
                                      toast.success(`已保存预设: ${newName}`)
                                    }
                                  }}
                                >
                                  <Save className="h-3 w-3 mr-0.5" />
                                  保存预设
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>保存为新预设</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                        {isCustomPreset && chip.preset_id && (
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="link"
                                size="sm"
                                className="p-0 h-auto text-[11px] text-red-500"
                              >
                                <Trash2 className="h-3 w-3 mr-0.5" />
                                删除预设
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogHeader>
                                <AlertDialogTitle>确定删除此预设？</AlertDialogTitle>
                                <AlertDialogDescription>
                                  此操作将删除该自定义芯片预设，且无法恢复。
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>取消</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => {
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
                                      toast.success('已删除预设')
                                    }
                                  }}
                                  className="bg-red-500 hover:bg-red-600"
                                >
                                  删除
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            ))}
          </div>

          {/* Chip间连接配置 */}
          <div className="p-3.5 bg-gray-100 rounded-lg border border-gray-200/50">
            <span className="block font-semibold mb-2.5 text-gray-900">连接配置</span>
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
            <div className="mt-3">
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
      </TabsContent>
    </Tabs>
  )

  // Switch配置内容（只有Switch类型定义）
  const switchConfigContent = (
    <div>
      <span className="text-gray-500 text-[11px] block mb-2">
        定义可用的Switch型号，在各层级的连接配置中使用
      </span>
      {switchConfig.switch_types.map((swType, index) => (
        <div key={swType.id} className="mb-2 p-2 bg-gray-100 rounded-lg">
          <div className="flex gap-2 items-center">
            <Input
              placeholder="名称"
              value={swType.name}
              onChange={(e) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], name: e.target.value }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              className="flex-1 h-7"
            />
            <NumberInput
              min={8}
              max={1024}
              value={swType.port_count}
              onChange={(v) => {
                const newTypes = [...switchConfig.switch_types]
                newTypes[index] = { ...newTypes[index], port_count: v || 48 }
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
              className="w-20 h-7"
            />
            <span className="text-xs text-gray-600">端口</span>
            <Button
              variant="ghost"
              size="sm"
              className="text-red-500 hover:text-red-600 h-7 w-7 p-0"
              disabled={switchConfig.switch_types.length <= 1}
              onClick={() => {
                const newTypes = switchConfig.switch_types.filter((_, i) => i !== index)
                setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
              }}
            >
              <MinusCircle className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          const newId = `switch_${Date.now()}`
          const newTypes = [...switchConfig.switch_types, { id: newId, name: '新Switch', port_count: 48 }]
          setSwitchConfig(prev => ({ ...prev, switch_types: newTypes }))
        }}
        className="w-full border-dashed"
      >
        <Plus className="h-4 w-4 mr-1" />
        添加Switch类型
      </Button>
    </div>
  )

  return (
    <TooltipProvider>
      <div className="flex flex-col">
        {/* 拓扑配置内容 */}
        <>
            {/* 拓扑汇总 */}
            <div className="mb-3">
              <BaseCard
                title={<>拓扑汇总 <span className="text-xs font-normal text-gray-400 ml-2">{summaryText}</span></>}
                accentColor="#5E6AD2"
                collapsible
                defaultExpanded={false}
              >
                {topologyConfigContent}
              </BaseCard>
            </div>

            {/* 层级配置 */}
            <div className="mb-3">
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
            <div className="mb-3">
              <BaseCard
                title="Switch配置"
                accentColor="#52c41a"
                collapsible
                defaultExpanded={false}
              >
                {switchConfigContent}
              </BaseCard>
            </div>

            {/* 保存/加载/清除配置按钮 */}
            <div className="grid grid-cols-3 gap-2 mt-4">
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="outline" className="text-red-500 hover:text-red-600">
                    <Trash2 className="h-4 w-4 mr-1" />
                    清除缓存
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>清除所有缓存</AlertDialogTitle>
                    <AlertDialogDescription>
                      确定要清除所有缓存数据吗？清除后页面将刷新。
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>取消</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={async () => {
                        try {
                          await clearAllCache()
                          toast.success('缓存已清除，即将刷新页面')
                          setTimeout(() => window.location.reload(), 500)
                        } catch (error) {
                          toast.error('清除缓存失败')
                        }
                      }}
                      className="bg-red-500 hover:bg-red-600"
                    >
                      确定
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <Button
                variant="outline"
                onClick={() => setSaveModalOpen(true)}
              >
                <Save className="h-4 w-4 mr-1" />
                保存配置
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  loadConfigList()
                  setLoadModalOpen(true)
                }}
              >
                <FolderOpen className="h-4 w-4 mr-1" />
                加载配置
              </Button>
            </div>
          </>

        {/* 保存配置模态框 */}
        <Dialog open={saveModalOpen} onOpenChange={setSaveModalOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>保存配置</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <span className="block mb-1">配置名称 *</span>
                <Input
                  placeholder="输入配置名称"
                  value={configName}
                  onChange={(e) => setConfigName(e.target.value)}
                />
              </div>
              <div>
                <span className="block mb-1">描述 (可选)</span>
                <Textarea
                  placeholder="输入配置描述"
                  value={configDesc}
                  onChange={(e) => setConfigDesc(e.target.value)}
                  rows={2}
                />
              </div>
              {savedConfigs.some(c => c.name === configName.trim()) && (
                <span className="text-amber-500 text-xs">
                  同名配置已存在，保存将覆盖原配置
                </span>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => {
                setSaveModalOpen(false)
                setConfigName('')
                setConfigDesc('')
              }}>
                取消
              </Button>
              <Button onClick={handleSaveConfig}>保存</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* 加载配置模态框 */}
        <Dialog open={loadModalOpen} onOpenChange={setLoadModalOpen}>
          <DialogContent className="max-w-[480px]">
            <DialogHeader>
              <DialogTitle>加载配置</DialogTitle>
            </DialogHeader>
            {savedConfigs.length === 0 ? (
              <span className="text-gray-500 py-4">暂无保存的配置</span>
            ) : (
              <div className="space-y-2 py-4">
                {savedConfigs.map(config => {
                  // 计算汇总信息
                  const hasExtendedConfig = Boolean(config.rack_config || config.chip_configs)
                  const totalChips = config.chip_configs?.reduce((sum, c) => sum + c.total_count, 0) || 0
                  const chipTypeNames = config.chip_configs?.map(c => c.hardware.chip_type).join(', ') || ''

                  return (
                    <Card
                      key={config.name}
                      className="cursor-pointer hover:bg-gray-50"
                      onClick={() => handleLoadConfig(config)}
                    >
                      <CardContent className="p-3">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <span className="font-semibold">{config.name}</span>
                            {config.description && (
                              <div><span className="text-gray-500 text-xs">{config.description}</span></div>
                            )}
                            <div className="mt-1">
                              <span className="text-gray-500 text-[11px]">
                                Pod:{config.pod_count} | Rack:{config.racks_per_pod}
                                {hasExtendedConfig && totalChips > 0 && ` | Chip: ${totalChips}`}
                              </span>
                            </div>
                            {hasExtendedConfig && chipTypeNames && (
                              <div className="mt-0.5">
                                <span className="text-green-500 text-[10px]">
                                  芯片: {chipTypeNames}
                                </span>
                              </div>
                            )}
                          </div>
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-red-500 hover:text-red-600 h-7 w-7 p-0"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent onClick={(e) => e.stopPropagation()}>
                              <AlertDialogHeader>
                                <AlertDialogTitle>确定删除此配置？</AlertDialogTitle>
                                <AlertDialogDescription>
                                  此操作将删除配置 "{config.name}"，且无法恢复。
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>取消</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleDeleteConfig(config.name)
                                  }}
                                  className="bg-red-500 hover:bg-red-600"
                                >
                                  删除
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  )
}
