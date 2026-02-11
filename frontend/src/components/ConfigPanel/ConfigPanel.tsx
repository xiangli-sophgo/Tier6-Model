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
// Card导入已移除 - 使用BaseCard代替
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { BaseCard } from '@/components/common/BaseCard'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { InfoTooltip } from '@/components/ui/info-tooltip'
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
import { getTopologies, getTopology as getTopology, createTopology, updateTopology, deleteTopology } from '../../api/math_model'
import type { TopologyListItem, TopologyConfig } from '../../types/math_model'
import { clearAllCache } from '../../utils/storage'
import {
  ChipIcon,
  BoardIcon,
  FlexBoardConfig,
  RackConfig,
  ConfigPanelProps,
  DEFAULT_RACK_CONFIG,
  DEFAULT_SWITCH_CONFIG,
  DEFAULT_HARDWARE_PARAMS,
  loadCachedConfig,
  saveCachedConfig,
  HardwareParams,
  configRowStyle,
  createDefaultChipPreset,
} from './shared'
import { SwitchLevelConfig, ConnectionEditPanel } from './components'
import { getChipList, getChipConfig, saveCustomChipPreset, getChipInterconnectConfig } from '../../utils/llmDeployment/presets'

// NumberInput 从公共组件导入
import { NumberInput } from '@/components/ui/number-input'
import { FormInputField } from '@/components/ui/form-input-field'
import { ChipParamEditor } from './ChipParamEditor'
import { setNested } from '../../utils/nestedObjectEditor'

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  topology,
  onGenerate,
  currentLevel = 'datacenter',
  // 芯片选择相关
  selectedChipId,
  onChipTabActivate,
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

  // 灵活Rack配置
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

  // 硬件参数配置（多芯片独立配置 + 互联参数）
  const [hardwareParams, setHardwareParams] = useState<HardwareParams>(() => {
    if (cachedConfig?.hardwareParams?.chips) {
      // 新格式：多芯片配置
      return {
        chips: { ...DEFAULT_HARDWARE_PARAMS.chips, ...cachedConfig.hardwareParams.chips },
        interconnect: {
          c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...cachedConfig.hardwareParams.interconnect?.c2c },
          b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...cachedConfig.hardwareParams.interconnect?.b2b },
          r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...cachedConfig.hardwareParams.interconnect?.r2r },
          p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...cachedConfig.hardwareParams.interconnect?.p2p },
        },
      }
    }
    // 兼容旧格式 (chip -> chips)
    if (cachedConfig?.hardwareParams?.chip) {
      const chipName = cachedConfig.hardwareParams.chip.name || 'SG2262'
      return {
        chips: { [chipName]: createDefaultChipPreset(chipName) },
        interconnect: {
          c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...cachedConfig.hardwareParams.interconnect?.c2c },
          b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...cachedConfig.hardwareParams.interconnect?.b2b },
          r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...cachedConfig.hardwareParams.interconnect?.r2r },
          p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...cachedConfig.hardwareParams.interconnect?.p2p },
        },
      }
    }
    return DEFAULT_HARDWARE_PARAMS
  })

  // 互联通信延迟配置
  const [commLatencyConfig, setCommLatencyConfig] = useState({
    bandwidth_utilization: 0.95,
    sync_latency_us: 0,
    switch_latency_us: 1.0,
    cable_latency_us: 0.025,
    memory_read_latency_us: 0.15,
    memory_write_latency_us: 0.01,
    noc_latency_us: 0.05,
    die_to_die_latency_us: 0.04,
  })

  // 保存/加载配置状态
  const [savedConfigs, setSavedConfigs] = useState<TopologyListItem[]>([])
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [loadModalOpen, setLoadModalOpen] = useState(false)
  const [configName, setConfigName] = useState('')
  const [configDesc, setConfigDesc] = useState('')

  // 加载配置列表
  const loadConfigList = async () => {
    try {
      const res = await getTopologies()
      setSavedConfigs(res.topologies)
    } catch (error) {
      console.error('加载配置列表失败:', error)
    }
  }

  useEffect(() => {
    loadConfigList()
  }, [])

  // 配置变化时自动保存到localStorage
  useEffect(() => {
    saveCachedConfig({ podCount, racksPerPod, rackConfig, switchConfig, manualConnectionConfig, hardwareParams })
  }, [podCount, racksPerPod, rackConfig, switchConfig, manualConnectionConfig, hardwareParams])

  // 配置变化时自动生成拓扑（防抖500ms）
  const isFirstRender = useRef(true)
  useEffect(() => {
    // 跳过首次渲染（避免页面加载时重复生成）
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }

    const timer = setTimeout(() => {
      // console.log('🔧 [ConfigPanel] 生成拓扑配置:', {
      //   podCount,
      //   racksPerPod,
      //   rackConfig: {
      //     total_u: rackConfig.total_u,
      //     boards: rackConfig.boards,
      //     boardsCount: rackConfig.boards.length,
      //   },
      //   switchConfig: switchConfig?.inter_board,
      // })
      onGenerate({
        pod_count: podCount,
        racks_per_pod: racksPerPod,
        rack_config: rackConfig,
        switch_config: switchConfig,
        manual_connections: manualConnectionConfig,
        interconnect_config: hardwareParams.interconnect,
      })
    }, 500)

    return () => clearTimeout(timer)
  }, [podCount, racksPerPod, rackConfig, switchConfig, manualConnectionConfig, hardwareParams.interconnect, onGenerate])

  // 当 rackConfig.boards 变化时，确保每个芯片类型都有对应的硬件配置
  useEffect(() => {
    const chipNames = new Set<string>()
    for (const board of rackConfig.boards) {
      for (const chip of board.chips) {
        chipNames.add(chip.name)
      }
    }

    setHardwareParams(prev => {
      const newChips = { ...prev.chips }
      let changed = false

      for (const name of chipNames) {
        if (!newChips[name]) {
          // 尝试从预设加载配置，否则使用默认值
          const board = rackConfig.boards.find(b => b.chips.some(c => c.name === name))
          const chipItem = board?.chips.find(c => c.name === name)
          if (chipItem?.preset_id) {
            // TODO: 从后端 API 获取芯片预设
            // 目前使用默认值，后续需要改为调用后端 API
            newChips[name] = createDefaultChipPreset(name)
            changed = true
            continue
          }
          // 使用默认值
          newChips[name] = createDefaultChipPreset(name)
          changed = true
        }
      }

      return changed ? { ...prev, chips: newChips } : prev
    })
  }, [rackConfig.boards])

  // 更新单个芯片参数的辅助函数（支持嵌套路径，如 "cores.count"）
  const updateChipParam = React.useCallback((chipName: string, path: string, value: any) => {
    setHardwareParams(prev => {
      const chip = JSON.parse(JSON.stringify(prev.chips[chipName])) as Record<string, any>
      setNested(chip, path, value)
      return {
        ...prev,
        chips: {
          ...prev.chips,
          [chipName]: chip as any
        }
      }
    })
  }, [])

  // 保存当前配置
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      toast.error('请输入配置名称')
      return
    }
    try {
      // 转换为 TopologyConfig 格式（pods 数组）
      const configToSave: TopologyConfig = {
        name: configName.trim(),
        description: configDesc.trim() || undefined,
        pods: [{
          count: podCount,
          racks: [{
            count: racksPerPod,
            boards: rackConfig ? rackConfig.boards.map(board => ({
              id: board.id,
              name: board.name,
              u_height: board.u_height,
              count: board.count,
              chips: board.chips.map(chip => ({
                name: chip.name,
                count: chip.count,
                preset_id: (chip as any).preset_id,  // ✅ 保留 preset_id
              })),
            })) : [],
            total_u: rackConfig?.total_u ?? 42,
          }],
        }],
        chips: hardwareParams?.chips as any,
        interconnect: {
          links: hardwareParams?.interconnect,
          comm_params: commLatencyConfig,
        },
      }

      // 检查是否已存在同名配置
      const exists = savedConfigs.some(c => c.name === configName.trim())
      if (exists) {
        await updateTopology(configName.trim(), configToSave)
      } else {
        await createTopology(configToSave)
      }
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

  // 加载指定配置（从后端获取完整 TopologyConfig 并解析 pods 格式）
  const handleLoadConfig = async (configName: string) => {
    try {
      const config = await getTopology(configName)

      // 从 pods 数组提取 podCount, racksPerPod, rackConfig
      if (config.pods && config.pods.length > 0) {
        const firstPod = config.pods[0]
        setPodCount(firstPod.count ?? 1)
        if (firstPod.racks && firstPod.racks.length > 0) {
          const firstRack = firstPod.racks[0]
          setRacksPerPod(firstRack.count ?? 1)
          setRackConfig({
            total_u: firstRack.total_u ?? 42,
            boards: firstRack.boards.map(b => ({
              id: b.id || '',
              name: b.name || 'Board',
              u_height: b.u_height || 2,
              count: b.count ?? 1,
              chips: b.chips.map(c => ({ name: c.name, count: c.count ?? 1 })),
            })),
          } as RackConfig)
        }
      }

      // 加载硬件参数配置
      if (config.chips) {
        const links = config.interconnect?.links
        setHardwareParams({
          chips: { ...DEFAULT_HARDWARE_PARAMS.chips, ...config.chips as any },
          interconnect: {
            c2c: { ...DEFAULT_HARDWARE_PARAMS.interconnect.c2c, ...links?.c2c },
            b2b: { ...DEFAULT_HARDWARE_PARAMS.interconnect.b2b, ...links?.b2b },
            r2r: { ...DEFAULT_HARDWARE_PARAMS.interconnect.r2r, ...links?.r2r },
            p2p: { ...DEFAULT_HARDWARE_PARAMS.interconnect.p2p, ...links?.p2p },
          },
        })
      }

      // 加载通信延迟配置
      if (config.interconnect?.comm_params) {
        setCommLatencyConfig(config.interconnect.comm_params as any)
      }

      setLoadModalOpen(false)
      toast.success(`已加载配置: ${configName}`)
    } catch (error) {
      console.error('加载配置失败:', error)
      toast.error('加载配置失败')
    }
  }

  // 删除配置
  const handleDeleteConfig = async (name: string) => {
    try {
      await deleteTopology(name)
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

    const newBandwidth = primaryInterconnect.intra_board_bandwidth_gbps
    const newLatency = primaryInterconnect.intra_board_latency_us // us

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

  // 芯片选择处理：当点击视图中的芯片时，切换到 Chip Tab 并滚动到对应芯片
  const chipPanelRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  useEffect(() => {
    if (selectedChipId) {
      // 切换到 Chip Tab
      setLayerTabKey('chip')
      if (onChipTabActivate) {
        onChipTabActivate()
      }

      // 滚动到对应的芯片面板
      setTimeout(() => {
        const panelElement = chipPanelRefs.current.get(selectedChipId)
        if (panelElement) {
          panelElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }
      }, 100) // 延迟确保 Tab 切换完成
    }
  }, [selectedChipId, onChipTabActivate])

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
      <TabsList className="grid w-full grid-cols-5">
        <TabsTrigger value="datacenter">数据中心</TabsTrigger>
        <TabsTrigger value="pod">Pod层</TabsTrigger>
        <TabsTrigger value="rack">Rack层</TabsTrigger>
        <TabsTrigger value="board">Board层</TabsTrigger>
        <TabsTrigger value="chip">Chip层</TabsTrigger>
      </TabsList>

      <TabsContent value="datacenter">
        <div className="space-y-3">
          {/* 节点配置 + 互联参数 - 合并的折叠面板 */}
          <BaseCard
            title="节点配置"
            collapsible
            defaultExpanded
            gradient
          >
            {/* 节点配置 */}
            <div className="grid grid-cols-3 gap-3">
              <FormInputField
                label="Pod 数量"
                tooltip="数据中心内的Pod数量"
                min={1}
                max={10}
                value={podCount}
                onChange={(v) => setPodCount(v || 1)}
              />
              <FormInputField
                label="P2P带宽 (GB/s)"
                tooltip="Pod间互联带宽"
                min={0}
                max={999999}
                step={1}
                value={hardwareParams.interconnect.p2p.bandwidth_gbps}
                onChange={(v) => setHardwareParams(prev => ({
                  ...prev,
                  interconnect: { ...prev.interconnect, p2p: { ...prev.interconnect.p2p, bandwidth_gbps: v ?? 100 } }
                }))}
              />
              <FormInputField
                label="P2P延迟 (us)"
                tooltip="Pod间互联延迟"
                min={0}
                step={0.1}
                value={hardwareParams.interconnect.p2p.latency_us}
                onChange={(v) => setHardwareParams(prev => ({
                  ...prev,
                  interconnect: { ...prev.interconnect, p2p: { ...prev.interconnect.p2p, latency_us: v ?? 5.0 } }
                }))}
              />
            </div>
          </BaseCard>
          {/* Pod间连接配置 - 折叠面板 */}
          <BaseCard title="连接配置" collapsible defaultExpanded gradient>
            <SwitchLevelConfig
              levelKey="inter_pod"
              config={switchConfig.inter_pod}
              switchTypes={switchConfig.switch_types}
              onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_pod: newConfig }))}
              configRowStyle={configRowStyle}
            />
          </BaseCard>
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
        <div className="space-y-3">
          {/* 节点配置 + 互联参数 - 合并的折叠面板 */}
          <BaseCard
            title="节点配置"
            collapsible
            defaultExpanded
            gradient
          >
            <div className="grid grid-cols-3 gap-3">
              <FormInputField
                label="每Pod机柜数"
                tooltip="每个Pod内的Rack数量"
                min={1}
                max={64}
                value={racksPerPod}
                onChange={(v) => setRacksPerPod(v || 1)}
              />
              <FormInputField
                label="R2R带宽 (GB/s)"
                tooltip="Rack间互联带宽"
                min={0}
                max={999999}
                step={1}
                value={hardwareParams.interconnect.r2r.bandwidth_gbps}
                onChange={(v) => setHardwareParams(prev => ({
                  ...prev,
                  interconnect: { ...prev.interconnect, r2r: { ...prev.interconnect.r2r, bandwidth_gbps: v ?? 200 } }
                }))}
              />
              <FormInputField
                label="R2R延迟 (us)"
                tooltip="Rack间互联延迟"
                min={0}
                step={0.1}
                value={hardwareParams.interconnect.r2r.latency_us}
                onChange={(v) => setHardwareParams(prev => ({
                  ...prev,
                  interconnect: { ...prev.interconnect, r2r: { ...prev.interconnect.r2r, latency_us: v ?? 2.0 } }
                }))}
              />
            </div>
          </BaseCard>
          {/* Rack间连接配置 - 折叠面板 */}
          <BaseCard title="连接配置" collapsible defaultExpanded gradient>
            <SwitchLevelConfig
              levelKey="inter_rack"
              config={switchConfig.inter_rack}
              switchTypes={switchConfig.switch_types}
              onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_rack: newConfig }))}
              configRowStyle={configRowStyle}
            />
          </BaseCard>
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
        <div className="space-y-3">
          {/* 节点配置 + 互联参数 - 合并的折叠面板 */}
          {(() => {
            const usedU = rackConfig.boards.reduce((sum, b) => sum + b.u_height * (b.count || 1), 0)
            const totalBoards = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1), 0)
            const totalChips = rackConfig.boards.reduce((sum, b) => sum + (b.count || 1) * b.chips.reduce((s, c) => s + c.count, 0), 0)
            const isOverflow = usedU > rackConfig.total_u
            return (
              <BaseCard
                title="节点配置"
                collapsible
                defaultExpanded
                gradient
              >
                    {/* B2B 互联参数 */}
                    <div className="grid grid-cols-2 gap-3 mb-3 pb-3 border-b border-dashed">
                      <FormInputField
                        label="B2B带宽 (GB/s)"
                        tooltip="Board间互联带宽"
                        min={0}
                        max={999999}
                        step={1}
                        value={hardwareParams.interconnect.b2b.bandwidth_gbps}
                        onChange={(v) => setHardwareParams(prev => ({
                          ...prev,
                          interconnect: { ...prev.interconnect, b2b: { ...prev.interconnect.b2b, bandwidth_gbps: v ?? 450 } }
                        }))}
                      />
                      <FormInputField
                        label="B2B延迟 (us)"
                        tooltip="Board间互联延迟"
                        min={0}
                        step={0.01}
                        value={hardwareParams.interconnect.b2b.latency_us}
                        onChange={(v) => setHardwareParams(prev => ({
                          ...prev,
                          interconnect: { ...prev.interconnect, b2b: { ...prev.interconnect.b2b, latency_us: v ?? 0.35 } }
                        }))}
                      />
                    </div>
                    {/* 编辑开关 */}
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-xs text-gray-600">
                        <span>容量: <strong>{rackConfig.total_u}U</strong></span>
                        <span className="mx-2 text-gray-300">|</span>
                        <span>已用: <strong className={isOverflow ? 'text-red-500' : ''}>{usedU}U</strong></span>
                        <span className="mx-2 text-gray-300">|</span>
                        <span>芯片: <strong>{totalChips}</strong></span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-gray-500 text-[11px]">编辑</span>
                        <Switch
                          checked={rackEditMode}
                          onCheckedChange={setRackEditMode}
                        />
                      </div>
                    </div>

                    {/* 编辑模式：Rack容量 */}
                    {rackEditMode && (
                      <div style={configRowStyle}>
                        <span className="text-xs">Rack容量</span>
                        <NumberInput
                          min={10}
                          max={60}
                          value={rackConfig.total_u || 42}
                          onChange={(v) => setRackConfig(prev => ({ ...prev, total_u: v || 42 }))}
                          className="w-[70px] h-7"
                          suffix="U"
                        />
                      </div>
                    )}

                    {/* 板卡列表 */}
                    <div className="mt-2">
                      {rackConfig.boards.map((board, boardIndex) => (
                        <div key={board.id} className="mb-1.5 p-1.5 px-2.5 bg-gray-50 rounded-lg border border-gray-200/50">
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
                                  <Select
                                    value={board.u_height.toString()}
                                    onValueChange={(v) => {
                                      const newBoards = [...rackConfig.boards]
                                      newBoards[boardIndex] = { ...newBoards[boardIndex], u_height: parseInt(v) }
                                      setRackConfig(prev => ({ ...prev, boards: newBoards }))
                                    }}
                                  >
                                    <SelectTrigger className="w-[70px] h-7">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="1">1U</SelectItem>
                                      <SelectItem value="2">2U</SelectItem>
                                      <SelectItem value="4">4U</SelectItem>
                                    </SelectContent>
                                  </Select>
                                  <span className="text-xs ml-2 whitespace-nowrap">数量:</span>
                                  <NumberInput
                                    min={1}
                                    max={42}
                                    value={board.count || 1}
                                    onChange={(v) => {
                                      const newBoards = [...rackConfig.boards]
                                      newBoards[boardIndex] = { ...newBoards[boardIndex], count: v || 1 }
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
              </BaseCard>
            )
          })()}

          {/* Board间连接配置 - 折叠面板 */}
          <BaseCard title="连接配置" collapsible defaultExpanded gradient>
            <SwitchLevelConfig
              levelKey="inter_board"
              config={switchConfig.inter_board}
              switchTypes={switchConfig.switch_types}
              onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_board: newConfig }))}
              configRowStyle={configRowStyle}
              viewMode={viewMode}
            />
          </BaseCard>
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
        <div className="space-y-3">
          {/* 芯片配置 + 互联参数 - 合并的折叠面板 */}
          <BaseCard
            title="节点配置"
            collapsible
            defaultExpanded
            gradient
          >
                {/* C2C 互联参数 */}
                <div className="grid grid-cols-2 gap-3 mb-3 pb-3 border-b border-dashed">
                  <FormInputField
                    label="C2C带宽 (GB/s)"
                    tooltip="Chip间互联带宽（板内）"
                    min={0}
                    max={999999}
                    step={1}
                    value={hardwareParams.interconnect.c2c.bandwidth_gbps}
                    onChange={(v) => setHardwareParams(prev => ({
                      ...prev,
                      interconnect: { ...prev.interconnect, c2c: { ...prev.interconnect.c2c, bandwidth_gbps: v ?? 900 } }
                    }))}
                  />
                  <FormInputField
                    label="C2C延迟 (us)"
                    tooltip="Chip间互联延迟"
                    min={0}
                    step={0.01}
                    value={hardwareParams.interconnect.c2c.latency_us}
                    onChange={(v) => setHardwareParams(prev => ({
                      ...prev,
                      interconnect: { ...prev.interconnect, c2c: { ...prev.interconnect.c2c, latency_us: v ?? 1.0 } }
                    }))}
                  />
                </div>
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
                  return (
                    <div key={chipIndex} className="mb-2 p-2 px-2.5 rounded-md" style={{ background: '#fafafa', border: '1px solid transparent' }}>
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
                              }
                            } else {
                              const preset = getChipConfig(value)
                              if (preset) {
                                newChips[chipIndex] = {
                                  ...newChips[chipIndex],
                                  name: preset.name,
                                  preset_id: value,
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
                            value={chip.count || 1}
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
                    </div>
                  )
                })}
              </div>
            ))}
          </BaseCard>

          {/* Chip间连接配置 - 折叠面板 */}
          <BaseCard title="连接配置" collapsible defaultExpanded gradient>
            <SwitchLevelConfig
              levelKey="inter_chip"
              config={switchConfig.inter_chip}
              switchTypes={switchConfig.switch_types}
              onChange={(newConfig) => setSwitchConfig(prev => ({ ...prev, inter_chip: newConfig }))}
              configRowStyle={configRowStyle}
            />
          </BaseCard>
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

      <TabsContent value="chip">
        <div className="space-y-3">
          {/* 按芯片名称分组显示（相同名称的芯片共享配置） */}
          {(() => {
            // 收集所有唯一的芯片名称及其总数
            const chipNameMap = new Map<string, { totalCount: number; boards: string[] }>()
            for (const board of rackConfig.boards) {
              for (const chip of board.chips) {
                const existing = chipNameMap.get(chip.name)
                if (existing) {
                  existing.totalCount += chip.count * (board.count || 1)
                  if (!existing.boards.includes(board.name)) {
                    existing.boards.push(board.name)
                  }
                } else {
                  chipNameMap.set(chip.name, {
                    totalCount: chip.count * (board.count || 1),
                    boards: [board.name]
                  })
                }
              }
            }
            const uniqueChips = Array.from(chipNameMap.entries())

            if (uniqueChips.length === 0) {
              return (
                <div className="text-center py-8 text-gray-400 text-sm">
                  请先在 Board 层配置芯片
                </div>
              )
            }

            return uniqueChips.map(([chipName, info], chipIndex) => {
              // 获取该芯片的硬件参数（如果不存在则使用默认值）
              const chipParams = hardwareParams.chips[chipName] || createDefaultChipPreset(chipName)

              return (
                <div
                  key={chipName}
                  ref={(el) => {
                    // 为了保持滚动到芯片面板的功能，使用第一个 board 的 id
                    const firstBoard = rackConfig.boards.find(b => b.chips.some(c => c.name === chipName))
                    if (firstBoard) {
                      const chipIdx = firstBoard.chips.findIndex(c => c.name === chipName)
                      const panelId = `${firstBoard.id}-${chipIdx}`
                      if (el) {
                        chipPanelRefs.current.set(panelId, el)
                      } else {
                        chipPanelRefs.current.delete(panelId)
                      }
                    }
                  }}
                >
                  <ChipParamEditor
                    chipName={chipName}
                    title={
                      <>
                        {chipName} <span className="text-gray-400 text-xs ml-2">共 {info.totalCount} 个</span>
                        <span className="text-gray-300 text-[10px] ml-2">({info.boards.join(', ')})</span>
                      </>
                    }
                    chipParams={chipParams}
                    originalParams={null}
                    isEditable={true}
                    onParamChange={(path, value) => updateChipParam(chipName, path, value)}
                    expanded={chipIndex === 0}
                  />
                </div>
              )
            })
          })()}
        </div>

        {/* 互联通信参数 - 全局配置 */}
        <div className="mt-4">

          {/* 互联通信参数 - 全局配置 */}
          <BaseCard title="Communication Parameters" collapsible defaultExpanded gradient>
            {/* Bandwidth */}
            <div className="grid grid-cols-2 gap-3 mb-3">
              <FormInputField
                  label="Bandwidth Utilization"
                  tooltip="Actual BW / Peak BW (0~1)"
                  min={0.5}
                  max={1.0}
                  step={0.01}
                  value={commLatencyConfig.bandwidth_utilization}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, bandwidth_utilization: v ?? 0.95 }))}
                />
                <FormInputField
                  label="Sync Latency (µs)"
                  tooltip="Synchronization barrier latency"
                  min={0}
                  max={10}
                  step={0.1}
                  value={commLatencyConfig.sync_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, sync_latency_us: v ?? 0 }))}
                />
            </div>

            {/* Network Latency */}
            <div className="border-t border-dashed my-3 pt-2">
              <span className="text-xs text-gray-500">Network Latency</span>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <FormInputField
                label="Switch Latency (µs)"
                  tooltip="Switch forwarding latency"
                  min={0}
                  max={10}
                  step={0.05}
                  value={commLatencyConfig.switch_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, switch_latency_us: v ?? 1.0 }))}
                />
                <FormInputField
                  label="Cable Latency (µs)"
                  tooltip="Cable transmission latency"
                  min={0}
                  max={1}
                  step={0.005}
                  value={commLatencyConfig.cable_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, cable_latency_us: v ?? 0.025 }))}
                />
            </div>

            {/* Chip Latency */}
            <div className="border-t border-dashed my-3 pt-2">
              <span className="text-xs text-gray-500">Chip Latency</span>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <FormInputField
                label="DDR Read Latency (µs)"
                  tooltip="HBM/GDDR read latency"
                  min={0}
                  max={1}
                  step={0.01}
                  value={commLatencyConfig.memory_read_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_read_latency_us: v ?? 0.15 }))}
                />
                <FormInputField
                  label="DDR Write Latency (µs)"
                  tooltip="HBM/GDDR write latency"
                  min={0}
                  max={1}
                  step={0.01}
                  value={commLatencyConfig.memory_write_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, memory_write_latency_us: v ?? 0.01 }))}
                />
                <FormInputField
                  label="NoC Latency (µs)"
                  tooltip="Network-on-Chip latency"
                  min={0}
                  max={1}
                  step={0.01}
                  value={commLatencyConfig.noc_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, noc_latency_us: v ?? 0.05 }))}
                />
                <FormInputField
                  label="Die-to-Die Latency (µs)"
                  tooltip="Die-to-Die interconnect latency"
                  min={0}
                  max={1}
                  step={0.01}
                  value={commLatencyConfig.die_to_die_latency_us}
                  onChange={(v) => setCommLatencyConfig(prev => ({ ...prev, die_to_die_latency_us: v ?? 0.04 }))}
                />
            </div>

            {/* 计算结果：通信启动开销 */}
            <div className="border-t border-dashed my-3 pt-2">
              <span className="text-xs text-gray-500">通信启动开销 (start_lat)</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <InfoTooltip
                content={
                  <div className="text-xs">
                    <div className="font-medium mb-1">AllReduce start_lat 计算公式:</div>
                    <div className="font-mono">2×c2c_latency + memory_read + memory_write + noc + 2×die_to_die</div>
                  </div>
                }
              >
                <div className="p-2 bg-gray-100 rounded border border-gray-300 cursor-help">
                  <span className="text-xs text-gray-500">AllReduce start_lat</span>
                  <div className="text-sm font-medium text-blue-500">
                    {(2 * hardwareParams.interconnect.c2c.latency_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us).toFixed(2)} µs
                  </div>
                </div>
              </InfoTooltip>
              <InfoTooltip
                content={
                  <div className="text-xs">
                    <div className="font-medium mb-1">Dispatch/Combine start_lat 计算公式:</div>
                    <div className="font-mono">2×c2c_latency + memory_read + memory_write + noc + 2×die_to_die + 2×switch + 2×cable</div>
                  </div>
                }
              >
                <div className="p-2 bg-gray-100 rounded border border-gray-300 cursor-help">
                  <span className="text-xs text-gray-500">Dispatch/Combine start_lat</span>
                  <div className="text-sm font-medium text-purple-500">
                    {(2 * hardwareParams.interconnect.c2c.latency_us + commLatencyConfig.memory_read_latency_us + commLatencyConfig.memory_write_latency_us + commLatencyConfig.noc_latency_us + 2 * commLatencyConfig.die_to_die_latency_us + 2 * commLatencyConfig.switch_latency_us + 2 * commLatencyConfig.cable_latency_us).toFixed(2)} µs
                  </div>
                </div>
              </InfoTooltip>
            </div>
          </BaseCard>
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
    <div className="flex flex-col gap-4">
        {/* 拓扑配置内容 */}
        <>
            {/* 拓扑汇总 */}
            <BaseCard
              title={<>拓扑汇总 <span className="text-xs font-normal text-gray-400 ml-2">{summaryText}</span></>}
              collapsible
              defaultExpanded={false}
              gradient
            >
              {topologyConfigContent}
            </BaseCard>

            {/* 层级配置 */}
            <BaseCard
              title="层级配置"
              collapsible
              defaultExpanded
              gradient
            >
              {layerConfigContent}
            </BaseCard>

            {/* Switch配置 */}
            <BaseCard
              title="Switch配置"
              collapsible
              defaultExpanded={false}
              gradient
            >
              {switchConfigContent}
            </BaseCard>

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
                onClick={() => {
                  // 自动生成配置名称: P{Pod数}-R{Rack总数}-B{Board总数}-C{Chip总数}
                  const autoName = `P${stats.pods}-R${stats.racks}-B${stats.boards}-C${stats.chips}`
                  setConfigName(autoName)
                  setSaveModalOpen(true)
                }}
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
                  return (
                    <BaseCard
                      key={config.name}
                      titleless
                      className="cursor-pointer hover:bg-gray-50"
                      onClick={() => handleLoadConfig(config.name)}
                    >
                      <div className="p-3">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <span className="font-semibold">{config.name}</span>
                            {config.description && (
                              <div><span className="text-gray-500 text-xs">{config.description}</span></div>
                            )}
                            {config.chip_count != null && (
                              <div className="mt-1">
                                <span className="text-gray-500 text-[11px]">
                                  Chip: {config.chip_count}
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
                      </div>
                    </BaseCard>
                  )
                })}
              </div>
            )}
          </DialogContent>
        </Dialog>
    </div>
  )
}
