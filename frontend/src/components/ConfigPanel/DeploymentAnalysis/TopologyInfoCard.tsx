/**
 * 拓扑信息卡片 - 可编辑展示组件
 *
 * 用于在部署分析面板中展示和编辑当前选中的拓扑配置信息
 * 支持芯片参数和互联参数的编辑
 *
 * 设计参考: TaskDetailPanel 的 InfoItem + InfoGrid 风格
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  AlertTriangle,
  Save,
  RefreshCw,
  ChevronUp,
  ChevronDown,
  Copy,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { BaseCard } from '@/components/common/BaseCard'
import { NumberInput } from '@/components/ui/number-input'
import { SavedConfig } from '../../../api/topology'
import { HardwareConfig, CommLatencyConfig } from '../../../utils/llmDeployment/types'
import { ChipGroupInfo } from '../../../utils/llmDeployment/topologyHardwareExtractor'
import { HardwareParams, DEFAULT_HARDWARE_PARAMS, createDefaultChipPreset } from '../shared'
import { ChipPreset } from '../../../types/tier6'

// 信息项组件（单个条目）- 与 TaskDetailPanel 保持一致
const InfoItem: React.FC<{
  label: string
  value: React.ReactNode
}> = ({ label, value }) => (
  <div className="flex flex-col gap-0.5 py-1.5 px-2.5 bg-gray-50/50 rounded border border-gray-100">
    <span className="text-[10px] text-gray-500">{label}</span>
    <span className="text-xs font-medium text-gray-800 break-all">{value ?? '-'}</span>
  </div>
)

// 信息网格组件（多列布局）
const InfoGrid: React.FC<{
  items: Array<{ label: string; value: React.ReactNode }>
  columns?: number
}> = ({ items, columns = 4 }) => (
  <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
    {items.map((item, index) => (
      <InfoItem key={index} label={item.label} value={item.value} />
    ))}
  </div>
)

// 分组标题（使用分割线）
const SectionHeader: React.FC<{
  title: string
  color?: string
}> = ({ title }) => {
  return (
    <div className="border-t border-dashed border-gray-200 pt-2 mt-2">
      <span className="text-xs text-gray-500">{title}</span>
    </div>
  )
}

// 局部CollapsibleSection已删除 - 统一使用BaseCard（带glassmorphism + collapsibleCount + onEdit）

interface TopologyInfoCardProps {
  /** 已保存的拓扑配置列表 */
  topologyConfigs: SavedConfig[]
  /** 当前选中的配置名称 */
  selectedConfigName?: string
  /** 选择配置的回调 */
  onSelectConfig: (configName: string | undefined) => void
  /** 跳转到互联拓扑页面的回调 */
  onNavigateToTopology?: () => void
  /** 芯片组信息 */
  chipGroups: ChipGroupInfo[]
  /** 当前选中的芯片类型 */
  selectedChipType?: string
  /** 选择芯片类型的回调 */
  onSelectChipType?: (chipType: string) => void
  /** 硬件配置（只读展示用） */
  hardwareConfig: HardwareConfig | null
  /** 拓扑层级信息 */
  topologyStats: {
    podCount: number
    rackCount: number
    boardCount: number
    chipCount: number
  }
  /** 互联参数（从 interconnect.links 获取） */
  interconnectParams?: {
    c2c?: { bandwidth_gbps: number; latency_us: number }
    b2b?: { bandwidth_gbps: number; latency_us: number }
    r2r?: { bandwidth_gbps: number; latency_us: number }
    p2p?: { bandwidth_gbps: number; latency_us: number }
  }
  /** 通信延迟配置 */
  commLatencyConfig?: CommLatencyConfig

  // ====== 可编辑模式新增 props ======
  /** 硬件参数（多芯片独立配置） - 可编辑 */
  hardwareParams?: HardwareParams
  /** 硬件参数变更回调 */
  onHardwareParamsChange?: (params: HardwareParams) => void
  /** 通信延迟配置变更回调 */
  onCommLatencyChange?: (config: CommLatencyConfig) => void
  /** 保存配置回调 */
  onSaveConfig?: () => void
  /** 另存为配置回调（参数：新配置名称，描述） */
  onSaveAsConfig?: (name: string, description?: string) => Promise<void>
  /** 重置配置回调 */
  onResetConfig?: () => void
  /** 拓扑配置列表（用于检查名称冲突） */
  allConfigs?: SavedConfig[]
}

// 格式化数值
const formatNumber = (value: number | undefined, decimals = 2): string => {
  if (value === undefined || value === null) return '-'
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(decimals)
}

// 格式化带宽
const formatBandwidth = (gbps: number | undefined): string => {
  if (gbps === undefined) return '-'
  if (gbps >= 1000) return `${(gbps / 1000).toFixed(1)} TB/s`
  return `${gbps} GB/s`
}

export const TopologyInfoCard: React.FC<TopologyInfoCardProps> = ({
  topologyConfigs,
  selectedConfigName,
  onSelectConfig,
  onNavigateToTopology,
  chipGroups,
  selectedChipType,
  onSelectChipType,
  hardwareConfig,
  topologyStats,
  interconnectParams,
  commLatencyConfig,
  // 可编辑模式 props
  hardwareParams,
  onHardwareParamsChange,
  onCommLatencyChange,
  onSaveConfig,
  onSaveAsConfig,
  onResetConfig,
  allConfigs,
}) => {
  const hasConfig = chipGroups.length > 0 && hardwareConfig

  // 是否为可编辑模式
  const isEditable = !!onHardwareParamsChange

  // 子卡片展开状态管理（默认全部折叠）
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    topology: false,
    comm_latency: false,
  })

  // 另存为弹窗状态
  const [saveAsModalOpen, setSaveAsModalOpen] = useState(false)
  const [newConfigName, setNewConfigName] = useState('')
  const [newConfigDesc, setNewConfigDesc] = useState('')

  // 原始配置快照（用于修改追踪）
  const [originalConfig, setOriginalConfig] = useState<{
    hardwareParams: HardwareParams | null
    commLatency: CommLatencyConfig | null
  }>({ hardwareParams: null, commLatency: null })

  // 使用ref记录上次的selectedConfigName，避免频繁更新快照
  const lastConfigNameRef = useRef<string | undefined>(selectedConfigName)
  const snapshotSavedRef = useRef<boolean>(false)

  // 当选择新的配置文件时，重置快照保存标记
  useEffect(() => {
    if (selectedConfigName !== lastConfigNameRef.current) {
      lastConfigNameRef.current = selectedConfigName
      snapshotSavedRef.current = false
    }
  }, [selectedConfigName])

  // 当配置就绪且还未保存快照时，保存原始快照（只保存一次）
  useEffect(() => {
    if (!snapshotSavedRef.current && hardwareParams && selectedConfigName) {
      snapshotSavedRef.current = true
      setOriginalConfig({
        hardwareParams: JSON.parse(JSON.stringify(hardwareParams)),
        commLatency: commLatencyConfig ? { ...commLatencyConfig } : null,
      })
      // console.log('[TopologyInfoCard] 保存原始快照:', { hardwareParams, commLatencyConfig })
    }
  }, [hardwareParams, commLatencyConfig, selectedConfigName])

  // 切换单个卡片
  const toggleSection = (key: string) => {
    setOpenSections(prev => ({ ...prev, [key]: !prev[key] }))
  }

  // 全部展开/折叠
  const toggleAllSections = () => {
    const allKeys = ['topology', ...chipGroups.map((_, i) => `chip_${i}`), 'comm_latency']
    const allOpen = allKeys.every(key => openSections[key])
    const newState: Record<string, boolean> = {}
    allKeys.forEach(key => { newState[key] = !allOpen })
    setOpenSections(newState)
  }

  // 更新单个芯片参数（支持嵌套路径，如 "cores.count" 或 "memory.gmem.capacity_gb"）
  const updateChipParam = (chipName: string, path: string, value: any) => {
    if (!hardwareParams || !onHardwareParamsChange) return
    const chip = { ...hardwareParams.chips[chipName] } as any
    const parts = path.split('.')

    if (parts.length === 1) {
      // 顶层字段
      chip[path] = value
    } else if (parts.length === 2) {
      // 一级嵌套，如 cores.count
      chip[parts[0]] = { ...chip[parts[0]], [parts[1]]: value }
    } else if (parts.length === 3) {
      // 二级嵌套，如 memory.gmem.capacity_gb
      chip[parts[0]] = {
        ...chip[parts[0]],
        [parts[1]]: { ...chip[parts[0]]?.[parts[1]], [parts[2]]: value }
      }
    } else if (parts.length === 4) {
      // 三级嵌套，如 compute_units.cube.mac_per_lane.BF16
      chip[parts[0]] = {
        ...chip[parts[0]],
        [parts[1]]: {
          ...chip[parts[0]]?.[parts[1]],
          [parts[2]]: { ...chip[parts[0]]?.[parts[1]]?.[parts[2]], [parts[3]]: value }
        }
      }
    }

    onHardwareParamsChange({
      ...hardwareParams,
      chips: { ...hardwareParams.chips, [chipName]: chip }
    })
  }

  // 获取嵌套属性值
  const getNestedValue = (obj: any, path: string): any => {
    return path.split('.').reduce((acc, part) => acc?.[part], obj)
  }

  // 更新互联参数
  const updateInterconnect = (level: 'c2c' | 'b2b' | 'r2r' | 'p2p', field: 'bandwidth_gbps' | 'latency_us', value: number) => {
    if (!hardwareParams || !onHardwareParamsChange) return
    onHardwareParamsChange({
      ...hardwareParams,
      interconnect: {
        ...hardwareParams.interconnect,
        [level]: { ...hardwareParams.interconnect[level], [field]: value }
      }
    })
  }

  // 更新通信延迟参数
  const updateCommLatency = (field: string, value: number) => {
    if (!commLatencyConfig || !onCommLatencyChange) return
    onCommLatencyChange({ ...commLatencyConfig, [field]: value })
  }

  // 检测互联参数是否被修改
  const isInterconnectModified = (level: 'c2c' | 'b2b' | 'r2r' | 'p2p', field: 'bandwidth_gbps' | 'latency_us'): boolean => {
    if (!originalConfig.hardwareParams || !hardwareParams) return false
    const originalValue = originalConfig.hardwareParams.interconnect[level][field]
    const currentValue = hardwareParams.interconnect[level][field]
    const isModified = originalValue !== currentValue
    // if (isModified) {
      // console.log(`[修改检测] ${level}.${field}: ${originalValue} → ${currentValue}`)
    // }
    return isModified
  }

  // 检测芯片参数是否被修改（支持嵌套路径）
  const isChipParamModified = (chipName: string, path: string): boolean => {
    if (!originalConfig.hardwareParams || !hardwareParams) {
      return false
    }
    const originalChip = originalConfig.hardwareParams.chips[chipName]
    const currentChip = hardwareParams.chips[chipName]
    if (!originalChip || !currentChip) {
      return false
    }
    const originalValue = getNestedValue(originalChip, path)
    const currentValue = getNestedValue(currentChip, path)
    const isModified = originalValue !== currentValue
    // if (isModified) {
      // console.log(`[修改检测] ${chipName}.${field}: ${(originalChip as any)[field]} → ${(currentChip as any)[field]}`)
    // }
    return isModified
  }

  // 检测通信延迟参数是否被修改
  const isCommLatencyModified = (field: string): boolean => {
    if (!originalConfig.commLatency || !commLatencyConfig) return false
    return (originalConfig.commLatency as any)[field] !== (commLatencyConfig as any)[field]
  }

  // 重置配置到原始状态
  const handleReset = () => {
    if (!originalConfig.hardwareParams && !originalConfig.commLatency) {
      toast.warning('没有可重置的原始配置')
      return
    }
    if (originalConfig.hardwareParams && onHardwareParamsChange) {
      onHardwareParamsChange(JSON.parse(JSON.stringify(originalConfig.hardwareParams)))
    }
    if (originalConfig.commLatency && onCommLatencyChange) {
      onCommLatencyChange({ ...originalConfig.commLatency })
    }
    toast.success('已重置到原始配置')
  }

  // 另存为处理函数
  const handleSaveAs = () => {
    setSaveAsModalOpen(true)
  }

  const handleConfirmSaveAs = async () => {
    if (!newConfigName.trim()) {
      toast.warning('请输入配置名称')
      return
    }
    // 检查名称是否已存在
    if ((allConfigs || topologyConfigs).some(c => c.name === newConfigName.trim())) {
      toast.error('配置名称已存在，请使用其他名称')
      return
    }
    if (onSaveAsConfig) {
      try {
        await onSaveAsConfig(newConfigName.trim(), newConfigDesc.trim() || undefined)
        setSaveAsModalOpen(false)
        setNewConfigName('')
        setNewConfigDesc('')
      } catch (error) {
        console.error('另存为失败:', error)
      }
    } else {
      toast.warning('另存为功能未实现')
    }
  }

  return (
    <BaseCard title="拓扑配置" collapsible gradient>
      <div className="space-y-3">
        {/* 配置文件选择器 - 和 Benchmark 风格一致 */}
        <div>
          <div className="mb-1 flex justify-between items-center">
            <span className="text-gray-500 text-xs"><span className="text-red-500">*</span> 拓扑配置文件</span>
            <Button
              variant="link"
              size="sm"
              className="p-0 h-auto text-xs"
              onClick={toggleAllSections}
            >
              {Object.values(openSections).every(v => v) ? (
                <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</>
              ) : (
                <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>
              )}
            </Button>
          </div>
          {topologyConfigs.length === 0 ? (
            <div className="w-full h-8 px-2 flex items-center text-xs text-gray-400 bg-gray-50 border border-gray-200 rounded">
              暂无配置
            </div>
          ) : (
            <Select
              value={selectedConfigName || '__placeholder__'}
              onValueChange={(v) => {
                if (v !== '__placeholder__') {
                  onSelectConfig(v)
                }
              }}
            >
              <SelectTrigger className="w-full h-7">
                <SelectValue placeholder="请选择配置" />
              </SelectTrigger>
              <SelectContent>
                {topologyConfigs.map(c => (
                  <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {!hasConfig ? (
          /* 未配置状态 */
          <div className="p-4 rounded-lg border border-amber-200 bg-amber-50/50 text-center">
            <AlertTriangle className="h-5 w-5 text-amber-500 mx-auto mb-2" />
            <p className="text-xs text-amber-700 mb-2">未找到拓扑配置</p>
            <p className="text-xs text-amber-600">
              请在「互联拓扑」中配置芯片和网络拓扑，或选择已保存的配置文件
            </p>
            {onNavigateToTopology && (
              <Button
                variant="outline"
                size="sm"
                className="mt-3 h-7 text-xs border-amber-300 text-amber-700 hover:bg-amber-100"
                onClick={onNavigateToTopology}
              >
                去配置拓扑
              </Button>
            )}
          </div>
        ) : (
          /* 已配置状态 - 只读展示 */
          <div className="space-y-2">
            {/* 互联拓扑 */}
            <BaseCard
              collapsible
              gradient
              defaultExpanded={false}
              title="互联拓扑"
              expanded={openSections.topology}
              onExpandChange={() => toggleSection('topology')}
              contentClassName="p-2"
            >
              <div className="space-y-2">
                <SectionHeader title="集群规模" />
                <InfoGrid
                  items={[
                    { label: 'Pod', value: topologyStats.podCount },
                    { label: 'Rack', value: topologyStats.rackCount },
                    { label: 'Board', value: topologyStats.boardCount },
                    { label: 'Chip', value: topologyStats.chipCount },
                  ]}
                />

                <SectionHeader title="层级互联" />
                {isEditable ? (
                  <div className="grid grid-cols-2 gap-3">
                    {(['c2c', 'b2b', 'r2r', 'p2p'] as const).map(level => {
                      const levelLabels = { c2c: 'C2C', b2b: 'B2B', r2r: 'R2R', p2p: 'P2P' }
                      const params = hardwareParams?.interconnect[level] || interconnectParams?.[level] || { bandwidth_gbps: 100, latency_us: 1 }
                      return (
                        <div key={level}>
                          <div className="mb-1 text-xs text-gray-600">{levelLabels[level]}</div>
                          <div className="flex gap-2">
                            <div className={`flex-1 p-2 rounded -m-2 mb-0 ${isInterconnectModified(level, 'bandwidth_gbps') ? 'bg-blue-50/50' : ''}`}>
                              <div className="mb-1 flex items-center gap-1.5 text-[11px] text-gray-500">
                                带宽 (GB/s)
                                {isInterconnectModified(level, 'bandwidth_gbps') && (
                                  <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                                )}
                              </div>
                              <NumberInput
                                min={0}
                                value={params.bandwidth_gbps}
                                onChange={(v) => {
                                  if (!hardwareParams) {
                                    // 初始化 hardwareParams
                                    const newParams: HardwareParams = {
                                      chips: {},
                                      interconnect: {
                                        c2c: interconnectParams?.c2c || DEFAULT_HARDWARE_PARAMS.interconnect.c2c,
                                        b2b: interconnectParams?.b2b || DEFAULT_HARDWARE_PARAMS.interconnect.b2b,
                                        r2r: interconnectParams?.r2r || DEFAULT_HARDWARE_PARAMS.interconnect.r2r,
                                        p2p: interconnectParams?.p2p || DEFAULT_HARDWARE_PARAMS.interconnect.p2p,
                                      }
                                    }
                                    newParams.interconnect[level].bandwidth_gbps = v ?? 100
                                    onHardwareParamsChange?.(newParams)
                                  } else {
                                    updateInterconnect(level, 'bandwidth_gbps', v ?? 100)
                                  }
                                }}
                                className="h-7"
                              />
                            </div>
                            <div className={`flex-1 p-2 rounded -m-2 mb-0 ${isInterconnectModified(level, 'latency_us') ? 'bg-blue-50/50' : ''}`}>
                              <div className="mb-1 flex items-center gap-1.5 text-[11px] text-gray-500">
                                延迟 (µs)
                                {isInterconnectModified(level, 'latency_us') && (
                                  <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                                )}
                              </div>
                              <NumberInput
                                min={0}
                                step={0.01}
                                value={params.latency_us}
                                onChange={(v) => {
                                  if (!hardwareParams) {
                                    // 初始化 hardwareParams
                                    const newParams: HardwareParams = {
                                      chips: {},
                                      interconnect: {
                                        c2c: interconnectParams?.c2c || DEFAULT_HARDWARE_PARAMS.interconnect.c2c,
                                        b2b: interconnectParams?.b2b || DEFAULT_HARDWARE_PARAMS.interconnect.b2b,
                                        r2r: interconnectParams?.r2r || DEFAULT_HARDWARE_PARAMS.interconnect.r2r,
                                        p2p: interconnectParams?.p2p || DEFAULT_HARDWARE_PARAMS.interconnect.p2p,
                                      }
                                    }
                                    newParams.interconnect[level].latency_us = v ?? 1
                                    onHardwareParamsChange?.(newParams)
                                  } else {
                                    updateInterconnect(level, 'latency_us', v ?? 1)
                                  }
                                }}
                                className="h-7"
                              />
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <InfoGrid
                    items={[
                      {
                        label: 'C2C (芯片间)',
                        value: `${formatBandwidth(interconnectParams?.c2c?.bandwidth_gbps)} / ${formatNumber(interconnectParams?.c2c?.latency_us)} µs`
                      },
                      {
                        label: 'B2B (板间)',
                        value: `${formatBandwidth(interconnectParams?.b2b?.bandwidth_gbps)} / ${formatNumber(interconnectParams?.b2b?.latency_us)} µs`
                      },
                      {
                        label: 'R2R (机架间)',
                        value: `${formatBandwidth(interconnectParams?.r2r?.bandwidth_gbps)} / ${formatNumber(interconnectParams?.r2r?.latency_us)} µs`
                      },
                      {
                        label: 'P2P (Pod间)',
                        value: `${formatBandwidth(interconnectParams?.p2p?.bandwidth_gbps)} / ${formatNumber(interconnectParams?.p2p?.latency_us)} µs`
                      },
                    ]}
                    columns={2}
                  />
                )}
              </div>
            </BaseCard>

            {/* 芯片参数 - 显示所有拓扑中使用的芯片 (Tier6 格式) */}
            {chipGroups.map((chipGroup, chipIndex) => {
              const chipName = chipGroup.chipType
              const sectionKey = `chip_${chipIndex}`
              // 获取芯片参数（可编辑模式从 hardwareParams 获取，否则从 hardwareConfig 获取）
              const chipParams = isEditable && hardwareParams?.chips[chipName]
                ? hardwareParams.chips[chipName] as ChipPreset
                : ((chipGroup.chipConfig || createDefaultChipPreset(chipName)) as ChipPreset)

                return (
                  <BaseCard
                    key={chipName}
                    collapsible
                    gradient
                    defaultExpanded={false}
                    title={`芯片参数: ${chipName}`}
                    expanded={openSections[sectionKey]}
                    onExpandChange={() => toggleSection(sectionKey)}
                    contentClassName="p-2"
                  >
                    <div className="space-y-2">
                      {/* 基础信息 */}
                      <SectionHeader title="基础信息" color="blue" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'architecture') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">架构</div>
                            <Input
                              value={chipParams.architecture || ''}
                              onChange={(e) => updateChipParam(chipName, 'architecture', e.target.value)}
                              className="h-7 text-xs"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'process') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">工艺</div>
                            <Input
                              value={chipParams.process || ''}
                              onChange={(e) => updateChipParam(chipName, 'process', e.target.value)}
                              className="h-7 text-xs"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'frequency_ghz') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">频率 (GHz)</div>
                            <NumberInput
                              min={0.1}
                              step={0.1}
                              value={chipParams.frequency_ghz}
                              onChange={(v) => updateChipParam(chipName, 'frequency_ghz', v ?? 1.0)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '架构', value: chipParams.architecture || '-' },
                            { label: '工艺', value: chipParams.process || '-' },
                            { label: '频率', value: chipParams.frequency_ghz ? `${chipParams.frequency_ghz} GHz` : '-' },
                          ]}
                          columns={3}
                        />
                      )}

                      {/* 核心配置 */}
                      <SectionHeader title="核心配置" color="blue" />
                      {isEditable ? (
                        <div className="grid grid-cols-2 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'cores.count') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">核心数</div>
                            <NumberInput
                              min={1}
                              value={chipParams.cores?.count}
                              onChange={(v) => updateChipParam(chipName, 'cores.count', v ?? 4)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'cores.lanes_per_core') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">Lane/核心</div>
                            <NumberInput
                              min={1}
                              value={chipParams.cores?.lanes_per_core}
                              onChange={(v) => updateChipParam(chipName, 'cores.lanes_per_core', v ?? 64)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '核心数', value: chipParams.cores?.count || '-' },
                            { label: 'Lane/核心', value: chipParams.cores?.lanes_per_core || '-' },
                          ]}
                          columns={2}
                        />
                      )}

                      {/* Cube MAC/Lane */}
                      <SectionHeader title="Cube MAC/Lane" color="purple" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          {['INT8', 'FP8', 'BF16', 'FP16', 'TF32', 'INT4'].map(dtype => (
                            <div key={dtype} className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, `compute_units.cube.mac_per_lane.${dtype}`) ? 'bg-blue-50/50' : ''}`}>
                              <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">{dtype}</div>
                              <NumberInput
                                min={0}
                                value={chipParams.compute_units?.cube?.mac_per_lane?.[dtype as keyof typeof chipParams.compute_units.cube.mac_per_lane]}
                                onChange={(v) => updateChipParam(chipName, `compute_units.cube.mac_per_lane.${dtype}`, v ?? 0)}
                                className="h-7"
                              />
                            </div>
                          ))}
                        </div>
                      ) : (
                        <InfoGrid
                          items={['INT8', 'FP8', 'BF16', 'FP16', 'TF32', 'INT4'].map(dtype => ({
                            label: dtype,
                            value: chipParams.compute_units?.cube?.mac_per_lane?.[dtype as keyof typeof chipParams.compute_units.cube.mac_per_lane] || '-'
                          }))}
                          columns={3}
                        />
                      )}

                      {/* GMEM */}
                      <SectionHeader title="GMEM (全局内存)" color="green" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.gmem.capacity_gb') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">容量 (GB)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.memory?.gmem?.capacity_gb}
                              onChange={(v) => updateChipParam(chipName, 'memory.gmem.capacity_gb', v ?? 64)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.gmem.bandwidth_gbps') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">带宽 (GB/s)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.memory?.gmem?.bandwidth_gbps}
                              onChange={(v) => updateChipParam(chipName, 'memory.gmem.bandwidth_gbps', v ?? 273)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.gmem.latency_ns') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">延迟 (ns)</div>
                            <NumberInput
                              min={0}
                              value={chipParams.memory?.gmem?.latency_ns}
                              onChange={(v) => updateChipParam(chipName, 'memory.gmem.latency_ns', v ?? 100)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '容量', value: chipParams.memory?.gmem?.capacity_gb ? `${chipParams.memory.gmem.capacity_gb} GB` : '-' },
                            { label: '带宽', value: chipParams.memory?.gmem?.bandwidth_gbps ? `${chipParams.memory.gmem.bandwidth_gbps} GB/s` : '-' },
                            { label: '延迟', value: chipParams.memory?.gmem?.latency_ns ? `${chipParams.memory.gmem.latency_ns} ns` : '-' },
                          ]}
                          columns={3}
                        />
                      )}

                      {/* LMEM */}
                      <SectionHeader title="LMEM (本地内存)" color="green" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.lmem.capacity_mb') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">容量 (MB)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.memory?.lmem?.capacity_mb}
                              onChange={(v) => updateChipParam(chipName, 'memory.lmem.capacity_mb', v ?? 64)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.lmem.bandwidth_gbps') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">带宽 (GB/s)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.memory?.lmem?.bandwidth_gbps}
                              onChange={(v) => updateChipParam(chipName, 'memory.lmem.bandwidth_gbps', v ?? 2000)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'memory.lmem.latency_ns') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">延迟 (ns)</div>
                            <NumberInput
                              min={0}
                              value={chipParams.memory?.lmem?.latency_ns}
                              onChange={(v) => updateChipParam(chipName, 'memory.lmem.latency_ns', v ?? 1)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '容量', value: chipParams.memory?.lmem?.capacity_mb ? `${chipParams.memory.lmem.capacity_mb} MB` : '-' },
                            { label: '带宽', value: chipParams.memory?.lmem?.bandwidth_gbps ? `${chipParams.memory.lmem.bandwidth_gbps} GB/s` : '-' },
                            { label: '延迟', value: chipParams.memory?.lmem?.latency_ns ? `${chipParams.memory.lmem.latency_ns} ns` : '-' },
                          ]}
                          columns={3}
                        />
                      )}

                      {/* DMA 引擎 - GDMA */}
                      <SectionHeader title="DMA 引擎 - GDMA" color="orange" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'dma_engines.gdma.bandwidth_gbps') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">带宽 (GB/s)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.dma_engines?.gdma?.bandwidth_gbps}
                              onChange={(v) => updateChipParam(chipName, 'dma_engines.gdma.bandwidth_gbps', v ?? 68)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'dma_engines.gdma.startup_latency_ns') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">启动延迟 (ns)</div>
                            <NumberInput
                              min={0}
                              value={chipParams.dma_engines?.gdma?.startup_latency_ns}
                              onChange={(v) => updateChipParam(chipName, 'dma_engines.gdma.startup_latency_ns', v ?? 100)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'dma_engines.gdma.efficiency') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">效率</div>
                            <NumberInput
                              min={0}
                              max={1}
                              step={0.01}
                              value={chipParams.dma_engines?.gdma?.efficiency}
                              onChange={(v) => updateChipParam(chipName, 'dma_engines.gdma.efficiency', v ?? 0.9)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '带宽', value: chipParams.dma_engines?.gdma?.bandwidth_gbps ? `${chipParams.dma_engines.gdma.bandwidth_gbps} GB/s` : '-' },
                            { label: '启动延迟', value: chipParams.dma_engines?.gdma?.startup_latency_ns ? `${chipParams.dma_engines.gdma.startup_latency_ns} ns` : '-' },
                            { label: '效率', value: formatNumber(chipParams.dma_engines?.gdma?.efficiency) },
                          ]}
                          columns={3}
                        />
                      )}

                      {/* NoC */}
                      <SectionHeader title="片上互联 NoC" color="orange" />
                      {isEditable ? (
                        <div className="grid grid-cols-3 gap-2">
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'interconnect.noc.topology') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">拓扑</div>
                            <Input
                              value={chipParams.interconnect?.noc?.topology || ''}
                              onChange={(e) => updateChipParam(chipName, 'interconnect.noc.topology', e.target.value)}
                              className="h-7 text-xs"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'interconnect.noc.bandwidth_gbps') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">带宽 (GB/s)</div>
                            <NumberInput
                              min={1}
                              value={chipParams.interconnect?.noc?.bandwidth_gbps}
                              onChange={(v) => updateChipParam(chipName, 'interconnect.noc.bandwidth_gbps', v ?? 1000)}
                              className="h-7"
                            />
                          </div>
                          <div className={`p-2 rounded -m-2 mb-0 ${isChipParamModified(chipName, 'interconnect.noc.latency_ns') ? 'bg-blue-50/50' : ''}`}>
                            <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">延迟 (ns)</div>
                            <NumberInput
                              min={0}
                              value={chipParams.interconnect?.noc?.latency_ns}
                              onChange={(v) => updateChipParam(chipName, 'interconnect.noc.latency_ns', v ?? 10)}
                              className="h-7"
                            />
                          </div>
                        </div>
                      ) : (
                        <InfoGrid
                          items={[
                            { label: '拓扑', value: chipParams.interconnect?.noc?.topology || '-' },
                            { label: '带宽', value: chipParams.interconnect?.noc?.bandwidth_gbps ? `${chipParams.interconnect.noc.bandwidth_gbps} GB/s` : '-' },
                            { label: '延迟', value: chipParams.interconnect?.noc?.latency_ns ? `${chipParams.interconnect.noc.latency_ns} ns` : '-' },
                          ]}
                          columns={3}
                        />
                      )}
                    </div>
                  </BaseCard>
                )
              }
            )}

            {/* 通信延迟参数 */}
            {commLatencyConfig && (
              <BaseCard
                collapsible
                gradient
                defaultExpanded={false}
                title="通信延迟"
                expanded={openSections.comm_latency}
                onExpandChange={() => toggleSection('comm_latency')}
                contentClassName="p-2"
              >
                <div className="space-y-2">
                  <SectionHeader title="协议参数" color="blue" />
                  {isEditable && onCommLatencyChange ? (
                    <div className="grid grid-cols-4 gap-2">
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('rtt_tp_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          TP RTT (µs)
                          {isCommLatencyModified('rtt_tp_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.rtt_tp_us}
                          onChange={(v) => updateCommLatency('rtt_tp_us', v ?? 0.35)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('rtt_ep_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          EP RTT (µs)
                          {isCommLatencyModified('rtt_ep_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.rtt_ep_us}
                          onChange={(v) => updateCommLatency('rtt_ep_us', v ?? 0.85)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('bandwidth_utilization') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          带宽利用率
                          {isCommLatencyModified('bandwidth_utilization') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          max={1}
                          step={0.01}
                          value={commLatencyConfig.bandwidth_utilization}
                          onChange={(v) => updateCommLatency('bandwidth_utilization', v ?? 0.95)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('sync_latency_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          同步延迟 (µs)
                          {isCommLatencyModified('sync_latency_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.1}
                          value={commLatencyConfig.sync_latency_us}
                          onChange={(v) => updateCommLatency('sync_latency_us', v ?? 0)}
                          className="h-7"
                        />
                      </div>
                    </div>
                  ) : (
                    <InfoGrid
                      items={[
                        { label: 'TP RTT', value: `${commLatencyConfig.rtt_tp_us} µs` },
                        { label: 'EP RTT', value: `${commLatencyConfig.rtt_ep_us} µs` },
                        { label: '带宽利用率', value: formatNumber(commLatencyConfig.bandwidth_utilization) },
                        { label: '同步延迟', value: `${commLatencyConfig.sync_latency_us} µs` },
                      ]}
                    />
                  )}

                  <SectionHeader title="网络延迟" color="green" />
                  {isEditable && onCommLatencyChange ? (
                    <div className="grid grid-cols-2 gap-2">
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('switch_delay_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          Switch延迟 (µs)
                          {isCommLatencyModified('switch_delay_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.switch_delay_us}
                          onChange={(v) => updateCommLatency('switch_delay_us', v ?? 1.0)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('cable_delay_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          Cable延迟 (µs)
                          {isCommLatencyModified('cable_delay_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.005}
                          value={commLatencyConfig.cable_delay_us}
                          onChange={(v) => updateCommLatency('cable_delay_us', v ?? 0.025)}
                          className="h-7"
                        />
                      </div>
                    </div>
                  ) : (
                    <InfoGrid
                      items={[
                        { label: 'Switch延迟', value: `${commLatencyConfig.switch_delay_us} µs` },
                        { label: 'Cable延迟', value: `${commLatencyConfig.cable_delay_us} µs` },
                      ]}
                      columns={2}
                    />
                  )}

                  <SectionHeader title="芯片延迟" color="purple" />
                  {isEditable && onCommLatencyChange ? (
                    <div className="grid grid-cols-3 gap-2">
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('memory_read_latency_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          Mem读 (µs)
                          {isCommLatencyModified('memory_read_latency_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.memory_read_latency_us}
                          onChange={(v) => updateCommLatency('memory_read_latency_us', v ?? 0.15)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('memory_write_latency_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          Mem写 (µs)
                          {isCommLatencyModified('memory_write_latency_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.memory_write_latency_us}
                          onChange={(v) => updateCommLatency('memory_write_latency_us', v ?? 0.01)}
                          className="h-7"
                        />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isCommLatencyModified('noc_latency_us') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs text-gray-600">
                          NoC (µs)
                          {isCommLatencyModified('noc_latency_us') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput
                          min={0}
                          step={0.01}
                          value={commLatencyConfig.noc_latency_us}
                          onChange={(v) => updateCommLatency('noc_latency_us', v ?? 0.05)}
                          className="h-7"
                        />
                      </div>
                    </div>
                  ) : (
                    <InfoGrid
                      items={[
                        { label: 'C2C', value: `${interconnectParams?.c2c?.latency_us || 0} µs` },
                        { label: 'Mem读', value: `${commLatencyConfig.memory_read_latency_us} µs` },
                        { label: 'Mem写', value: `${commLatencyConfig.memory_write_latency_us} µs` },
                        { label: 'NoC', value: `${commLatencyConfig.noc_latency_us} µs` },
                      ]}
                    />
                  )}

                  <SectionHeader title="启动延迟 (计算值)" color="orange" />
                  <InfoGrid
                    items={[
                      {
                        label: 'AllReduce',
                        value: (
                          <span className="text-blue-600 font-medium">
                            {(2 * (interconnectParams?.c2c?.latency_us || 0) +
                              (commLatencyConfig.memory_read_latency_us || 0) +
                              (commLatencyConfig.memory_write_latency_us || 0) +
                              (commLatencyConfig.noc_latency_us || 0) +
                              2 * (commLatencyConfig.die_to_die_latency_us || 0)).toFixed(2)} µs
                          </span>
                        )
                      },
                      {
                        label: 'Dispatch/Combine',
                        value: (
                          <span className="text-purple-600 font-medium">
                            {(2 * (interconnectParams?.c2c?.latency_us || 0) +
                              (commLatencyConfig.memory_read_latency_us || 0) +
                              (commLatencyConfig.memory_write_latency_us || 0) +
                              (commLatencyConfig.noc_latency_us || 0) +
                              2 * (commLatencyConfig.die_to_die_latency_us || 0) +
                              2 * (commLatencyConfig.switch_delay_us || 0) +
                              2 * (commLatencyConfig.cable_delay_us || 0)).toFixed(2)} µs
                          </span>
                        )
                      },
                    ]}
                    columns={2}
                  />
                </div>
              </BaseCard>
            )}

            {/* 保存/另存为/重置按钮 - 仅可编辑模式显示 */}
            {isEditable && (
              <div className="flex gap-2 mt-3 pt-3 border-t border-gray-100">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onSaveConfig}
                >
                  <Save className="h-3.5 w-3.5 mr-1" />
                  保存
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSaveAs}
                >
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  另存为
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                >
                  <RefreshCw className="h-3.5 w-3.5 mr-1" />
                  重置
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 另存为弹窗 */}
      <Dialog open={saveAsModalOpen} onOpenChange={setSaveAsModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>另存为新配置</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label className="block mb-2">配置名称 <span className="text-red-500">*</span></Label>
              <Input
                value={newConfigName}
                onChange={(e) => setNewConfigName(e.target.value)}
                placeholder="请输入配置名称"
                onKeyDown={(e) => e.key === 'Enter' && handleConfirmSaveAs()}
              />
            </div>
            <div>
              <Label className="block mb-2">描述 (可选)</Label>
              <Textarea
                value={newConfigDesc}
                onChange={(e) => setNewConfigDesc(e.target.value)}
                placeholder="请输入配置描述"
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => {
              setSaveAsModalOpen(false)
              setNewConfigName('')
              setNewConfigDesc('')
            }}>
              取消
            </Button>
            <Button onClick={handleConfirmSaveAs}>
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </BaseCard>
  )
}

export default TopologyInfoCard
