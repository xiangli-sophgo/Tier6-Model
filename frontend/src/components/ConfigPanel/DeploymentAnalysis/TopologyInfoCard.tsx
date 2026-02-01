/**
 * 拓扑信息卡片 - 只读展示组件
 *
 * 用于在部署分析面板中展示当前选中的拓扑配置信息
 * 只提供展示和配置文件选择，不提供修改功能
 *
 * 设计参考: TaskDetailPanel 的 InfoItem + InfoGrid 风格
 */

import React from 'react'
import {
  ExternalLink,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Pencil,
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { BaseCard } from '@/components/common/BaseCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { SavedConfig } from '../../../api/topology'
import { HardwareConfig, CommLatencyConfig } from '../../../utils/llmDeployment/types'
import { ChipGroupInfo } from '../../../utils/llmDeployment/topologyHardwareExtractor'

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

// 分组标题（带颜色边框）
const SectionHeader: React.FC<{
  title: string
  color?: string
}> = ({ title, color = 'purple' }) => {
  const colorMap: Record<string, string> = {
    purple: 'bg-purple-50 border-purple-400',
    blue: 'bg-blue-50 border-blue-400',
    green: 'bg-green-50 border-green-400',
    orange: 'bg-orange-50 border-orange-400',
  }
  return (
    <h5 className={`text-[10px] font-semibold text-gray-600 mb-1.5 uppercase tracking-wide px-2 py-1 ${colorMap[color] || colorMap.purple} border-l-2`}>
      {title}
    </h5>
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
  /** 硬件配置 */
  hardwareConfig: HardwareConfig | null
  /** 拓扑层级信息 */
  topologyStats: {
    podCount: number
    rackCount: number
    boardCount: number
    chipCount: number
  }
  /** 互联参数（从 hardware_params.interconnect 获取） */
  interconnectParams?: {
    c2c?: { bandwidth_gbps: number; latency_us: number }
    b2b?: { bandwidth_gbps: number; latency_us: number }
    r2r?: { bandwidth_gbps: number; latency_us: number }
    p2p?: { bandwidth_gbps: number; latency_us: number }
  }
  /** 通信延迟配置 */
  commLatencyConfig?: CommLatencyConfig
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
}) => {
  const hasConfig = chipGroups.length > 0 && hardwareConfig

  return (
    <Card className="bg-white/90 backdrop-blur-sm border-purple-100">
      <CardHeader className="py-2.5 px-3 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-gray-700">拓扑配置</CardTitle>
          {onNavigateToTopology && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-purple-600 hover:text-purple-700 hover:bg-purple-50"
              onClick={onNavigateToTopology}
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              去编辑
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-3 space-y-3">
        {/* 配置文件选择器 */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 whitespace-nowrap">配置文件</span>
          <Select
            value={selectedConfigName || '__current__'}
            onValueChange={(v) => onSelectConfig(v === '__current__' ? undefined : v)}
          >
            <SelectTrigger className="flex-1 h-7 text-xs">
              <SelectValue placeholder="使用当前拓扑" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__current__">
                <span className="text-gray-500">使用当前拓扑</span>
              </SelectItem>
              {topologyConfigs.map(c => (
                <SelectItem key={c.name} value={c.name}>{c.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
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
            {/* 芯片类型选择（如果有多种芯片） */}
            {chipGroups.length > 1 && (
              <div className="flex items-center gap-2 pb-2 border-b border-gray-100">
                <span className="text-xs text-gray-500 whitespace-nowrap">分析芯片</span>
                <Select
                  value={selectedChipType || ''}
                  onValueChange={(v) => onSelectChipType?.(v)}
                >
                  <SelectTrigger className="flex-1 h-7 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {chipGroups.map(g => (
                      <SelectItem key={g.presetId || g.chipType} value={g.presetId || g.chipType}>
                        {g.chipType} ({g.totalCount}个)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* 集群规模 */}
            <BaseCard collapsible glassmorphism
              title="集群规模"
              defaultOpen
              collapsibleCount={4}
              onEdit={onNavigateToTopology}
            >
              <InfoGrid
                items={[
                  { label: 'Pod', value: topologyStats.podCount },
                  { label: 'Rack', value: topologyStats.rackCount },
                  { label: 'Board', value: topologyStats.boardCount },
                  { label: 'Chip', value: topologyStats.chipCount },
                ]}
              />
            </BaseCard>

            {/* 芯片规格 */}
            <BaseCard collapsible glassmorphism
              title={`芯片规格: ${hardwareConfig.chip.name}`}
              collapsibleCount={6}
              onEdit={onNavigateToTopology}
            >
              <div className="space-y-2">
                <SectionHeader title="算力" color="blue" />
                <InfoGrid
                  items={[
                    { label: '核心数', value: hardwareConfig.chip.num_cores },
                    { label: 'BF16 算力', value: `${hardwareConfig.chip.compute_tflops_bf16} TFLOPS` },
                    { label: 'FP8 算力', value: `${hardwareConfig.chip.compute_tflops_fp8} TFLOPS` },
                  ]}
                  columns={3}
                />

                <SectionHeader title="显存" color="green" />
                <InfoGrid
                  items={[
                    { label: '容量', value: `${hardwareConfig.chip.memory_capacity_gb} GB` },
                    { label: '带宽', value: `${(hardwareConfig.chip.memory_bandwidth_gbps / 1000).toFixed(1)} TB/s` },
                    { label: '利用率', value: formatNumber(hardwareConfig.chip.memory_bandwidth_utilization) },
                  ]}
                  columns={3}
                />

                {/* LMEM */}
                {hardwareConfig.chip.lmem_capacity_mb && hardwareConfig.chip.lmem_capacity_mb > 0 && (
                  <>
                    <SectionHeader title="LMEM" color="purple" />
                    <InfoGrid
                      items={[
                        { label: '容量', value: `${hardwareConfig.chip.lmem_capacity_mb} MB` },
                        { label: '带宽', value: `${hardwareConfig.chip.lmem_bandwidth_gbps} GB/s` },
                      ]}
                      columns={2}
                    />
                  </>
                )}

                {/* 微架构参数 */}
                {hardwareConfig.chip.cube_m && (
                  <>
                    <SectionHeader title="微架构" color="orange" />
                    <InfoGrid
                      items={[
                        { label: 'Cube M', value: hardwareConfig.chip.cube_m },
                        { label: 'Cube K', value: hardwareConfig.chip.cube_k },
                        { label: 'Cube N', value: hardwareConfig.chip.cube_n },
                        { label: 'Lane', value: hardwareConfig.chip.lane_num },
                      ]}
                    />
                    <InfoGrid
                      items={[
                        { label: 'SRAM', value: hardwareConfig.chip.sram_size_kb ? `${hardwareConfig.chip.sram_size_kb} KB` : '-' },
                        { label: 'SRAM利用', value: formatNumber(hardwareConfig.chip.sram_utilization) },
                        { label: '重叠率', value: formatNumber(hardwareConfig.chip.compute_dma_overlap_rate) },
                      ]}
                      columns={3}
                    />
                  </>
                )}
              </div>
            </BaseCard>

            {/* 互联配置 */}
            <BaseCard collapsible glassmorphism title="层级互联" collapsibleCount={4} onEdit={onNavigateToTopology}>
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
            </BaseCard>

            {/* 通信延迟参数 */}
            {commLatencyConfig && (
              <BaseCard collapsible glassmorphism title="通信延迟" collapsibleCount={10} onEdit={onNavigateToTopology}>
                <div className="space-y-2">
                  <SectionHeader title="协议参数" color="blue" />
                  <InfoGrid
                    items={[
                      { label: 'TP RTT', value: `${commLatencyConfig.rtt_tp_us} µs` },
                      { label: 'EP RTT', value: `${commLatencyConfig.rtt_ep_us} µs` },
                      { label: '带宽利用率', value: formatNumber(commLatencyConfig.bandwidth_utilization) },
                      { label: '同步延迟', value: `${commLatencyConfig.sync_latency_us} µs` },
                    ]}
                  />

                  <SectionHeader title="网络延迟" color="green" />
                  <InfoGrid
                    items={[
                      { label: 'Switch延迟', value: `${commLatencyConfig.switch_delay_us} µs` },
                      { label: 'Cable延迟', value: `${commLatencyConfig.cable_delay_us} µs` },
                    ]}
                    columns={2}
                  />

                  <SectionHeader title="芯片延迟" color="purple" />
                  <InfoGrid
                    items={[
                      { label: 'C2C', value: `${commLatencyConfig.chip_to_chip_us} µs` },
                      { label: 'Mem读', value: `${commLatencyConfig.memory_read_latency_us} µs` },
                      { label: 'Mem写', value: `${commLatencyConfig.memory_write_latency_us} µs` },
                      { label: 'NoC', value: `${commLatencyConfig.noc_latency_us} µs` },
                    ]}
                  />

                  <SectionHeader title="启动延迟 (计算值)" color="orange" />
                  <InfoGrid
                    items={[
                      {
                        label: 'AllReduce',
                        value: (
                          <span className="text-blue-600 font-medium">
                            {(2 * (commLatencyConfig.chip_to_chip_us || 0) +
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
                            {(2 * (commLatencyConfig.chip_to_chip_us || 0) +
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
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default TopologyInfoCard
