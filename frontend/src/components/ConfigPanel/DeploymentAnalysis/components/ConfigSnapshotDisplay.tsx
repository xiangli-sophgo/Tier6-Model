/**
 * 配置快照展示组件
 * 使用 Accordion 展示完整的配置信息
 */

import React from 'react'
import { FileText, Database, Settings } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

// 描述项组件
const DescItem: React.FC<{ label: string; span?: number; children: React.ReactNode }> = ({
  label,
  span = 1,
  children,
}) => (
  <div className={span === 2 ? 'col-span-2' : ''}>
    <span className="text-gray-500 text-xs">{label}</span>
    <div className="text-sm">{children}</div>
  </div>
)

// 描述列表组件
const DescList: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
  <div className={`grid grid-cols-2 gap-x-4 gap-y-2 p-3 bg-white rounded border ${className || ''}`}>
    {children}
  </div>
)

interface ConfigSnapshotDisplayProps {
  configSnapshot: {
    model: Record<string, unknown>
    inference: Record<string, unknown>
    topology: Record<string, unknown>
  }
  benchmarkName?: string
  topologyConfigName?: string
}

export const ConfigSnapshotDisplay: React.FC<ConfigSnapshotDisplayProps> = ({
  configSnapshot,
  benchmarkName,
  topologyConfigName,
}) => {
  if (!configSnapshot) {
    return (
      <div className="flex flex-col items-center justify-center py-6 text-gray-400">
        <div className="text-sm">配置快照不可用</div>
      </div>
    )
  }

  const { model, inference, topology } = configSnapshot

  // 提取通信延迟配置（新格式: interconnect.comm_params, 旧格式: comm_latency_config）
  const commLatencyConfig = (topology as any).interconnect?.comm_params || (topology as any).comm_latency_config
  // 提取互联参数配置（新格式: interconnect.links, 旧格式: hardware_params.interconnect）
  const interconnectParams = (topology as any).interconnect?.links || (topology as any).hardware_params?.interconnect

  // 计算总芯片数
  const calculateTotalChips = () => {
    const pods = (topology as any).pods || []
    let total = 0
    for (const pod of pods) {
      for (const rack of pod.racks || []) {
        for (const board of rack.boards || []) {
          total += (board.chips || []).length
        }
      }
    }
    return total
  }

  const totalChips = calculateTotalChips()

  return (
    <Accordion type="multiple" defaultValue={['benchmark']} className="bg-gray-50 rounded-lg border">
      {/* Benchmark 配置 */}
      <AccordionItem value="benchmark" className="border-b">
        <AccordionTrigger className="px-4 py-3 hover:bg-gray-100">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-blue-500" />
            <span className="font-semibold">Benchmark 配置</span>
            {benchmarkName && (
              <Badge variant="outline" className="ml-2 bg-blue-50 text-blue-700 border-blue-200">
                {benchmarkName}
              </Badge>
            )}
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-4 pb-4">
          <DescList>
            <DescItem label="模型名称" span={2}>{(model as any).model_name || '-'}</DescItem>
            <DescItem label="层数">{(model as any).num_layers || '-'}</DescItem>
            <DescItem label="隐藏维度">{(model as any).hidden_size || '-'}</DescItem>
            <DescItem label="注意力头数">{(model as any).num_attention_heads || '-'}</DescItem>
            <DescItem label="KV头数">{(model as any).num_key_value_heads || '-'}</DescItem>
            <DescItem label="中间层维度">{(model as any).intermediate_size || '-'}</DescItem>
            <DescItem label="词汇表大小">{(model as any).vocab_size || '-'}</DescItem>
            <DescItem label="批次大小">{(inference as any).batch_size || '-'}</DescItem>
            <DescItem label="输入序列长度">{(inference as any).input_seq_length || '-'}</DescItem>
            <DescItem label="输出序列长度">{(inference as any).output_seq_length || '-'}</DescItem>
            <DescItem label="推理模式">{(inference as any).mode || 'decode'}</DescItem>
          </DescList>
        </AccordionContent>
      </AccordionItem>

      {/* 拓扑配置 */}
      <AccordionItem value="topology" className="border-b">
        <AccordionTrigger className="px-4 py-3 hover:bg-gray-100">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-green-500" />
            <span className="font-semibold">拓扑配置</span>
            {topologyConfigName && (
              <Badge variant="outline" className="ml-2 bg-green-50 text-green-700 border-green-200">
                {topologyConfigName}
              </Badge>
            )}
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-4 pb-4">
          <DescList>
            <DescItem label="Pod 数量">{((topology as any).pods || []).length}</DescItem>
            <DescItem label="总芯片数">
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">{totalChips}</Badge>
            </DescItem>
            <DescItem label="Rack 数量">
              {((topology as any).pods || []).reduce((sum: number, pod: any) => sum + (pod.racks || []).length, 0)}
            </DescItem>
            <DescItem label="Board 数量">
              {((topology as any).pods || []).reduce(
                (sum: number, pod: any) =>
                  sum + (pod.racks || []).reduce((s: number, rack: any) => s + (rack.boards || []).length, 0),
                0
              )}
            </DescItem>
          </DescList>

          {/* 芯片硬件信息（如果有） */}
          {(() => {
            const pods = (topology as any).pods || []
            const topologyChips = (topology as any).chips || (topology as any).hardware_params?.chips
            if (pods.length > 0 && pods[0].racks && pods[0].racks[0].boards && pods[0].racks[0].boards[0].chips) {
              const chipInfo = pods[0].racks[0].boards[0].chips[0]
              // 从顶层 chips 获取详细芯片配置 (Tier6 ChipPreset 格式)
              const chipConfig = topologyChips?.[chipInfo.name]
              return (
                <DescList className="mt-3">
                  <DescItem label="芯片型号" span={2}>{chipInfo.name || '-'}</DescItem>
                  <DescItem label="核心数">{chipConfig?.cores?.count || '-'}</DescItem>
                  <DescItem label="Lane/核心">{chipConfig?.cores?.lanes_per_core || '-'}</DescItem>
                  <DescItem label="GMEM 容量">{chipConfig?.memory?.gmem?.capacity_gb ? `${chipConfig.memory.gmem.capacity_gb} GB` : '-'}</DescItem>
                  <DescItem label="GMEM 带宽">{chipConfig?.memory?.gmem?.bandwidth_gbps ? `${chipConfig.memory.gmem.bandwidth_gbps} GB/s` : '-'}</DescItem>
                </DescList>
              )
            }
            return null
          })()}
        </AccordionContent>
      </AccordionItem>

      {/* 通信延迟配置 */}
      {commLatencyConfig && (
        <AccordionItem value="comm_latency">
          <AccordionTrigger className="px-4 py-3 hover:bg-gray-100">
            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 text-purple-500" />
              <span className="font-semibold">通信延迟配置</span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-4 pb-4">
            <DescList>
              <DescItem label="TP RTT (μs)">{commLatencyConfig.rtt_tp_us ?? '-'}</DescItem>
              <DescItem label="EP RTT (μs)">{commLatencyConfig.rtt_ep_us ?? '-'}</DescItem>
              <DescItem label="带宽利用率">
                {commLatencyConfig.bandwidth_utilization
                  ? `${(commLatencyConfig.bandwidth_utilization * 100).toFixed(0)}%`
                  : '-'}
              </DescItem>
              <DescItem label="同步延迟 (μs)">{commLatencyConfig.sync_latency_us ?? '-'}</DescItem>
              <DescItem label="交换机延迟 (μs)">{commLatencyConfig.switch_delay_us ?? '-'}</DescItem>
              <DescItem label="线缆延迟 (μs)">{commLatencyConfig.cable_delay_us ?? '-'}</DescItem>
              <DescItem label="芯片间延迟 (μs)">{interconnectParams?.c2c?.latency_us ?? '-'}</DescItem>
              <DescItem label="内存读延迟 (μs)">{commLatencyConfig.memory_read_latency_us ?? '-'}</DescItem>
              <DescItem label="内存写延迟 (μs)">{commLatencyConfig.memory_write_latency_us ?? '-'}</DescItem>
              <DescItem label="NoC延迟 (μs)">{commLatencyConfig.noc_latency_us ?? '-'}</DescItem>
              <DescItem label="Die间延迟 (μs)" span={2}>{commLatencyConfig.die_to_die_latency_us ?? '-'}</DescItem>
            </DescList>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  )
}
