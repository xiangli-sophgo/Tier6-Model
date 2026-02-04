/**
 * 算子时间分解图 - Treemap 矩形面积占比图
 *
 * 展示不同算子类型的时间占比，使用矩形面积表示
 */

import React, { useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { GanttChartData } from '../../../../utils/llmDeployment/types'

interface OperatorTimeBreakdownChartProps {
  data: GanttChartData | null
  height?: number
}

/** 算子类型颜色映射 - 超低饱和度浅色配色，与主题色 #5E6AD2 搭配 */
const OPERATOR_COLORS: Record<string, string> = {
  // 计算类算子 - 浅蓝紫色系（接近主题色）
  compute: '#A5AEE8',           // 浅蓝紫
  embedding: '#B8C0ED',         // 更浅蓝紫
  layernorm: '#CAD0F2',         // 极浅蓝紫
  attention_qkv: '#9DA7E3',     // 稍深蓝紫
  attention_score: '#AFB8EA',   // 浅蓝紫
  attention_softmax: '#C0C8EF',  // 更浅蓝紫
  attention_output: '#D1D8F4',  // 极浅蓝紫
  ffn_gate: '#98A2E0',          // 稍深蓝紫
  ffn_up: '#A9B3E6',            // 浅蓝紫
  ffn_down: '#BAC3EC',          // 更浅蓝紫
  lm_head: '#8F99DC',           // 中蓝紫

  // MLA - 浅紫色系
  rmsnorm_q_lora: '#C4B8E8',    // 浅紫
  rmsnorm_kv_lora: '#D0C7ED',   // 更浅紫
  mm_q_lora_a: '#DBD5F2',       // 极浅紫
  mm_q_lora_b: '#E6E0F7',       // 淡紫
  mm_kv_lora_a: '#EFEAF9',      // 极淡紫
  attn_fc: '#BDB0E5',           // 浅紫
  bmm_qk: '#B5A9E0',            // 稍深浅紫
  bmm_sv: '#AFA3DD',            // 中浅紫

  // MoE - 浅橙色系（互补色）
  moe_gate: '#F0C4B8',          // 浅橙
  moe_expert: '#F5D0C7',        // 更浅橙
  moe_shared_expert: '#FADCD5', // 极浅橙

  // 访存 - 浅金色系
  weight_load: '#F0E0C4',       // 浅金
  kv_cache_read: '#F5E8D5',     // 更浅金
  kv_cache_write: '#FAF0E5',    // 极浅金
  hbm_read: '#FDF5ED',          // 淡金
  hbm_write: '#F7EBDC',         // 中浅金

  // 通信 - 浅蓝色系（与主题色同色系）
  tp_comm: '#B8D5E8',           // 浅蓝
  pp_comm: '#C4B8E8',           // 浅紫（与MLA呼应）
  ep_comm: '#F0C4B8',           // 浅橙（与MoE呼应）
  ep_dispatch: '#D0C7ED',       // 更浅紫
  ep_combine: '#C9BFE9',        // 浅紫
  sp_allgather: '#AECCE0',      // 稍深浅蓝
  sp_reduce_scatter: '#A5C3D9', // 中浅蓝
}

/** 算子类型标签映射 */
const OPERATOR_LABELS: Record<string, string> = {
  compute: '通用计算',
  embedding: 'Embedding',
  layernorm: 'LayerNorm',
  attention_qkv: 'Attn QKV',
  attention_score: 'Attn Score',
  attention_softmax: 'Attn Softmax',
  attention_output: 'Attn Out',
  ffn_gate: 'FFN Gate',
  ffn_up: 'FFN Up',
  ffn_down: 'FFN Down',
  lm_head: 'LM Head',

  rmsnorm_q_lora: 'MLA RMS Q',
  rmsnorm_kv_lora: 'MLA RMS KV',
  mm_q_lora_a: 'MLA Q↓',
  mm_q_lora_b: 'MLA Q↑',
  mm_kv_lora_a: 'MLA KV',
  attn_fc: 'MLA FC',
  bmm_qk: 'MLA QK',
  bmm_sv: 'MLA SV',

  moe_gate: 'MoE Gate',
  moe_expert: 'MoE Expert',
  moe_shared_expert: 'Shared Expert',

  weight_load: 'Weight Load',
  kv_cache_read: 'KV Read',
  kv_cache_write: 'KV Write',
  hbm_read: 'HBM Read',
  hbm_write: 'HBM Write',

  tp_comm: 'TP',
  pp_comm: 'PP',
  ep_comm: 'EP',
  ep_dispatch: 'EP Dispatch',
  ep_combine: 'EP Combine',
  sp_allgather: 'SP AG',
  sp_reduce_scatter: 'SP RS',
}

interface TreemapNode {
  name: string
  value: number
  itemStyle?: { color: string }
  children?: TreemapNode[]
}

/** 聚合算子数据并构建 Treemap 树 */
function buildOperatorTreemap(
  tasks: GanttChartData['tasks'],
  phase?: 'prefill' | 'decode'
): TreemapNode {
  const operatorMap = new Map<string, number>()

  for (const task of tasks) {
    // 过滤阶段
    if (phase && task.phase !== phase) continue

    const opType = task.type
    const duration = (task.end - task.start) * 1000 // ms to us

    const current = operatorMap.get(opType) || 0
    operatorMap.set(opType, current + duration)
  }

  // 构建子节点
  const children: TreemapNode[] = []
  for (const [opType, time] of operatorMap.entries()) {
    children.push({
      name: OPERATOR_LABELS[opType] || opType,
      value: time,
      itemStyle: {
        color: OPERATOR_COLORS[opType] || '#999',
      },
    })
  }

  // 按时间降序排序
  children.sort((a, b) => b.value - a.value)

  // 计算总时间
  const totalTime = Array.from(operatorMap.values()).reduce((sum, t) => sum + t, 0)

  return {
    name: '算子总时间',
    value: totalTime,
    children,
  }
}

export const OperatorTimeBreakdownChart: React.FC<OperatorTimeBreakdownChartProps> = ({
  data,
  height = 450,
}) => {
  const [selectedPhase, setSelectedPhase] = useState<'prefill' | 'decode' | 'all'>('all')

  // 构建 Treemap 数据
  const treemapData = useMemo(() => {
    if (!data || data.tasks.length === 0) return null
    const phase = selectedPhase === 'all' ? undefined : selectedPhase
    return buildOperatorTreemap(data.tasks, phase)
  }, [data, selectedPhase])

  const option: EChartsOption = useMemo(() => {
    if (!treemapData || !treemapData.children || treemapData.children.length === 0) {
      return {
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'center',
          textStyle: { color: '#999', fontSize: 12 },
        },
      }
    }

    return {
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        borderColor: 'transparent',
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        padding: [10, 14],
        confine: true,
        formatter: (params: any) => {
          const { name, treePathInfo } = params
          const rootValue = treePathInfo[0]?.value || 0

          // 格式化时间函数
          const formatTime = (value: number): string => {
            if (value < 1000) {
              return `${value.toFixed(0)} µs`
            } else if (value < 1000000) {
              return `${(value / 1000).toFixed(2)} ms`
            } else {
              return `${(value / 1000000).toFixed(2)} s`
            }
          }

          // 显示所有算子的时间
          if (!treemapData.children) return ''

          let html = '<div style="font-size: 12px; max-height: 400px; overflow-y: auto;">'
          html += '<div style="font-weight: 600; margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.2);">算子时间分解</div>'

          // 按时间排序（已经排序过了）
          treemapData.children.forEach((child) => {
            const percentage = ((child.value / rootValue) * 100).toFixed(1)
            const timeStr = formatTime(child.value)
            const color = child.itemStyle?.color || '#999'

            // 判断是否为当前悬停的算子
            const isHovered = child.name === name
            const bgColor = isHovered ? 'rgba(24, 144, 255, 0.2)' : 'transparent'
            const fontWeight = isHovered ? '600' : '400'

            html += `
              <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px; padding: 4px 6px; border-radius: 4px; background: ${bgColor}; font-weight: ${fontWeight};">
                <span style="width: 10px; height: 10px; border-radius: 2px; background: ${color}; flex-shrink: 0;"></span>
                <span style="flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${child.name}</span>
                <span style="color: rgba(255,255,255,0.7); white-space: nowrap;">${timeStr} (${percentage}%)</span>
              </div>
            `
          })

          html += '</div>'
          return html
        },
      },
      series: [
        {
          type: 'treemap',
          data: [treemapData],
          // 禁用缩放和点击下钻
          roam: false,
          nodeClick: false,
          // 面包屑导航
          breadcrumb: {
            show: false,
          },
          // 标签配置
          label: {
            show: true,
            formatter: (params: any) => {
              const { name, value, treePathInfo } = params
              const rootValue = treePathInfo[0]?.value || value
              const percentageNum = (value / rootValue) * 100
              const percentage = percentageNum.toFixed(1)

              // 格式化时间
              let timeStr = ''
              if (value < 1000) {
                timeStr = `${value.toFixed(0)}µs`
              } else if (value < 1000000) {
                timeStr = `${(value / 1000).toFixed(1)}ms`
              } else {
                timeStr = `${(value / 1000000).toFixed(1)}s`
              }

              // 根据矩形大小决定显示内容
              if (percentageNum < 2) {
                return '' // 太小不显示
              } else if (percentageNum < 5) {
                return `{name|${name}}` // 只显示名称
              } else {
                return `{name|${name}}\n{value|${timeStr}}\n{percent|${percentage}%}` // 显示完整信息
              }
            },
            rich: {
              name: {
                fontSize: 13,
                fontWeight: 600,
                color: '#333',
              },
              value: {
                fontSize: 12,
                color: '#666',
              },
              percent: {
                fontSize: 11,
                color: '#999',
              },
            },
          },
          // 层级配置
          levels: [
            {
              // 根节点
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 2,
                gapWidth: 2,
              },
              upperLabel: {
                show: false,
              },
            },
            {
              // 算子节点
              itemStyle: {
                borderColor: '#fff',
                borderWidth: 1,
                gapWidth: 1,
              },
            },
          ],
        },
      ],
    }
  }, [treemapData])

  if (!data || data.tasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-gray-400">
        <div className="text-sm">运行模拟以生成算子时间分解</div>
      </div>
    )
  }

  return (
    <div>
      {/* 工具栏 */}
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="text-[11px] text-gray-500">
          矩形面积代表时间占比 | 鼠标悬停查看详情
        </span>
        <Select
          value={selectedPhase}
          onValueChange={(value) => setSelectedPhase(value as 'prefill' | 'decode' | 'all')}
        >
          <SelectTrigger className="w-[100px] h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">全部阶段</SelectItem>
            <SelectItem value="prefill">Prefill</SelectItem>
            <SelectItem value="decode">Decode</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Treemap 图表 */}
      <ReactECharts
        option={option}
        style={{ height: `${height}px`, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}

export default OperatorTimeBreakdownChart
