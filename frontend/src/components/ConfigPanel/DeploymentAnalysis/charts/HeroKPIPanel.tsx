/**
 * Hero KPI 面板 - 顶部关键指标卡片组
 */

import React, { useMemo } from 'react'
import { Clock, Zap, Gauge, Rocket } from 'lucide-react'
import { conditionalTooltip } from '@/components/ui/info-tooltip'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'
import { COLORS } from '../../../../utils/design-tokens'
import { formatNumber, getMetricDecimals } from '../../../../utils/formatters'

interface HeroKPIPanelProps {
  result: PlanAnalysisResult
  selectedMetric: string | null
  onMetricClick: (metric: string) => void
}

const colors = {
  primary: COLORS.brand.primary.main,
  primaryLight: COLORS.brand.primary.light,
  success: COLORS.semantic.success.main,
  warning: COLORS.semantic.warning.main,
  border: COLORS.border.light,
  text: COLORS.text.primary,
  textSecondary: COLORS.text.secondary,
}

interface KPICardProps {
  id: string
  icon: React.ReactNode
  label: string
  value: string
  unit: string
  subValue?: string
  isSelected: boolean
  onClick: () => void
  status?: 'good' | 'warning' | 'bad'
  tooltip?: string
}

const KPICard: React.FC<KPICardProps> = ({
  icon,
  label,
  value,
  unit,
  subValue,
  isSelected,
  onClick,
  status = 'good',
  tooltip,
}) => {
  const statusColors = {
    good: colors.success,
    warning: colors.warning,
    bad: COLORS.semantic.error.main,
  }

  const card = (
    <div
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: 140,
        padding: '14px 16px',
        background: isSelected ? colors.primaryLight : '#fff',
        borderRadius: 10,
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        border: isSelected ? `1.5px solid ${colors.primary}` : `1px solid ${colors.border}`,
        boxShadow: isSelected
          ? `0 4px 12px rgba(94, 106, 210, 0.15)`
          : '0 1px 3px rgba(0,0,0,0.04)',
      }}
    >
      {/* 标题行 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <span style={{ color: isSelected ? colors.primary : colors.textSecondary, fontSize: 14 }}>
          {icon}
        </span>
        <span style={{ fontSize: 13, color: colors.textSecondary, fontWeight: 500 }}>
          {label}
        </span>
      </div>

      {/* 数值 */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
        <span
          style={{
            fontSize: 24,
            fontWeight: 600,
            fontFamily: '"JetBrains Mono", monospace',
            color: isSelected ? colors.primary : colors.text,
            letterSpacing: '-0.5px',
          }}
        >
          {value}
        </span>
        <span style={{ fontSize: 13, color: colors.textSecondary }}>
          {unit}
        </span>
      </div>

      {/* 副指标 */}
      {subValue && (
        <div style={{ marginTop: 6 }}>
          <span
            style={{
              fontSize: 12,
              color: statusColors[status],
              fontWeight: 500,
            }}
          >
            {subValue}
          </span>
        </div>
      )}
    </div>
  )

  return conditionalTooltip(card, tooltip) as React.ReactElement
}

export const HeroKPIPanel: React.FC<HeroKPIPanelProps> = ({
  result,
  selectedMetric,
  onMetricClick,
}) => {
  const { latency, throughput, memory, score } = result

  // 格式化数值（KPI卡片紧凑显示，大数值缩写，精度使用统一配置）
  const formatKPIValue = (value: number, metric: string): string => {
    const decimals = getMetricDecimals(metric)
    if (value >= 10000) return (value / 1000).toFixed(1) + 'k'
    if (value >= 1000) return value.toFixed(Math.min(decimals, 1))
    return formatNumber(value, decimals)
  }

  const kpiItems = useMemo(() => {
    // 计算状态
    const ttftStatus = (() => {
      const ttft = latency.prefill_total_latency_ms
      if (ttft < 200) return 'good' as const
      if (ttft < 500) return 'warning' as const
      return 'bad' as const
    })()
    const tpotStatus = (() => {
      const tpot = latency.decode_per_token_latency_ms
      if (tpot < 20) return 'good' as const
      if (tpot < 50) return 'warning' as const
      return 'bad' as const
    })()
    const mfuStatus = (() => {
      const mfu = throughput.model_flops_utilization * 100
      if (mfu >= 40) return 'good' as const
      if (mfu >= 20) return 'warning' as const
      return 'bad' as const
    })()

    return [
    {
      id: 'ttft',
      icon: <Clock className="h-4 w-4" />,
      label: 'TTFT',
      value: formatKPIValue(latency.prefill_total_latency_ms, 'ttft'),
      unit: 'ms',
      subValue: `延迟评分 ${score.latency_score.toFixed(0)}`,
      status: ttftStatus,
      tooltip: 'Time To First Token - 首个Token延迟',
    },
    {
      id: 'tpot',
      icon: <Rocket className="h-4 w-4" />,
      label: 'TPOT',
      value: formatKPIValue(latency.decode_per_token_latency_ms, 'tpot'),
      unit: 'ms',
      subValue: `E2E ${formatKPIValue(latency.end_to_end_latency_ms, 'end_to_end_latency')}ms`,
      status: tpotStatus,
      tooltip: 'Time Per Output Token - 每Token延迟',
    },
    {
      id: 'throughput',
      icon: <Zap className="h-4 w-4" />,
      label: 'Throughput',
      value: formatKPIValue(throughput.tokens_per_second, 'tps'),
      unit: 'tok/s',
      subValue: `${throughput.requests_per_second.toFixed(1)} req/s`,
      status: 'good' as const,
      tooltip: '吞吐量 - 每秒生成Token数',
    },
    {
      id: 'mfu',
      icon: <Gauge className="h-4 w-4" />,
      label: 'MFU',
      value: formatNumber(throughput.model_flops_utilization * 100, getMetricDecimals('mfu')),
      unit: '%',
      subValue: `显存 ${formatNumber(memory.total_per_chip_gb, getMetricDecimals('dram_occupy'))}GB/芯片`,
      status: mfuStatus,
      tooltip: 'Model FLOPs Utilization - 模型算力利用率',
    },
  ]
  }, [latency, throughput, memory, score])

  return (
    <div
      style={{
        display: 'flex',
        gap: 12,
        marginBottom: 16,
        flexWrap: 'wrap',
      }}
    >
      {kpiItems.map((item) => (
        <KPICard
          key={item.id}
          {...item}
          isSelected={selectedMetric === item.id}
          onClick={() => onMetricClick(item.id)}
        />
      ))}
    </div>
  )
}
