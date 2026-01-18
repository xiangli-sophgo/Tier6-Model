/**
 * Hero KPI 面板 - 顶部关键指标卡片组
 */

import React from 'react'
import { Typography, Tooltip } from 'antd'
import {
  ClockCircleOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'

const { Text } = Typography

interface HeroKPIPanelProps {
  result: PlanAnalysisResult
  selectedMetric: string | null
  onMetricClick: (metric: string) => void
}

const colors = {
  primary: '#4F6BED',
  primaryLight: 'rgba(79, 107, 237, 0.08)',
  success: '#52c41a',
  warning: '#faad14',
  border: '#E5E5E5',
  text: '#1A1A1A',
  textSecondary: '#666666',
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
    bad: '#ff4d4f',
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
        <Text style={{ fontSize: 13, color: colors.textSecondary, fontWeight: 500 }}>
          {label}
        </Text>
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

  return tooltip ? <Tooltip title={tooltip}>{card}</Tooltip> : card
}

export const HeroKPIPanel: React.FC<HeroKPIPanelProps> = ({
  result,
  selectedMetric,
  onMetricClick,
}) => {
  const { latency, throughput, memory, score } = result

  // 计算 TTFT 状态
  const getTTFTStatus = (): 'good' | 'warning' | 'bad' => {
    const ttft = latency.prefill_total_latency_ms
    if (ttft < 200) return 'good'
    if (ttft < 500) return 'warning'
    return 'bad'
  }

  // 计算 TPOT 状态
  const getTPOTStatus = (): 'good' | 'warning' | 'bad' => {
    const tpot = latency.decode_per_token_latency_ms
    if (tpot < 20) return 'good'
    if (tpot < 50) return 'warning'
    return 'bad'
  }

  // 计算 MFU 状态
  const getMFUStatus = (): 'good' | 'warning' | 'bad' => {
    const mfu = throughput.model_flops_utilization * 100
    if (mfu >= 40) return 'good'
    if (mfu >= 20) return 'warning'
    return 'bad'
  }

  // 格式化数值
  const formatValue = (value: number, decimals: number = 1): string => {
    if (value >= 10000) return (value / 1000).toFixed(1) + 'k'
    if (value >= 1000) return value.toFixed(0)
    if (value >= 100) return value.toFixed(0)
    if (value >= 10) return value.toFixed(decimals)
    return value.toFixed(decimals + 1)
  }

  const kpiItems = [
    {
      id: 'ttft',
      icon: <ClockCircleOutlined />,
      label: 'FTL',
      value: formatValue(latency.prefill_total_latency_ms),
      unit: 'ms',
      subValue: `延迟评分 ${score.latency_score.toFixed(0)}`,
      status: getTTFTStatus(),
      tooltip: 'First Token Latency - 首个Token延迟',
    },
    {
      id: 'tpot',
      icon: <RocketOutlined />,
      label: 'TPOT',
      value: formatValue(latency.decode_per_token_latency_ms, 2),
      unit: 'ms',
      subValue: `E2E ${formatValue(latency.end_to_end_latency_ms)}ms`,
      status: getTPOTStatus(),
      tooltip: 'Time Per Output Token - 每Token延迟',
    },
    {
      id: 'throughput',
      icon: <ThunderboltOutlined />,
      label: 'Throughput',
      value: formatValue(throughput.tokens_per_second),
      unit: 'tok/s',
      subValue: `${throughput.requests_per_second.toFixed(1)} req/s`,
      status: 'good' as const,
      tooltip: '吞吐量 - 每秒生成Token数',
    },
    {
      id: 'mfu',
      icon: <DashboardOutlined />,
      label: 'MFU',
      value: (throughput.model_flops_utilization * 100).toFixed(1),
      unit: '%',
      subValue: `显存 ${memory.total_per_chip_gb.toFixed(1)}GB/芯片`,
      status: getMFUStatus(),
      tooltip: 'Model FLOPs Utilization - 模型算力利用率',
    },
  ]

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
