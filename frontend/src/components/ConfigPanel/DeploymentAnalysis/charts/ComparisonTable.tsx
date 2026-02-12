/**
 * 公式计算 vs 仿真结果 对比表格
 */

import React from 'react'
import { InfoTooltip } from '@/components/ui/info-tooltip'
import { FormulaVsSimComparison } from '../../../../utils/llmDeployment/types'
import { tableStyle, thStyle, tdStyle } from '../../../ui/common-styles'
import { COLORS } from '../../../../utils/design-tokens'
import { formatNumber, getMetricDecimals } from '../../../../utils/formatters'

// 格式化偏差显示
const formatDeviation = (pct: number): string => {
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(1)}%`
}

// 判断偏差是否显著
const isSignificantDeviation = (pct: number, threshold = 10): boolean => {
  return Math.abs(pct) >= threshold
}

interface ComparisonTableProps {
  comparison: FormulaVsSimComparison
}

/** 表格行配置 */
const TABLE_ROWS = [
  {
    key: 'ttft',
    label: 'TTFT',
    tooltip: '首 Token 延迟 (Time To First Token)',
    formatValue: (v: number) => `${formatNumber(v, getMetricDecimals('ttft'))} ms`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.ttft_ms,
    getSim: (c: FormulaVsSimComparison) => c.simulation.ttft_ms,
    getDev: (c: FormulaVsSimComparison) => c.deviation.ttft_pct,
  },
  {
    key: 'tpot',
    label: 'TPOT',
    tooltip: '平均每 Token 延迟 (Time Per Output Token)',
    formatValue: (v: number) => `${formatNumber(v, getMetricDecimals('tpot'))} ms`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.tpot_ms,
    getSim: (c: FormulaVsSimComparison) => c.simulation.tpot_ms,
    getDev: (c: FormulaVsSimComparison) => c.deviation.tpot_pct,
  },
  {
    key: 'mfu',
    label: 'MFU',
    tooltip: '模型算力利用率 (Model FLOPs Utilization)',
    formatValue: (v: number) => `${formatNumber(v * 100, getMetricDecimals('mfu'))}%`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.mfu,
    getSim: (c: FormulaVsSimComparison) => c.simulation.mfu,
    getDev: (c: FormulaVsSimComparison) => c.deviation.mfu_pct,
  },
  {
    key: 'mbu',
    label: 'MBU',
    tooltip: '内存带宽利用率 (Memory Bandwidth Utilization)',
    formatValue: (v: number) => `${formatNumber(v * 100, getMetricDecimals('mbu'))}%`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.mbu,
    getSim: (c: FormulaVsSimComparison) => c.simulation.mbu,
    getDev: (c: FormulaVsSimComparison) => c.deviation.mbu_pct,
  },
  {
    key: 'score',
    label: '综合评分',
    tooltip: '加权综合评分 (0-100)',
    formatValue: (v: number) => formatNumber(v, getMetricDecimals('score')),
    getFormula: (c: FormulaVsSimComparison) => c.formula.score,
    getSim: (c: FormulaVsSimComparison) => c.simulation.score,
    getDev: (c: FormulaVsSimComparison) => c.deviation.score_pct,
    isHighlight: true,
  },
]

const getDeviationStyle = (pct: number): React.CSSProperties => {
  const isSignificant = isSignificantDeviation(pct)
  return {
    ...tdStyle,
    textAlign: 'right',
    color: isSignificant ? (pct > 0 ? COLORS.semantic.error.main : COLORS.semantic.success.main) : COLORS.text.secondary,
    fontWeight: isSignificant ? 600 : 400,
  }
}

export const ComparisonTable: React.FC<ComparisonTableProps> = ({ comparison }) => {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>指标</th>
            <th style={{ ...thStyle, textAlign: 'right', color: COLORS.palette.blue.main }}>公式估算</th>
            <th style={{ ...thStyle, textAlign: 'right', color: COLORS.palette.green.main }}>仿真结果</th>
            <th style={{ ...thStyle, textAlign: 'right' }}>偏差</th>
          </tr>
        </thead>
        <tbody>
          {TABLE_ROWS.map((row) => {
            const formulaValue = row.getFormula(comparison)
            const simValue = row.getSim(comparison)
            const deviation = row.getDev(comparison)

            return (
              <tr
                key={row.key}
                style={row.isHighlight ? { background: COLORS.palette.gold.light } : undefined}
              >
                <td style={tdStyle}>
                  <InfoTooltip content={row.tooltip}>
                    <span className={row.isHighlight ? 'font-semibold' : ''}>{row.label}</span>
                  </InfoTooltip>
                </td>
                <td style={{ ...tdStyle, textAlign: 'right', color: COLORS.palette.blue.main }}>
                  {row.formatValue(formulaValue)}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right', color: COLORS.palette.green.main }}>
                  {row.formatValue(simValue)}
                </td>
                <td style={getDeviationStyle(deviation)}>
                  {formatDeviation(deviation)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div style={{ marginTop: 12, fontSize: 11, color: COLORS.text.secondary, lineHeight: 1.6 }}>
        <div style={{ marginBottom: 4, fontWeight: 500, color: COLORS.text.primary }}>偏差说明:</div>
        <div>• 偏差 = (仿真 - 公式) / 公式 × 100%，<span style={{ color: COLORS.semantic.error.main }}>红色</span>表示仿真值偏高，<span style={{ color: COLORS.semantic.success.main }}>绿色</span>表示仿真值偏低</div>
        <div>• <b>TTFT/TPOT 偏差原因</b>: 公式使用简化模型估算，仿真考虑了细粒度操作(13个子操作/层)和实际内存访问延迟</div>
        <div>• <b>MFU 偏差原因</b>: 公式基于 Roofline 理论估算(考虑 HBM 效率 85%)，仿真动态计算实际利用率</div>
        <div>• <b>MBU 偏差原因</b>: 仿真包含 KV Cache 读取开销，公式采用简化模型</div>
        <div style={{ marginTop: 4, color: COLORS.text.muted }}>注: 偏差 ±10% 以内属于合理范围，超过 20% 需检查配置参数</div>
      </div>
    </div>
  )
}

export default ComparisonTable
