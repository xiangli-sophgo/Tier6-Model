/**
 * 公式计算 vs 仿真结果 对比表格
 */

import React from 'react'
import { Typography, Tooltip } from 'antd'
import { FormulaVsSimComparison } from '../../../../utils/llmDeployment/simulation/types'
import { formatDeviation, isSignificantDeviation } from '../../../../utils/llmDeployment/simulationScorer'

const { Text } = Typography

interface ComparisonTableProps {
  comparison: FormulaVsSimComparison
}

/** 表格行配置 */
const TABLE_ROWS = [
  {
    key: 'ttft',
    label: 'FTL',
    tooltip: '首 Token 延迟 (First Token Latency)',
    formatValue: (v: number) => `${v.toFixed(2)} ms`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.ttft_ms,
    getSim: (c: FormulaVsSimComparison) => c.simulation.ttft_ms,
    getDev: (c: FormulaVsSimComparison) => c.deviation.ttft_pct,
  },
  {
    key: 'tpot',
    label: 'TPOT',
    tooltip: '平均每 Token 延迟 (Time Per Output Token)',
    formatValue: (v: number) => `${v.toFixed(2)} ms`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.tpot_ms,
    getSim: (c: FormulaVsSimComparison) => c.simulation.tpot_ms,
    getDev: (c: FormulaVsSimComparison) => c.deviation.tpot_pct,
  },
  {
    key: 'mfu',
    label: 'MFU',
    tooltip: '模型算力利用率 (Model FLOPs Utilization)',
    formatValue: (v: number) => `${(v * 100).toFixed(1)}%`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.mfu,
    getSim: (c: FormulaVsSimComparison) => c.simulation.mfu,
    getDev: (c: FormulaVsSimComparison) => c.deviation.mfu_pct,
  },
  {
    key: 'mbu',
    label: 'MBU',
    tooltip: '内存带宽利用率 (Memory Bandwidth Utilization)',
    formatValue: (v: number) => `${(v * 100).toFixed(1)}%`,
    getFormula: (c: FormulaVsSimComparison) => c.formula.mbu,
    getSim: (c: FormulaVsSimComparison) => c.simulation.mbu,
    getDev: (c: FormulaVsSimComparison) => c.deviation.mbu_pct,
  },
  {
    key: 'score',
    label: '综合评分',
    tooltip: '加权综合评分 (0-100)',
    formatValue: (v: number) => v.toFixed(1),
    getFormula: (c: FormulaVsSimComparison) => c.formula.score,
    getSim: (c: FormulaVsSimComparison) => c.simulation.score,
    getDev: (c: FormulaVsSimComparison) => c.deviation.score_pct,
    isHighlight: true,
  },
]

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: 12,
}

const thStyle: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'left',
  fontWeight: 600,
  borderBottom: '2px solid #e8e8e8',
  background: '#fafafa',
}

const tdStyle: React.CSSProperties = {
  padding: '8px 12px',
  borderBottom: '1px solid #f0f0f0',
}

const getDeviationStyle = (pct: number): React.CSSProperties => {
  const isSignificant = isSignificantDeviation(pct)
  return {
    ...tdStyle,
    textAlign: 'right',
    color: isSignificant ? (pct > 0 ? '#cf1322' : '#389e0d') : '#666',
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
            <th style={{ ...thStyle, textAlign: 'right', color: '#1890ff' }}>公式估算</th>
            <th style={{ ...thStyle, textAlign: 'right', color: '#52c41a' }}>仿真结果</th>
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
                style={row.isHighlight ? { background: '#fffbe6' } : undefined}
              >
                <td style={tdStyle}>
                  <Tooltip title={row.tooltip}>
                    <Text strong={row.isHighlight}>{row.label}</Text>
                  </Tooltip>
                </td>
                <td style={{ ...tdStyle, textAlign: 'right', color: '#1890ff' }}>
                  {row.formatValue(formulaValue)}
                </td>
                <td style={{ ...tdStyle, textAlign: 'right', color: '#52c41a' }}>
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
      <div style={{ marginTop: 12, fontSize: 11, color: '#666', lineHeight: 1.6 }}>
        <div style={{ marginBottom: 4, fontWeight: 500, color: '#333' }}>偏差说明:</div>
        <div>• 偏差 = (仿真 - 公式) / 公式 × 100%，<span style={{ color: '#cf1322' }}>红色</span>表示仿真值偏高，<span style={{ color: '#389e0d' }}>绿色</span>表示仿真值偏低</div>
        <div>• <b>TTFT/TPOT 偏差原因</b>: 公式使用简化模型估算，仿真考虑了细粒度操作(13个子操作/层)和实际内存访问延迟</div>
        <div>• <b>MFU 偏差原因</b>: 公式基于 Roofline 理论估算(考虑 HBM 效率 85%)，仿真动态计算实际利用率</div>
        <div>• <b>MBU 偏差原因</b>: 仿真包含 KV Cache 读取开销，公式采用简化模型</div>
        <div style={{ marginTop: 4, color: '#999' }}>注: 偏差 ±10% 以内属于合理范围，超过 20% 需检查配置参数</div>
      </div>
    </div>
  )
}

export default ComparisonTable
