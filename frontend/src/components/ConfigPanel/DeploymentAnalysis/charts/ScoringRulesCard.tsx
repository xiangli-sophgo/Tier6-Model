/**
 * 评分规则说明卡
 */

import React, { useState } from 'react'
import { Collapse, Typography, Tag, Tooltip } from 'antd'
import {
  ClockCircleOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  AimOutlined,
  CalculatorOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import { ScoreWeights, DEFAULT_SCORE_WEIGHTS } from '../../../../utils/llmDeployment/types'

const { Text } = Typography

interface ScoringRulesCardProps {
  weights?: ScoreWeights
}

const ruleItemStyle: React.CSSProperties = {
  padding: '8px 0',
  borderBottom: '1px solid #f0f0f0',
}

const ruleHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginBottom: 4,
}

const formulaStyle: React.CSSProperties = {
  fontSize: 13,
  color: '#666',
  fontFamily: 'monospace',
  background: '#f5f5f5',
  padding: '6px 10px',
  borderRadius: 4,
  marginTop: 6,
}

export const ScoringRulesCard: React.FC<ScoringRulesCardProps> = ({
  weights = DEFAULT_SCORE_WEIGHTS,
}) => {
  const [activeKey, setActiveKey] = useState<string[]>([])

  const rules = [
    {
      key: 'latency',
      name: '延迟评分',
      weight: weights.latency,
      icon: <ClockCircleOutlined style={{ color: '#1890ff' }} />,
      color: 'blue',
      description: 'TTFT (Time To First Token) 越低越好',
      formula: 'score = max(0, min(100, 100 - (TTFT - 100) / 9))',
      details: [
        'TTFT < 100ms → 100分 (满分)',
        'TTFT = 550ms → 50分',
        'TTFT > 1000ms → 0分',
      ],
    },
    {
      key: 'throughput',
      name: '吞吐评分',
      weight: weights.throughput,
      icon: <ThunderboltOutlined style={{ color: '#52c41a' }} />,
      color: 'green',
      description: 'MFU (Model FLOPs Utilization) 越高越好',
      formula: 'score = min(100, MFU × 200)',
      details: [
        'MFU ≥ 50% → 100分 (满分)',
        'MFU = 25% → 50分',
        'MFU = 0% → 0分',
      ],
    },
    {
      key: 'efficiency',
      name: '效率评分',
      weight: weights.efficiency,
      icon: <DashboardOutlined style={{ color: '#faad14' }} />,
      color: 'orange',
      description: '计算和显存利用率综合评估',
      formula: 'score = (compute_util + memory_util) / 2 × 100',
      details: [
        '综合利用率越高分数越高',
        '避免资源浪费',
      ],
    },
    {
      key: 'balance',
      name: '均衡评分',
      weight: weights.balance,
      icon: <AimOutlined style={{ color: '#722ed1' }} />,
      color: 'purple',
      description: '负载均衡程度评估',
      formula: 'score = load_balance_score × 100',
      details: [
        'TP/PP/EP 均匀切分时得分高',
        '不均匀切分会降低分数',
      ],
    },
  ]

  const items = [
    {
      key: 'rules',
      label: (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <CalculatorOutlined style={{ fontSize: 16 }} />
          <Text strong style={{ fontSize: 15 }}>评分规则说明</Text>
          <Tooltip title="点击展开查看详细评分规则">
            <InfoCircleOutlined style={{ color: '#999', fontSize: 14 }} />
          </Tooltip>
        </div>
      ),
      children: (
        <div>
          {rules.map((rule, index) => (
            <div
              key={rule.key}
              style={{
                ...ruleItemStyle,
                borderBottom: index === rules.length - 1 ? 'none' : ruleItemStyle.borderBottom,
              }}
            >
              <div style={ruleHeaderStyle}>
                {rule.icon}
                <Text strong style={{ fontSize: 14 }}>{rule.name}</Text>
                <Tag color={rule.color} style={{ fontSize: 12, padding: '2px 6px', lineHeight: '18px' }}>
                  权重 {(rule.weight * 100).toFixed(0)}%
                </Tag>
              </div>
              <Text type="secondary" style={{ fontSize: 13 }}>{rule.description}</Text>
              <div style={formulaStyle}>{rule.formula}</div>
              <div style={{ marginTop: 6, paddingLeft: 24 }}>
                {rule.details.map((detail, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#666' }}>• {detail}</div>
                ))}
              </div>
            </div>
          ))}

          {/* 综合评分公式 */}
          <div style={{ marginTop: 16, padding: 12, background: '#e6f7ff', borderRadius: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <CalculatorOutlined style={{ color: '#1890ff', fontSize: 16 }} />
              <Text strong style={{ fontSize: 14 }}>综合评分公式</Text>
            </div>
            <div style={{ ...formulaStyle, background: '#fff' }}>
              综合评分 = {(weights.latency * 100).toFixed(0)}% × 延迟 + {(weights.throughput * 100).toFixed(0)}% × 吞吐 + {(weights.efficiency * 100).toFixed(0)}% × 效率 + {(weights.balance * 100).toFixed(0)}% × 均衡
            </div>
          </div>
        </div>
      ),
    },
  ]

  return (
    <Collapse
      size="small"
      activeKey={activeKey}
      onChange={(keys) => setActiveKey(keys as string[])}
      style={{
        background: '#fafafa',
        borderRadius: 8,
        marginBottom: 12,
      }}
      items={items}
    />
  )
}
