/**
 * 评分规则说明卡
 */

import React from 'react'
import { Clock, Zap, Gauge, Target, Calculator, Info } from 'lucide-react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { ScoreWeights, DEFAULT_SCORE_WEIGHTS } from '../../../../utils/llmDeployment/types'

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
  const [isOpen, setIsOpen] = React.useState(false)

  const rules = [
    {
      key: 'latency',
      name: '延迟评分',
      weight: weights.latency,
      icon: <Clock className="h-4 w-4 text-blue-500" />,
      color: 'blue' as const,
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
      icon: <Zap className="h-4 w-4 text-green-500" />,
      color: 'green' as const,
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
      icon: <Gauge className="h-4 w-4 text-yellow-500" />,
      color: 'yellow' as const,
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
      icon: <Target className="h-4 w-4 text-purple-500" />,
      color: 'purple' as const,
      description: '负载均衡程度评估',
      formula: 'score = load_balance_score × 100',
      details: [
        'TP/PP/EP 均匀切分时得分高',
        '不均匀切分会降低分数',
      ],
    },
  ]

  const badgeColorMap = {
    blue: 'bg-blue-100 text-blue-800 border-blue-200',
    green: 'bg-green-100 text-green-800 border-green-200',
    yellow: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    purple: 'bg-purple-100 text-purple-800 border-purple-200',
  } as const

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="bg-gray-50 rounded-lg mb-3">
      <CollapsibleTrigger className="flex items-center justify-between w-full p-3 hover:bg-gray-100 rounded-lg transition-colors">
        <div className="flex items-center gap-2">
          <Calculator className="h-4 w-4" />
          <span className="font-semibold text-[15px]">评分规则说明</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3.5 w-3.5 text-gray-400" />
              </TooltipTrigger>
              <TooltipContent>点击展开查看详细评分规则</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-3 pb-3">
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
                <span className="font-semibold text-sm">{rule.name}</span>
                <Badge variant="outline" className={`text-xs px-1.5 py-0.5 ${badgeColorMap[rule.color]}`}>
                  权重 {(rule.weight * 100).toFixed(0)}%
                </Badge>
              </div>
              <span className="text-[13px] text-gray-500">{rule.description}</span>
              <div style={formulaStyle}>{rule.formula}</div>
              <div style={{ marginTop: 6, paddingLeft: 24 }}>
                {rule.details.map((detail, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#666' }}>• {detail}</div>
                ))}
              </div>
            </div>
          ))}

          {/* 综合评分公式 */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Calculator className="h-4 w-4 text-blue-500" />
              <span className="font-semibold text-sm">综合评分公式</span>
            </div>
            <div style={{ ...formulaStyle, background: '#fff' }}>
              综合评分 = {(weights.latency * 100).toFixed(0)}% × 延迟 + {(weights.throughput * 100).toFixed(0)}% × 吞吐 + {(weights.efficiency * 100).toFixed(0)}% × 效率 + {(weights.balance * 100).toFixed(0)}% × 均衡
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
