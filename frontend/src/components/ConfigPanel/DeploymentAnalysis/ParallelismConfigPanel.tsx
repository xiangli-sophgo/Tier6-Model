/**
 * 并行策略配置面板
 */

import React from 'react'
import {
  Typography,
  Button,
  InputNumber,
  Select,
  Radio,
  Tooltip,
} from 'antd'
import {
  ParallelismStrategy,
  SearchConstraints,
  ScoreWeights,
  DEFAULT_SCORE_WEIGHTS,
} from '../../../utils/llmDeployment/types'
import { configRowStyle } from './ConfigSelectors'

const { Text } = Typography

// 优化目标对应的预设权重
const OPTIMIZATION_TARGET_WEIGHTS: Record<string, ScoreWeights> = {
  latency: { latency: 0.7, throughput: 0.15, efficiency: 0.1, balance: 0.05 },
  throughput: { latency: 0.1, throughput: 0.7, efficiency: 0.15, balance: 0.05 },
  efficiency: { latency: 0.1, throughput: 0.2, efficiency: 0.6, balance: 0.1 },
  balanced: { ...DEFAULT_SCORE_WEIGHTS },
}

// 自定义权重存储
const CUSTOM_WEIGHTS_KEY = 'llm_custom_score_weights'

function loadCustomWeights(): ScoreWeights | null {
  try {
    const data = localStorage.getItem(CUSTOM_WEIGHTS_KEY)
    return data ? JSON.parse(data) : null
  } catch {
    return null
  }
}

function saveCustomWeights(weights: ScoreWeights) {
  localStorage.setItem(CUSTOM_WEIGHTS_KEY, JSON.stringify(weights))
}

interface ParallelismConfigPanelProps {
  mode: 'manual' | 'auto'
  onModeChange: (mode: 'manual' | 'auto') => void
  manualStrategy: ParallelismStrategy
  onManualStrategyChange: (strategy: ParallelismStrategy) => void
  searchConstraints: SearchConstraints
  onSearchConstraintsChange: (constraints: SearchConstraints) => void
  maxChips: number
  scoreWeights: ScoreWeights
  onScoreWeightsChange: (weights: ScoreWeights) => void
}

export const ParallelismConfigPanel: React.FC<ParallelismConfigPanelProps> = ({
  mode,
  onModeChange,
  manualStrategy,
  onManualStrategyChange,
  searchConstraints,
  onSearchConstraintsChange,
  maxChips,
  scoreWeights,
  onScoreWeightsChange,
}) => {
  const totalParallelism = manualStrategy.dp * manualStrategy.tp * manualStrategy.pp * manualStrategy.ep
  const [showWeights, setShowWeights] = React.useState(false)
  const [hasCustomWeights, setHasCustomWeights] = React.useState(() => loadCustomWeights() !== null)

  // 计算权重总和
  const weightSum = scoreWeights.latency + scoreWeights.throughput + scoreWeights.efficiency + scoreWeights.balance

  // 当前优化目标
  const currentTarget = (searchConstraints as any).optimization_target || 'balanced'

  // 优化目标变化时，更新权重
  const handleTargetChange = (target: string) => {
    onSearchConstraintsChange({ ...searchConstraints, optimization_target: target as any })
    // 如果不是自定义，使用预设权重
    if (target !== 'custom') {
      onScoreWeightsChange(OPTIMIZATION_TARGET_WEIGHTS[target] || DEFAULT_SCORE_WEIGHTS)
    } else {
      // 使用保存的自定义权重
      const saved = loadCustomWeights()
      if (saved) {
        onScoreWeightsChange(saved)
      }
    }
  }

  // 保存当前权重为自定义
  const handleSaveCustomWeights = () => {
    saveCustomWeights(scoreWeights)
    setHasCustomWeights(true)
  }

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <Radio.Group
          size="small"
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
          buttonStyle="solid"
        >
          <Radio.Button value="manual">手动指定</Radio.Button>
          <Radio.Button value="auto">自动搜索</Radio.Button>
        </Radio.Group>
      </div>

      {mode === 'manual' ? (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 4 }}>
            {(['dp', 'tp', 'pp', 'ep', 'sp'] as const).map((key) => (
              <div key={key} style={{ textAlign: 'center' }}>
                <Text style={{ fontSize: 11, display: 'block' }}>{key.toUpperCase()}</Text>
                <InputNumber
                  size="small"
                  min={1}
                  max={64}
                  value={manualStrategy[key]}
                  onChange={(v) => onManualStrategyChange({ ...manualStrategy, [key]: v || 1 })}
                  style={{ width: '100%' }}
                />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, textAlign: 'center' }}>
            <Text type={totalParallelism > maxChips ? 'danger' : 'secondary'} style={{ fontSize: 12 }}>
              总并行度: {totalParallelism} / {maxChips} 芯片
            </Text>
          </div>
        </div>
      ) : (
        <div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>最大芯片数</Text>
            <InputNumber
              size="small"
              min={1}
              max={1024}
              value={searchConstraints.max_chips || maxChips}
              onChange={(v) => onSearchConstraintsChange({ ...searchConstraints, max_chips: v || maxChips })}
              style={{ width: 80 }}
            />
          </div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>优化目标</Text>
            <Select
              size="small"
              value={currentTarget}
              onChange={handleTargetChange}
              style={{ width: 110 }}
              options={[
                { value: 'latency', label: '低延迟' },
                { value: 'throughput', label: '高吞吐' },
                { value: 'efficiency', label: '高效率' },
                { value: 'balanced', label: '均衡' },
                ...(hasCustomWeights ? [{ value: 'custom', label: '自定义' }] : []),
              ]}
            />
          </div>

          {/* 评分权重配置 - 仅在自动搜索模式显示 */}
          <div style={{ marginTop: 12 }}>
            <div
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              onClick={() => setShowWeights(!showWeights)}
            >
              <Text style={{ fontSize: 12 }}>评分权重</Text>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Text type="secondary" style={{ fontSize: 10 }}>
                  延迟:{scoreWeights.latency} 吞吐:{scoreWeights.throughput}
                </Text>
                <Text type="secondary" style={{ fontSize: 11 }}>{showWeights ? '▲' : '▼'}</Text>
              </div>
            </div>

            {showWeights && (
              <div style={{ marginTop: 8, padding: 8, background: '#fafafa', borderRadius: 6 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  <div>
                    <Tooltip title="首Token延迟(TTFT)的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>延迟</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.latency}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, latency: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="Token吞吐量和MFU的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>吞吐</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.throughput}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, throughput: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="计算和显存利用率的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>效率</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.efficiency}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, efficiency: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div>
                    <Tooltip title="负载均衡程度的重要性权重">
                      <Text style={{ fontSize: 11, cursor: 'help' }}>均衡</Text>
                    </Tooltip>
                    <InputNumber
                      size="small"
                      min={0}
                      max={1}
                      step={0.05}
                      value={scoreWeights.balance}
                      onChange={(v) => onScoreWeightsChange({ ...scoreWeights, balance: v || 0 })}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
                <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text type={Math.abs(weightSum - 1) > 0.01 ? 'danger' : 'secondary'} style={{ fontSize: 10 }}>
                    权重总和: {weightSum.toFixed(2)} {Math.abs(weightSum - 1) > 0.01 && '(建议=1.0)'}
                  </Text>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button
                      size="small"
                      type="link"
                      style={{ fontSize: 10, padding: 0 }}
                      onClick={handleSaveCustomWeights}
                    >
                      保存为自定义
                    </Button>
                    <Button
                      size="small"
                      type="link"
                      style={{ fontSize: 10, padding: 0 }}
                      onClick={() => onScoreWeightsChange(OPTIMIZATION_TARGET_WEIGHTS[currentTarget] || DEFAULT_SCORE_WEIGHTS)}
                    >
                      重置
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default ParallelismConfigPanel
