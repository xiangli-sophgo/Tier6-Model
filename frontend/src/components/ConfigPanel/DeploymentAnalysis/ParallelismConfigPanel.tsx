/**
 * 并行策略配置面板
 */

import React from 'react'
import {
  Typography,
  InputNumber,
  Radio,
  Tooltip,
  Tag,
} from 'antd'
import {
  ParallelismStrategy,
  LLMModelConfig,
  HardwareConfig,
} from '../../../utils/llmDeployment/types'
import { configRowStyle } from './ConfigSelectors'

const { Text } = Typography

interface ParallelismConfigPanelProps {
  mode: 'manual' | 'auto'
  onModeChange: (mode: 'manual' | 'auto') => void
  manualStrategy: ParallelismStrategy
  onManualStrategyChange: (strategy: ParallelismStrategy) => void
  maxChips: number
  modelConfig: LLMModelConfig
  hardwareConfig: HardwareConfig
}

export const ParallelismConfigPanel: React.FC<ParallelismConfigPanelProps> = ({
  mode,
  onModeChange,
  manualStrategy,
  onManualStrategyChange,
  maxChips,
  modelConfig,
  hardwareConfig,
}) => {
  // 总芯片数 = dp * tp（不包括 PP）
  const totalParallelism = manualStrategy.dp * manualStrategy.tp
  const [showConstraints, setShowConstraints] = React.useState(true)

  // 计算搜索约束范围
  const isMoE = modelConfig.model_type === 'moe' && modelConfig.moe_config
  const maxTP = Math.min(128, modelConfig.num_attention_heads, hardwareConfig.node.chips_per_node)
  const maxEP = isMoE && modelConfig.moe_config ? modelConfig.moe_config.num_experts : 1

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <Radio.Group
          size="small"
          value={mode}
          onChange={(e) => onModeChange(e.target.value)}
          buttonStyle="solid"
        >
          <Radio.Button value="auto">自动搜索</Radio.Button>
          <Radio.Button value="manual">手动指定</Radio.Button>
        </Radio.Group>
      </div>

      {mode === 'manual' ? (
        <div>
          {/* DS_TPU 模式：只有 DP, TP, EP (+ moe_tp for MoE) */}
          <div style={{ display: 'grid', gridTemplateColumns: isMoE ? 'repeat(4, 1fr)' : 'repeat(3, 1fr)', gap: 4 }}>
            {/* DP */}
            <div style={{ textAlign: 'center' }}>
              <Text style={{ fontSize: 11, display: 'block' }}>DP</Text>
              <InputNumber
                size="small"
                min={1}
                max={1024}
                value={manualStrategy.dp}
                onChange={(v) => onManualStrategyChange({ ...manualStrategy, dp: v || 1 })}
                style={{ width: '100%' }}
              />
            </div>
            {/* TP */}
            <div style={{ textAlign: 'center' }}>
              <Tooltip title="张量并行 (Attention)">
                <Text style={{ fontSize: 11, display: 'block' }}>TP</Text>
              </Tooltip>
              <InputNumber
                size="small"
                min={1}
                max={128}
                value={manualStrategy.tp}
                onChange={(v) => onManualStrategyChange({ ...manualStrategy, tp: v || 1 })}
                style={{ width: '100%' }}
              />
            </div>
            {/* EP - 仅 MoE */}
            {isMoE && (
              <div style={{ textAlign: 'center' }}>
                <Tooltip title="专家并行">
                  <Text style={{ fontSize: 11, display: 'block' }}>EP</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={1}
                  max={256}
                  value={manualStrategy.ep}
                  onChange={(v) => onManualStrategyChange({ ...manualStrategy, ep: v || 1 })}
                  style={{ width: '100%' }}
                />
              </div>
            )}
            {/* moe_tp - 仅 MoE */}
            {isMoE && (
              <div style={{ textAlign: 'center' }}>
                <Tooltip title="MoE 专家内张量并行">
                  <Text style={{ fontSize: 11, display: 'block' }}>MoE_TP</Text>
                </Tooltip>
                <InputNumber
                  size="small"
                  min={1}
                  max={128}
                  value={manualStrategy.moe_tp || 1}
                  onChange={(v) => onManualStrategyChange({ ...manualStrategy, moe_tp: v || 1 })}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
          <div style={{ marginTop: 8, textAlign: 'center' }}>
            <Text type={totalParallelism > maxChips ? 'danger' : 'secondary'} style={{ fontSize: 12 }}>
              总芯片数: {totalParallelism} / {maxChips}
            </Text>
            {isMoE && (
              <div style={{ marginTop: 4 }}>
                <Text
                  type={manualStrategy.dp * manualStrategy.tp === (manualStrategy.moe_tp || 1) * manualStrategy.ep ? 'success' : 'danger'}
                  style={{ fontSize: 11 }}
                >
                  {manualStrategy.dp * manualStrategy.tp === (manualStrategy.moe_tp || 1) * manualStrategy.ep
                    ? '✓ MoE 约束满足'
                    : `✗ 约束不满足: ${manualStrategy.dp}×${manualStrategy.tp}=${manualStrategy.dp * manualStrategy.tp} ≠ ${manualStrategy.moe_tp || 1}×${manualStrategy.ep}=${(manualStrategy.moe_tp || 1) * manualStrategy.ep}`}
                </Text>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>使用芯片数</Text>
            <Tag color="green" style={{ fontSize: 11 }}>
              {maxChips} 个
            </Tag>
          </div>
          <div style={configRowStyle}>
            <Text style={{ fontSize: 12 }}>优化目标</Text>
            <Tag color="blue" style={{ fontSize: 11 }}>
              TPS per Chip
            </Tag>
          </div>

          {/* 搜索约束显示 */}
          <div style={{ marginTop: 12 }}>
            <div
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              onClick={() => setShowConstraints(!showConstraints)}
            >
              <Text style={{ fontSize: 12 }}>搜索约束</Text>
              <Text type="secondary" style={{ fontSize: 11 }}>{showConstraints ? '▲' : '▼'}</Text>
            </div>

            {showConstraints && (
              <div style={{ marginTop: 8, padding: 8, background: '#fafafa', borderRadius: 6 }}>
                <div style={{ display: 'grid', gap: 6, fontSize: 11 }}>
                  {/* 并行模式说明 */}
                  <div style={{ marginBottom: 4, paddingBottom: 4, borderBottom: '1px dashed #d9d9d9' }}>
                    {/* <Tag color="green" style={{ fontSize: 10, margin: 0 }}>
                      DS_TPU 模式
                    </Tag> */}
                    <Text type="secondary" style={{ fontSize: 10, marginLeft: 8 }}>
                      DP × TP {isMoE ? '× EP' : ''}（无 PP/SP）
                    </Text>
                  </div>

                  {/* TP 约束 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text style={{ fontSize: 11 }}>TP 约束</Text>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      ≤ min(128, 头数{modelConfig.num_attention_heads}, Board内芯片数{hardwareConfig.node.chips_per_node}) = {maxTP}
                      <span style={{ margin: '0 12px' }}>&</span>
                      头数 {modelConfig.num_attention_heads} % TP == 0
                    </Text>
                  </div>

                  {/* EP 约束 - 仅 MoE */}
                  {isMoE && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text style={{ fontSize: 11 }}>EP 约束</Text>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        ≤ 专家数 {maxEP}
                        <span style={{ margin: '0 12px' }}>&</span>
                        专家数 {maxEP} % EP == 0
                      </Text>
                    </div>
                  )}

                  {/* moe_tp 约束 - 仅 MoE */}
                  {isMoE && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text style={{ fontSize: 11 }}>MoE_TP 约束</Text>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        ≤ min(128, 头数{modelConfig.num_attention_heads}, Board内芯片数{hardwareConfig.node.chips_per_node}) = {maxTP}
                        <span style={{ margin: '0 12px' }}>&</span>
                        头数 {modelConfig.num_attention_heads} % MoE_TP == 0
                      </Text>
                    </div>
                  )}

                  {/* MoE 约束 */}
                  {isMoE && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text style={{ fontSize: 11 }}>MoE 约束</Text>
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        DP × TP = MoE_TP × EP
                      </Text>
                    </div>
                  )}

                  {/* Board 约束 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text style={{ fontSize: 11 }}>Board 内芯片数量</Text>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {hardwareConfig.node.chips_per_node} 个/Board
                    </Text>
                  </div>

                  {/* 当前搜索策略说明 */}
                  <div style={{ marginTop: 4, paddingTop: 6, borderTop: '1px dashed #d9d9d9' }}>
                    <Text type="secondary" style={{ fontSize: 10, lineHeight: '1.4' }}>
                      {isMoE ? (
                        <>
                          • 枚举所有 TP/EP/moe_tp 因子组合<br />
                          • 通过约束自动计算 DP<br />
                        </>
                      ) : (
                        <>
                          • 枚举所有 TP 因子<br />
                          • 根据芯片数计算 DP<br />
                        </>
                      )}
                      • TP 优先放置在节点内（高带宽）
                    </Text>
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
