/**
 * 并行策略配置面板
 */

import React from 'react'
import { Badge } from '@/components/ui/badge'
import { NumberInput } from '@/components/ui/number-input'
import { HelpTooltip } from '@/components/ui/info-tooltip'
import {
  ParallelismStrategy,
  LLMModelConfig,
  HardwareConfig,
} from '../../../utils/llmDeployment/types'

interface ParallelismConfigPanelProps {
  mode: 'manual' | 'auto' | 'sweep'
  onModeChange: (mode: 'manual' | 'auto' | 'sweep') => void
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
  const [showConstraints, setShowConstraints] = React.useState(false)

  // 计算搜索约束范围
  const isMoE = modelConfig.model_type === 'moe' && modelConfig.moe_config
  const maxTP = Math.min(128, modelConfig.num_attention_heads, maxChips)
  const maxEP = isMoE && modelConfig.moe_config ? modelConfig.moe_config.num_experts : 1

  return (
    <div>
      {/* 模式切换按钮组 */}
        <div className="mb-3">
          <div className="flex rounded-md border border-gray-200 overflow-hidden w-fit">
            <button
              onClick={() => onModeChange('auto')}
              className={`px-3 py-1 text-xs transition-colors ${
                mode === 'auto' ? 'bg-blue-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              自动搜索
            </button>
            <button
              onClick={() => onModeChange('manual')}
              className={`px-3 py-1 text-xs transition-colors border-l border-gray-200 ${
                mode === 'manual' ? 'bg-blue-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              手动指定
            </button>
            <button
              onClick={() => onModeChange('sweep')}
              className={`px-3 py-1 text-xs transition-colors border-l border-gray-200 ${
                mode === 'sweep' ? 'bg-blue-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              参数遍历
            </button>
          </div>
        </div>

        {mode === 'manual' ? (
          <div>
            {/* DS_TPU 模式：只有 DP, TP, EP (+ moe_tp for MoE) */}
            <div className={`grid gap-1 ${isMoE ? 'grid-cols-4' : 'grid-cols-3'}`}>
              {/* DP */}
              <div className="text-center">
                <span className="text-[11px] block mb-1">DP</span>
                <NumberInput
                  value={manualStrategy.dp}
                  onChange={(v) => v !== undefined && onManualStrategyChange({ ...manualStrategy, dp: v })}
                  min={1}
                  max={1024}
                  className="h-7 text-center"
                />
              </div>
              {/* TP */}
              <div className="text-center">
                <HelpTooltip
                  label="TP"
                  content="张量并行 (Attention)"
                  labelClassName="text-[11px] block mb-1 cursor-help"
                />
                <NumberInput
                  value={manualStrategy.tp}
                  onChange={(v) => v !== undefined && onManualStrategyChange({ ...manualStrategy, tp: v })}
                  min={1}
                  max={128}
                  className="h-7 text-center"
                />
              </div>
              {/* EP - 仅 MoE */}
              {isMoE && (
                <div className="text-center">
                  <HelpTooltip
                    label="EP"
                    content="专家并行"
                    labelClassName="text-[11px] block mb-1 cursor-help"
                  />
                  <NumberInput
                    value={manualStrategy.ep}
                    onChange={(v) => v !== undefined && onManualStrategyChange({ ...manualStrategy, ep: v })}
                    min={1}
                    max={256}
                    className="h-7 text-center"
                  />
                </div>
              )}
              {/* moe_tp - 仅 MoE */}
              {isMoE && (
                <div className="text-center">
                  <HelpTooltip
                    label="MoE_TP"
                    content="MoE 专家内张量并行"
                    labelClassName="text-[11px] block mb-1 cursor-help"
                  />
                  <NumberInput
                    value={manualStrategy.moe_tp || 1}
                    onChange={(v) => v !== undefined && onManualStrategyChange({ ...manualStrategy, moe_tp: v })}
                    min={1}
                    max={128}
                    className="h-7 text-center"
                  />
                </div>
              )}
            </div>
            <div className="mt-2 text-center">
              <span className={`text-xs ${totalParallelism > maxChips ? 'text-red-500' : 'text-gray-500'}`}>
                总芯片数: {totalParallelism} / {maxChips}
              </span>
              {isMoE && (
                <div className="mt-1">
                  <span
                    className={`text-[11px] ${
                      manualStrategy.dp * manualStrategy.tp === (manualStrategy.moe_tp || 1) * manualStrategy.ep
                        ? 'text-green-600'
                        : 'text-red-500'
                    }`}
                  >
                    {manualStrategy.dp * manualStrategy.tp === (manualStrategy.moe_tp || 1) * manualStrategy.ep
                      ? '✓ MoE 约束满足'
                      : `✗ 约束不满足: ${manualStrategy.dp}×${manualStrategy.tp}=${manualStrategy.dp * manualStrategy.tp} ≠ ${manualStrategy.moe_tp || 1}×${manualStrategy.ep}=${(manualStrategy.moe_tp || 1) * manualStrategy.ep}`}
                  </span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div>
            {/* 搜索约束显示（仅自动搜索模式） */}
            {mode === 'auto' && (
              <div>
                <div
                  className="flex justify-between items-center cursor-pointer"
                  onClick={() => setShowConstraints(!showConstraints)}
                >
                  <span className="text-xs">搜索约束</span>
                  <span className="text-[11px] text-gray-500">{showConstraints ? '▲' : '▼'}</span>
                </div>

              {showConstraints && (
                <div className="mt-2 p-2 bg-gray-50 rounded-md">
                  <div className="grid gap-1.5 text-[11px]">
                    {/* 并行模式说明 */}
                    <div className="mb-1 pb-1 border-b border-dashed border-gray-300">
                      <span className="text-[10px] text-gray-500 ml-2">
                        DP × TP {isMoE ? '× EP' : ''}（无 PP/SP）
                      </span>
                    </div>

                    {/* TP 约束 */}
                    <div className="flex justify-between">
                      <span>TP 约束</span>
                      <span className="text-gray-500">
                        ≤ min(128, 头数{modelConfig.num_attention_heads}, Board内芯片数{maxChips}) = {maxTP}
                        <span className="mx-3">&</span>
                        头数 {modelConfig.num_attention_heads} % TP == 0
                      </span>
                    </div>

                    {/* EP 约束 - 仅 MoE */}
                    {isMoE && (
                      <div className="flex justify-between">
                        <span>EP 约束</span>
                        <span className="text-gray-500">
                          ≤ 专家数 {maxEP}
                          <span className="mx-3">&</span>
                          专家数 {maxEP} % EP == 0
                        </span>
                      </div>
                    )}

                    {/* moe_tp 约束 - 仅 MoE */}
                    {isMoE && (
                      <div className="flex justify-between">
                        <span>MoE_TP 约束</span>
                        <span className="text-gray-500">
                          ≤ min(128, 头数{modelConfig.num_attention_heads}, Board内芯片数{maxChips}) = {maxTP}
                          <span className="mx-3">&</span>
                          头数 {modelConfig.num_attention_heads} % MoE_TP == 0
                        </span>
                      </div>
                    )}

                    {/* MoE 约束 */}
                    {isMoE && (
                      <div className="flex justify-between">
                        <span>MoE 约束</span>
                        <span className="text-gray-500">DP × TP = MoE_TP × EP</span>
                      </div>
                    )}

                    {/* Board 约束 */}
                    <div className="flex justify-between">
                      <span>Board 内芯片数量</span>
                      <span className="text-gray-500">{maxChips} 个/Board</span>
                    </div>

                    {/* 当前搜索策略说明 */}
                    <div className="mt-1 pt-1.5 border-t border-dashed border-gray-300">
                      <span className="text-[10px] text-gray-500 leading-relaxed">
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
                      </span>
                    </div>
                  </div>
                </div>
              )}
              </div>
            )}
          </div>
        )}
    </div>
  )
}

export default ParallelismConfigPanel
