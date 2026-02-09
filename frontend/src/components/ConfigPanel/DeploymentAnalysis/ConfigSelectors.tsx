/**
 * 配置选择器 - 共享工具和遗留组件
 *
 * 共享导出：
 *   - colors, configRowStyle, sectionCardStyle, sectionTitleStyle (样式常量)
 *   - BENCHMARK_TOOLTIPS, ConfigLabel (UI 辅助)
 *   - getDtypeBits, formatSeqLen, generateBenchmarkName (工具函数)
 *
 * 遗留组件 (过渡期保留，后续迁移到新编辑器后删除)：
 *   - ModelConfigSelector (旧版，使用 LLMModelConfig)
 *   - HardwareConfigSelector (旧版，使用 HardwareConfig)
 *
 * 新编辑器在独立文件中：
 *   - ModelPresetEditor.tsx (替代 ModelConfigSelector)
 *   - BenchmarkEditor.tsx (提取自本文件)
 *   - ChipPresetEditor.tsx (替代 HardwareConfigSelector)
 *   - TopologyEditor.tsx (新增)
 */

import React, { useState, useCallback, useEffect } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { InfoTooltip, HelpTooltip } from '@/components/ui/info-tooltip'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
} from '../../../utils/llmDeployment/types'
import { calculateModelParams } from '../../../api/model'
import { useDebouncedValue } from '@/hooks/useDebouncedCallback'
import {
  getModelList,
  getChipList,
  getModelPreset,
  createHardwareConfig,
} from '../../../utils/llmDeployment/presets'

// ============================================
// 样式常量 - 从设计令牌和公共样式导入
// ============================================

import { colors as designColors } from '../../../utils/design-tokens'
import { sectionCardStyle as commonSectionCardStyle, sectionTitleStyle as commonSectionTitleStyle, configRowStyle as commonConfigRowStyle } from '../../ui/common-styles'

export const colors = designColors
export const configRowStyle = commonConfigRowStyle
export const sectionCardStyle = commonSectionCardStyle
export const sectionTitleStyle = commonSectionTitleStyle

// ============================================
// Benchmark 参数提示
// ============================================

export const BENCHMARK_TOOLTIPS: Record<string, string> = {
  model_name: '选择预设模型',
  model_type: 'Dense: 标准密集模型; MoE: 混合专家稀疏模型',
  hidden_size: 'Hidden Size: 隐藏层维度',
  num_layers: 'Num Layers: Transformer 层数',
  intermediate_size: 'Intermediate Size: FFN 中间层维度',
  vocab_size: 'Vocab Size: 词表大小',
  num_attention_heads: 'Num Attention Heads: 注意力头数',
  num_kv_heads: 'Num KV Heads: KV 头数 (GQA)',
  attention_type: 'Attention 类型: MHA/GQA/MQA/MLA',
  weight_dtype: '权重存储精度',
  activation_dtype: '激活/KV Cache 精度',
  mla_variant: 'MLA 实现变体',
  kv_lora_rank: 'KV LoRA Rank: KV 压缩后的隐维度',
  q_lora_rank: 'Q LoRA Rank: Query 的 LoRA rank',
  qk_nope_head_dim: 'QK Nope Head Dim: 非 RoPE 头维度',
  qk_rope_head_dim: 'QK RoPE Head Dim: RoPE 头维度',
  v_head_dim: 'V Head Dim: Value 头维度',
  num_experts: 'Num Experts: 专家总数',
  num_experts_per_tok: 'Top-K: 每个 token 激活的专家数',
  num_shared_experts: 'Shared Experts: 共享专家数量',
  expert_intermediate_size: 'Expert FFN Size: 专家 FFN 维度',
  first_k_dense_replace: 'First K Dense: 前 K 层使用 Dense 替代 MoE',
  batch_size: 'Batch Size: 批处理大小',
  input_seq_length: '输入 Token 数量',
  output_seq_length: '输出 Token 数量',
  max_seq_length: 'KV Cache 最大长度',
}

// ConfigLabel 组件 - 使用统一的 HelpTooltip
interface ConfigLabelProps {
  name: string
  label?: string
}

export const ConfigLabel: React.FC<ConfigLabelProps> = ({ name, label }) => (
  <HelpTooltip
    label={label || name}
    content={BENCHMARK_TOOLTIPS[name] || name}
  />
)

// ============================================
// 工具函数
// ============================================

export function formatSeqLen(len: number): string {
  if (len >= 1024 && len % 1024 === 0) return `${len / 1024}K`
  return String(len)
}

export function getDtypeBits(dtype: string): number {
  const bitsMap: Record<string, number> = { 'fp32': 32, 'fp16': 16, 'bf16': 16, 'fp8': 8, 'int8': 8, 'int4': 4 }
  return bitsMap[dtype] || 16
}

function parseModelName(modelName: string): { name: string; size: string } {
  const sizeMatch = modelName.match(/(\d+\.?\d*)[BMK]/i)
  if (sizeMatch) {
    const size = sizeMatch[1] + sizeMatch[0].slice(-1).toUpperCase()
    const name = modelName.replace(/[-_]?\d+\.?\d*[BMK][-_]?/i, '').replace(/-+$/, '').replace(/^-+/, '').replace(/-Instruct|-Chat|-Base/i, '').trim()
    return { name, size }
  }
  return { name: modelName, size: '' }
}

export function generateBenchmarkName(model: LLMModelConfig, inference: InferenceConfig): string {
  if (!model?.model_name) return 'Unknown-Model'
  const { name, size } = parseModelName(model.model_name)
  const seqIn = formatSeqLen(inference.input_seq_length)
  const seqOut = formatSeqLen(inference.output_seq_length)
  const wBits = getDtypeBits(model.weight_dtype)
  const aBits = getDtypeBits(model.activation_dtype)
  const parts = [size ? `${name}-${size}` : name, `S${seqIn}`, `O${seqOut}`, `W${wBits}A${aBits}`, `B${inference.batch_size}`]
  return parts.join('-')
}

// ============================================
// 自定义模型存储 (遗留)
// ============================================

const CUSTOM_MODELS_KEY = 'llm_custom_models'

function loadCustomModels(): Record<string, LLMModelConfig> {
  try {
    const data = localStorage.getItem(CUSTOM_MODELS_KEY)
    return data ? JSON.parse(data) : {}
  } catch {
    return {}
  }
}

function saveCustomModels(models: Record<string, LLMModelConfig>) {
  localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(models))
}

// ============================================
// 模型配置选择器 (遗留，使用 LLMModelConfig)
// ============================================

interface ModelConfigSelectorProps {
  value: LLMModelConfig
  onChange: (config: LLMModelConfig) => void
}

export const ModelConfigSelector: React.FC<ModelConfigSelectorProps> = ({ value, onChange }) => {
  const [presetId, setPresetId] = useState<string>('deepseek-v3')
  const [editMode, setEditMode] = useState<boolean>(false)
  const [editBackup, setEditBackup] = useState<LLMModelConfig | null>(null)
  const [customModels, setCustomModels] = useState<Record<string, LLMModelConfig>>(loadCustomModels)
  const [saveModalVisible, setSaveModalVisible] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [paramsStr, setParamsStr] = useState<string>('--')
  const modelList = getModelList()

  const debouncedModelConfig = useDebouncedValue(value, 300)

  useEffect(() => {
    let cancelled = false
    calculateModelParams(debouncedModelConfig)
      .then((res) => {
        if (!cancelled) setParamsStr(res.formatted)
      })
      .catch(() => {
        if (!cancelled) setParamsStr('--')
      })
    return () => { cancelled = true }
  }, [debouncedModelConfig])

  const handlePresetChange = (id: string) => {
    setPresetId(id)
    if (customModels[id]) {
      onChange({ ...customModels[id] })
    } else {
      onChange(getModelPreset(id))
    }
  }

  const handleSaveModel = () => {
    if (!saveName.trim()) return
    const customId = `custom_${saveName.trim().toLowerCase().replace(/\s+/g, '_')}`
    const newModels = { ...customModels, [customId]: { ...value, model_name: saveName.trim() } }
    setCustomModels(newModels)
    saveCustomModels(newModels)
    setPresetId(customId)
    setSaveModalVisible(false)
    setSaveName('')
  }

  const handleDeleteCustomModel = (id: string) => {
    const newModels = { ...customModels }
    delete newModels[id]
    setCustomModels(newModels)
    saveCustomModels(newModels)
    if (presetId === id) {
      setPresetId('deepseek-v3')
      onChange(getModelPreset('deepseek-v3'))
    }
  }

  const allModelOptions = [
    ...Object.entries(customModels).map(([id, config]) => ({
      value: id,
      label: `[自定义] ${config.model_name}`,
      isCustom: true,
    })),
    ...modelList.map(m => ({
      value: m.id,
      label: m.params ? `${m.name} (${m.params})` : m.name,
      isCustom: false,
    })),
  ]

  const updateField = <K extends keyof LLMModelConfig>(field: K, val: LLMModelConfig[K]) => {
    onChange({ ...value, [field]: val })
  }

  const updateMoeField = <K extends keyof NonNullable<LLMModelConfig['moe_config']>>(
    field: K,
    val: NonNullable<LLMModelConfig['moe_config']>[K]
  ) => {
    if (value.moe_config) {
      onChange({ ...value, moe_config: { ...value.moe_config, [field]: val } })
    }
  }

  const updateMlaField = <K extends keyof NonNullable<LLMModelConfig['mla_config']>>(
    field: K,
    val: NonNullable<LLMModelConfig['mla_config']>[K]
  ) => {
    if (value.mla_config) {
      onChange({ ...value, mla_config: { ...value.mla_config, [field]: val } })
    }
  }

  const isCustomModel = presetId.startsWith('custom_')
  const estimateParams = () => ({ value: paramsStr, breakdown: `总参数量: ${paramsStr}` })

  return (
    <div>
        <div style={configRowStyle}>
          <span className="text-sm">模型选择</span>
          <Select value={presetId} onValueChange={handlePresetChange}>
            <SelectTrigger className="w-[280px] h-7">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {allModelOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  <div className="flex justify-between items-center w-full">
                    <span>{opt.label}</span>
                    {opt.isCustom && (
                      <Button
                        variant="link"
                        size="sm"
                        className="h-5 p-0 text-red-500"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDeleteCustomModel(opt.value)
                        }}
                      >
                        删除
                      </Button>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {editMode ? (
          <div className="p-2 bg-gray-100 rounded-md text-xs">
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="模型名称" content="自定义模型显示名称" labelClassName="text-[13px] cursor-help" />
              <Input value={value.model_name} onChange={(e) => updateField('model_name', e.target.value)} className="w-[180px] h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="模型类型" content="Dense: 标准密集模型; MoE: 混合专家稀疏模型" labelClassName="text-[13px] cursor-help" />
              <Select value={value.model_type} onValueChange={(v) => {
                if (v === 'moe' && !value.moe_config) {
                  onChange({ ...value, model_type: v, moe_config: { num_experts: 8, num_experts_per_tok: 2, expert_capacity_factor: 1.25 } })
                } else {
                  updateField('model_type', v as any)
                }
              }}>
                <SelectTrigger className="w-20 h-7"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="dense">Dense</SelectItem>
                  <SelectItem value="moe">MoE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="隐藏层维度" content="Hidden Size" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={64} value={value.hidden_size} onChange={(v) => updateField('hidden_size', v || 4096)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="层数" content="Num Layers" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={1} value={value.num_layers} onChange={(v) => updateField('num_layers', v || 32)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="注意力头数" content="Num Attention Heads" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={1} value={value.num_attention_heads} onChange={(v) => updateField('num_attention_heads', v || 32)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="KV头数" content="Num KV Heads" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={1} value={value.num_kv_heads} onChange={(v) => updateField('num_kv_heads', v || 8)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="FFN维度" content="Intermediate Size" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={64} value={value.intermediate_size} onChange={(v) => updateField('intermediate_size', v || 11008)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="词表大小" content="Vocab Size" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={1000} value={value.vocab_size} onChange={(v) => updateField('vocab_size', v || 32000)} className="w-20 h-7" />
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="权重精度" content="权重精度" labelClassName="text-[13px] cursor-help" />
              <Select value={value.weight_dtype} onValueChange={(v) => updateField('weight_dtype', v as any)}>
                <SelectTrigger className="w-[90px] h-7"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fp32">FP32</SelectItem>
                  <SelectItem value="bf16">BF16</SelectItem>
                  <SelectItem value="fp16">FP16</SelectItem>
                  <SelectItem value="fp8">FP8</SelectItem>
                  <SelectItem value="int8">INT8</SelectItem>
                  <SelectItem value="int4">INT4</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="激活精度" content="激活精度" labelClassName="text-[13px] cursor-help" />
              <Select value={value.activation_dtype} onValueChange={(v) => updateField('activation_dtype', v as any)}>
                <SelectTrigger className="w-[90px] h-7"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="fp32">FP32</SelectItem>
                  <SelectItem value="bf16">BF16</SelectItem>
                  <SelectItem value="fp16">FP16</SelectItem>
                  <SelectItem value="fp8">FP8</SelectItem>
                  <SelectItem value="int8">INT8</SelectItem>
                  <SelectItem value="int4">INT4</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex justify-between items-center mb-1">
              <HelpTooltip label="最大序列长度" content="Max Sequence Length" labelClassName="text-[13px] cursor-help" />
              <NumberInput min={128} value={value.max_seq_length} onChange={(v) => updateField('max_seq_length', v || 4096)} className="w-20 h-7" />
            </div>

            {value.model_type === 'moe' && value.moe_config && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <Badge variant="outline" className="mb-1.5 bg-purple-50 text-purple-700 border-purple-200">MoE 参数</Badge>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="专家数量" content="Num Experts" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={2} value={value.moe_config.num_experts} onChange={(v) => updateMoeField('num_experts', v || 8)} className="w-20 h-7" />
                </div>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="激活专家数" content="Top-K" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={1} value={value.moe_config.num_experts_per_tok} onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} className="w-20 h-7" />
                </div>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="共享专家数" content="Shared Experts" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={0} value={value.moe_config.num_shared_experts || 0} onChange={(v) => updateMoeField('num_shared_experts', v || 0)} className="w-20 h-7" />
                </div>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="专家FFN维度" content="Expert FFN Size" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={64} value={value.moe_config.expert_intermediate_size} onChange={(v) => updateMoeField('expert_intermediate_size', v)} className="w-20 h-7" placeholder="同FFN" />
                </div>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="前K层Dense" content="First K Dense" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={0} value={value.moe_config.first_k_dense_replace || 0} onChange={(v) => updateMoeField('first_k_dense_replace', v || 0)} className="w-20 h-7" />
                </div>
              </div>
            )}

            {value.attention_type === 'mla' && value.mla_config && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <Badge variant="outline" className="mb-1.5 bg-cyan-50 text-cyan-700 border-cyan-200">MLA 并行度</Badge>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="MLA TP" content="MLA TP" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={1} value={value.mla_config.mla_tp} onChange={(v) => updateMlaField('mla_tp', v)} className="w-20 h-7" placeholder="同TP" />
                </div>
                <div className="flex justify-between items-center mb-1">
                  <HelpTooltip label="MLA DP" content="MLA DP" labelClassName="text-[13px] cursor-help" />
                  <NumberInput min={1} value={value.mla_config.mla_dp} onChange={(v) => updateMlaField('mla_dp', v)} className="w-20 h-7" placeholder="同DP" />
                </div>
                <span className="text-gray-400 text-[10px]">约束: MLA_TP x MLA_DP = TP x DP</span>
              </div>
            )}
            <div className="mt-2 pt-2 border-t border-gray-200">
              <InfoTooltip content={<pre className="whitespace-pre-wrap m-0 text-[13px]">{estimateParams().breakdown}</pre>}><span className="text-gray-500 cursor-help">估算参数量: <b>{estimateParams().value}</b></span></InfoTooltip>
            </div>
          </div>
        ) : (
          <div className="p-2 bg-gray-100 rounded-md text-xs">
            <div className="grid grid-cols-2 gap-1">
              <InfoTooltip content="Hidden Size"><span className="text-gray-500 cursor-help">隐藏层: {value.hidden_size}</span></InfoTooltip>
              <InfoTooltip content="Num Layers"><span className="text-gray-500 cursor-help">层数: {value.num_layers}</span></InfoTooltip>
              <InfoTooltip content="Num Attention Heads"><span className="text-gray-500 cursor-help">注意力头: {value.num_attention_heads}</span></InfoTooltip>
              <InfoTooltip content="Num KV Heads"><span className="text-gray-500 cursor-help">KV头: {value.num_kv_heads}</span></InfoTooltip>
              <InfoTooltip content="Intermediate Size"><span className="text-gray-500 cursor-help">FFN: {value.intermediate_size}</span></InfoTooltip>
              <InfoTooltip content="W=权重精度, A=激活精度"><span className="text-gray-500 cursor-help">精度: W{getDtypeBits(value.weight_dtype)}A{getDtypeBits(value.activation_dtype)}</span></InfoTooltip>
            </div>
            {value.model_type === 'moe' && value.moe_config && (
              <div className="mt-1 pt-1 border-t border-gray-200">
                <InfoTooltip content="Mixture of Experts"><Badge variant="outline" className="cursor-help bg-purple-50 text-purple-700 border-purple-200">MoE</Badge></InfoTooltip>
                <InfoTooltip content={`总共${value.moe_config.num_experts}个专家，每个token激活${value.moe_config.num_experts_per_tok}个`}><span className="text-gray-500 text-[13px] ml-1 cursor-help">{value.moe_config.num_experts}专家 x {value.moe_config.num_experts_per_tok}激活{value.moe_config.num_shared_experts ? ` + ${value.moe_config.num_shared_experts}共享` : ''}</span></InfoTooltip>
              </div>
            )}
            <div className="mt-1.5 pt-1.5 border-t border-gray-200 flex justify-between items-center">
              <InfoTooltip content={<pre className="whitespace-pre-wrap m-0 text-[13px]">{estimateParams().breakdown}</pre>}><span className="text-gray-500 cursor-help">估算参数量: <b>{estimateParams().value}</b></span></InfoTooltip>
            </div>
          </div>
        )}

        <div className="flex gap-1 mt-2">
          {editMode && (
            <Button variant="outline" size="sm" className="flex-1" onClick={() => {
              if (editBackup) onChange(editBackup)
              setEditMode(false)
              setEditBackup(null)
            }}>取消</Button>
          )}
          <Button variant={editMode ? 'default' : 'outline'} size="sm" className="flex-1" onClick={() => {
            if (!editMode) setEditBackup({ ...value })
            setEditMode(!editMode)
          }}>{editMode ? '完成编辑' : '编辑模型参数'}</Button>
          {isCustomModel && (
            <Button size="sm" className="flex-1" onClick={() => {
              const newModels = { ...customModels, [presetId]: { ...value } }
              setCustomModels(newModels)
              saveCustomModels(newModels)
            }}>保存</Button>
          )}
          {editMode && (
            <Button variant="outline" size="sm" className="flex-1" onClick={() => {
              setSaveName(value.model_name)
              setSaveModalVisible(true)
            }}>另存为</Button>
          )}
          {isCustomModel && (
            <Button variant="destructive" size="sm" onClick={() => handleDeleteCustomModel(presetId)}>删除</Button>
          )}
        </div>

        {saveModalVisible && (
          <div className="p-3 bg-white border border-gray-300 rounded-md mt-2">
            <div className="mb-2"><span className="text-xs">输入自定义模型名称：</span></div>
            <Input value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="如: My-Custom-Model" className="mb-2" onKeyDown={(e) => e.key === 'Enter' && handleSaveModel()} />
            <div className="flex gap-2 justify-end">
              <Button variant="outline" size="sm" onClick={() => setSaveModalVisible(false)}>取消</Button>
              <Button size="sm" onClick={handleSaveModel}>保存</Button>
            </div>
          </div>
        )}
      </div>
  )
}

// ============================================
// 硬件配置选择器 (遗留)
// ============================================

interface HardwareConfigSelectorProps {
  value: HardwareConfig
  onChange: (config: HardwareConfig) => void
}

export const HardwareConfigSelector: React.FC<HardwareConfigSelectorProps> = ({ value, onChange }) => {
  const [chipId, setChipId] = useState<string>('h100-sxm')
  const [boardId, setBoardId] = useState<string>('dgx-h100')
  const [rackId, setRackId] = useState<string>('ib-ndr')
  const [podId, setPodId] = useState<string>('ib-ndr')

  const chipList = getChipList()
  const boardOptions = [
    { value: 'dgx-h100', label: 'DGX H100 (8卡 NVLink)' },
    { value: 'dgx-a100', label: 'DGX A100 (8卡 NVLink)' },
    { value: 'pcie-8gpu', label: '通用 PCIe (8卡)' },
  ]

  const handleConfigChange = useCallback((newChipId: string, newBoardId: string, newRackId: string, newPodId: string) => {
    const config = createHardwareConfig(newChipId, newBoardId, newRackId, newPodId)
    onChange(config)
  }, [onChange])

  const handleChipChange = (id: string) => { setChipId(id); handleConfigChange(id, boardId, rackId, podId) }
  const handleBoardChange = (id: string) => { setBoardId(id); handleConfigChange(chipId, id, rackId, podId) }
  const handleRackChange = (id: string) => { setRackId(id); handleConfigChange(chipId, boardId, id, podId) }
  const handlePodChange = (id: string) => { setPodId(id); handleConfigChange(chipId, boardId, rackId, id) }

  const totalChips = (value.board?.chips_per_board || 8) * (value.pod?.racks_per_pod || 1) * (value.rack?.boards_per_rack || 4)

  return (
    <div>
      <div style={configRowStyle}>
        <span className="text-sm">芯片类型</span>
        <Select value={chipId} onValueChange={handleChipChange}>
          <SelectTrigger className="w-[140px] h-7"><SelectValue /></SelectTrigger>
          <SelectContent>{chipList.map(c => (<SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>))}</SelectContent>
        </Select>
      </div>
      <div style={configRowStyle}>
        <span className="text-sm">Board 类型</span>
        <Select value={boardId} onValueChange={handleBoardChange}>
          <SelectTrigger className="w-[160px] h-7"><SelectValue /></SelectTrigger>
          <SelectContent>{boardOptions.map(o => (<SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>))}</SelectContent>
        </Select>
      </div>
      <div className="p-2 bg-blue-50 rounded-md text-xs mt-2">
        <div className="flex justify-between">
          <span className="text-gray-500">总芯片数: <b>{totalChips}</b></span>
          <span className="text-gray-500">显存: <b>{value.chip?.memory.gmem.capacity_gb || 64}GB</b>/卡</span>
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-gray-500">算力: {value.chip?.compute_units.cube.mac_per_lane.BF16 ?
            (value.chip.frequency_ghz * value.chip.cores.count * value.chip.cores.lanes_per_core *
             value.chip.compute_units.cube.mac_per_lane.BF16 *
             (value.chip.compute_units.cube.m || 16) * (value.chip.compute_units.cube.n || 8) * 2 / 1000).toFixed(0) : 768} TFLOPs</span>
          <span className="text-gray-500">Board内: {value.board?.b2b_bandwidth_gbps || 900} GB/s</span>
        </div>
      </div>
    </div>
  )
}
