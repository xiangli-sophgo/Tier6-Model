/**
 * 配置选择器组件
 *
 * 包含：ModelConfigSelector, BenchmarkConfigSelector, HardwareConfigSelector
 */

import React, { useState, useCallback, useEffect } from 'react'
import { Save, Copy, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react'
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
import { BaseCard } from '@/components/common/BaseCard'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
} from '../../../utils/llmDeployment/types'
import { calculateModelParams } from '../../../api/model'
import {
  getModelList,
  getChipList,
  getModelPreset,
  createHardwareConfig,
} from '../../../utils/llmDeployment/presets'
import { listBenchmarks, createBenchmark } from '../../../api/topology'
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

const BENCHMARK_TOOLTIPS: Record<string, string> = {
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

const ConfigLabel: React.FC<ConfigLabelProps> = ({ name, label }) => (
  <HelpTooltip
    label={label || name}
    content={BENCHMARK_TOOLTIPS[name] || name}
  />
)


// ============================================
// 自定义模型存储
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
// 模型配置选择器
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

  useEffect(() => {
    let cancelled = false
    calculateModelParams(value)
      .then((res) => {
        if (!cancelled) setParamsStr(res.formatted)
      })
      .catch(() => {
        if (!cancelled) setParamsStr('--')
      })
    return () => { cancelled = true }
  }, [value])

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
                <span className="text-gray-400 text-[10px]">约束: MLA_TP × MLA_DP = TP × DP</span>
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
                <InfoTooltip content={`总共${value.moe_config.num_experts}个专家，每个token激活${value.moe_config.num_experts_per_tok}个`}><span className="text-gray-500 text-[13px] ml-1 cursor-help">{value.moe_config.num_experts}专家 × {value.moe_config.num_experts_per_tok}激活{value.moe_config.num_shared_experts ? ` + ${value.moe_config.num_shared_experts}共享` : ''}</span></InfoTooltip>
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
// Benchmark 配置选择器 (合并模型+推理参数)
// ============================================

interface BenchmarkConfigSelectorProps {
  modelConfig: LLMModelConfig
  onModelChange: (config: LLMModelConfig) => void
  inferenceConfig: InferenceConfig
  onInferenceChange: (config: InferenceConfig) => void
  // 通知父组件当前选中的 Benchmark 配置文件名
  onBenchmarkSelect?: (benchmarkName: string | undefined) => void
}

const LAST_BENCHMARK_KEY = 'llm_last_benchmark_id'

function formatSeqLen(len: number): string {
  if (len >= 1024 && len % 1024 === 0) return `${len / 1024}K`
  return String(len)
}

function getDtypeBits(dtype: string): number {
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
  const { name, size } = parseModelName(model.model_name)
  const seqIn = formatSeqLen(inference.input_seq_length)
  const seqOut = formatSeqLen(inference.output_seq_length)
  const wBits = getDtypeBits(model.weight_dtype)
  const aBits = getDtypeBits(model.activation_dtype)
  const parts = [size ? `${name}-${size}` : name, `S${seqIn}`, `O${seqOut}`, `W${wBits}A${aBits}`, `B${inference.batch_size}`]
  return parts.join('-')
}

interface CustomBenchmark {
  id: string
  name: string
  model: LLMModelConfig
  inference: InferenceConfig
}

export const BenchmarkConfigSelector: React.FC<BenchmarkConfigSelectorProps> = ({
  modelConfig,
  onModelChange,
  inferenceConfig,
  onInferenceChange,
  onBenchmarkSelect,
}) => {
  const [presetId, setPresetId] = useState<string>('')
  const [customBenchmarks, setCustomBenchmarks] = useState<CustomBenchmark[]>([])
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    basic: false,
    attention: false,
    precision: false,
    inference: false,
    moe: false,
  })
  const [paramsStr, setParamsStr] = useState<string>('--')

  // 原始配置快照（用于修改追踪）
  const [originalConfig, setOriginalConfig] = useState<{
    model: LLMModelConfig | null
    inference: InferenceConfig | null
  }>({ model: null, inference: null })

  const modelList = getModelList()

  useEffect(() => {
    let cancelled = false
    calculateModelParams(modelConfig)
      .then((res) => { if (!cancelled) setParamsStr(res.formatted) })
      .catch(() => { if (!cancelled) setParamsStr('--') })
    return () => { cancelled = true }
  }, [modelConfig])

  useEffect(() => {
    listBenchmarks().then((benchmarks) => {
      const mapped = benchmarks.map(b => ({
        id: b.id,
        name: b.name,
        model: b.model as unknown as LLMModelConfig,
        inference: b.inference as unknown as InferenceConfig,
      }))
      setCustomBenchmarks(mapped)
      if (mapped.length > 0) {
        const lastBenchmarkId = localStorage.getItem(LAST_BENCHMARK_KEY)
        const initialId = (lastBenchmarkId && mapped.find(b => b.id === lastBenchmarkId)) ? lastBenchmarkId : mapped[0].id
        setPresetId(initialId)
        // 通知父组件选中的 Benchmark
        onBenchmarkSelect?.(initialId)
        const initialBenchmark = mapped.find(b => b.id === initialId)
        if (initialBenchmark) {
          onModelChange(initialBenchmark.model)
          onInferenceChange(initialBenchmark.inference)
          // 保存原始配置快照（用于修改追踪）
          setOriginalConfig({
            model: { ...initialBenchmark.model },
            inference: { ...initialBenchmark.inference },
          })
        }
      }
    })
  }, [])

  const currentBenchmarkName = generateBenchmarkName(modelConfig, inferenceConfig)

  const isConfigModified = useCallback(() => {
    const match = customBenchmarks.find(c => c.id === presetId)
    if (match) return generateBenchmarkName(match.model, match.inference) !== currentBenchmarkName
    return true
  }, [presetId, currentBenchmarkName, customBenchmarks])

  const handlePresetChange = (id: string) => {
    setPresetId(id)
    localStorage.setItem(LAST_BENCHMARK_KEY, id)
    // 通知父组件选中的 Benchmark
    onBenchmarkSelect?.(id)
    const match = customBenchmarks.find(c => c.id === id)
    if (match) {
      onModelChange(match.model)
      onInferenceChange(match.inference)
      // 保存原始配置快照（用于修改追踪）
      setOriginalConfig({
        model: { ...match.model },
        inference: { ...match.inference },
      })
    }
  }

  const handleSave = async () => {
    if (isConfigModified()) {
      const newBenchmark: CustomBenchmark = { id: currentBenchmarkName, name: currentBenchmarkName, model: { ...modelConfig }, inference: { ...inferenceConfig } }
      const success = await createBenchmark({ id: currentBenchmarkName, name: currentBenchmarkName, model: modelConfig as unknown as Record<string, unknown>, inference: inferenceConfig as unknown as Record<string, unknown> })
      if (success) {
        const existingIndex = customBenchmarks.findIndex(b => b.id === currentBenchmarkName)
        if (existingIndex >= 0) {
          const updated = [...customBenchmarks]
          updated[existingIndex] = newBenchmark
          setCustomBenchmarks(updated)
        } else {
          setCustomBenchmarks([...customBenchmarks, newBenchmark])
        }
        setPresetId(currentBenchmarkName)
        localStorage.setItem(LAST_BENCHMARK_KEY, currentBenchmarkName)
        // 通知父组件选中的 Benchmark
        onBenchmarkSelect?.(currentBenchmarkName)
        toast.success(`已保存: ${currentBenchmarkName}`)
      } else {
        toast.error('保存失败')
      }
    }
  }

  const handleReset = () => {
    const match = customBenchmarks.find(c => c.id === presetId)
    if (match) {
      onModelChange(match.model)
      onInferenceChange(match.inference)
      toast.info('已重置到原始配置')
    }
  }

  const handleSaveAs = async () => {
    const newBenchmark: CustomBenchmark = { id: currentBenchmarkName, name: currentBenchmarkName, model: { ...modelConfig }, inference: { ...inferenceConfig } }
    const success = await createBenchmark({ id: currentBenchmarkName, name: currentBenchmarkName, model: modelConfig as unknown as Record<string, unknown>, inference: inferenceConfig as unknown as Record<string, unknown> })
    if (success) {
      const existingIndex = customBenchmarks.findIndex(b => b.id === currentBenchmarkName)
      if (existingIndex >= 0) {
        const updated = [...customBenchmarks]
        updated[existingIndex] = newBenchmark
        setCustomBenchmarks(updated)
      } else {
        setCustomBenchmarks([...customBenchmarks, newBenchmark])
      }
      setPresetId(currentBenchmarkName)
      localStorage.setItem(LAST_BENCHMARK_KEY, currentBenchmarkName)
      // 通知父组件选中的 Benchmark
      onBenchmarkSelect?.(currentBenchmarkName)
      toast.success(`已保存: ${currentBenchmarkName}`)
    } else {
      toast.error('保存失败')
    }
  }

  const dropdownOptions = customBenchmarks.map(c => ({ value: c.id, label: c.name }))
  const updateModelField = <K extends keyof LLMModelConfig>(field: K, val: LLMModelConfig[K]) => { onModelChange({ ...modelConfig, [field]: val }) }
  const updateMoeField = <K extends keyof NonNullable<LLMModelConfig['moe_config']>>(field: K, val: NonNullable<LLMModelConfig['moe_config']>[K]) => {
    if (modelConfig.moe_config) onModelChange({ ...modelConfig, moe_config: { ...modelConfig.moe_config, [field]: val } })
  }

  const toggleSection = (key: string) => { setOpenSections(prev => ({ ...prev, [key]: !prev[key] })) }

  // 检测模型字段是否被修改
  const isModelFieldModified = (fieldName: keyof LLMModelConfig): boolean => {
    if (!originalConfig.model) return false
    return modelConfig[fieldName] !== originalConfig.model[fieldName]
  }

  // 检测推理字段是否被修改
  const isInferenceFieldModified = (fieldName: keyof InferenceConfig): boolean => {
    if (!originalConfig.inference) return false
    return inferenceConfig[fieldName] !== originalConfig.inference[fieldName]
  }

  // 检测MoE字段是否被修改
  const isMoeFieldModified = (fieldName: string): boolean => {
    if (!originalConfig.model || !originalConfig.model.moe_config || !modelConfig.moe_config) return false
    return (modelConfig.moe_config as any)[fieldName] !== (originalConfig.model.moe_config as any)[fieldName]
  }

  // 检测MLA字段是否被修改
  const isMlaFieldModified = (fieldName: string): boolean => {
    if (!originalConfig.model || !originalConfig.model.mla_config || !modelConfig.mla_config) return false
    return (modelConfig.mla_config as any)[fieldName] !== (originalConfig.model.mla_config as any)[fieldName]
  }

  const sections = [
    { key: 'basic', label: '基础参数' },
    { key: 'attention', label: '注意力配置' },
    { key: 'precision', label: '精度配置' },
    { key: 'inference', label: '推理参数' },
  ]

  return (
    <div>
        <div className="mb-3">
          <div className="mb-1 flex justify-between items-center">
            <span className="text-gray-500 text-xs"><span className="text-red-500">*</span> Benchmark 配置文件</span>
            <Button variant="link" size="sm" className="p-0 h-auto text-xs" onClick={() => {
              const allOpen = sections.every(s => openSections[s.key])
              const newState: Record<string, boolean> = {}
              sections.forEach(s => { newState[s.key] = !allOpen })
              setOpenSections(newState)
            }}>
              {sections.every(s => openSections[s.key]) ? <><ChevronUp className="h-3 w-3 mr-1" />全部折叠</> : <><ChevronDown className="h-3 w-3 mr-1" />全部展开</>}
            </Button>
          </div>
          <Select value={presetId} onValueChange={handlePresetChange}>
            <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
            <SelectContent>
              {dropdownOptions.map((opt) => (<SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>))}
            </SelectContent>
          </Select>
        </div>

        {/* 折叠面板 */}
        <div className="space-y-2 mb-3">
          {/* 基础参数 */}
          <BaseCard
            title="基础参数"
            collapsible
            expanded={openSections.basic}
            onExpandChange={() => toggleSection('basic')}
            contentClassName="p-2"
            gradient
          >
                <div className="grid grid-cols-2 gap-2">
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('model_name') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="model_name" label="模型选择" />
                      {isModelFieldModified('model_name') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                          已修改
                        </Badge>
                      )}
                    </div>
                    <Select value={modelConfig.model_name} onValueChange={(name) => { const preset = modelList.find(m => m.name === name || m.id === name); if (preset) onModelChange(getModelPreset(preset.id)) }}>
                      <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                      <SelectContent>{modelList.map(m => (<SelectItem key={m.id} value={m.name}>{m.params ? `${m.name} (${m.params})` : m.name}</SelectItem>))}</SelectContent>
                    </Select>
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('model_type') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="model_type" label="模型类型" />
                      {isModelFieldModified('model_type') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">
                          已修改
                        </Badge>
                      )}
                    </div>
                    <Select value={modelConfig.model_type} onValueChange={(v) => {
                      if (v === 'moe' && !modelConfig.moe_config) onModelChange({ ...modelConfig, model_type: v, moe_config: { num_experts: 8, num_experts_per_tok: 2, expert_capacity_factor: 1.25 } })
                      else updateModelField('model_type', v as any)
                    }}>
                      <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                      <SelectContent><SelectItem value="dense">Dense</SelectItem><SelectItem value="moe">MoE</SelectItem></SelectContent>
                    </Select>
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('hidden_size') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="hidden_size" label="隐藏层维度" />
                      {isModelFieldModified('hidden_size') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={64} value={modelConfig.hidden_size} onChange={(v) => updateModelField('hidden_size', v || 4096)} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('num_layers') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="num_layers" label="层数" />
                      {isModelFieldModified('num_layers') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={modelConfig.num_layers} onChange={(v) => updateModelField('num_layers', v || 32)} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('intermediate_size') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="intermediate_size" label="FFN维度" />
                      {isModelFieldModified('intermediate_size') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={64} value={modelConfig.intermediate_size} onChange={(v) => updateModelField('intermediate_size', v || 11008)} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('vocab_size') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="vocab_size" label="词表大小" />
                      {isModelFieldModified('vocab_size') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1000} value={modelConfig.vocab_size} onChange={(v) => updateModelField('vocab_size', v || 32000)} />
                  </div>
                </div>
                {modelConfig.model_type === 'moe' && modelConfig.moe_config && (
                  <>
                    <div className="my-2 border-t border-dashed border-gray-200 pt-2"><span className="text-xs text-gray-500">MoE 参数</span></div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className={`p-2 rounded -m-2 mb-0 ${isMoeFieldModified('num_experts') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="num_experts" label="专家数量" />
                          {isMoeFieldModified('num_experts') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={2} value={modelConfig.moe_config.num_experts} onChange={(v) => updateMoeField('num_experts', v || 8)} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMoeFieldModified('num_experts_per_tok') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="num_experts_per_tok" label="激活专家数" />
                          {isMoeFieldModified('num_experts_per_tok') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={1} value={modelConfig.moe_config.num_experts_per_tok} onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMoeFieldModified('num_shared_experts') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="num_shared_experts" label="共享专家数" />
                          {isMoeFieldModified('num_shared_experts') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={0} value={modelConfig.moe_config.num_shared_experts || 0} onChange={(v) => updateMoeField('num_shared_experts', v || 0)} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMoeFieldModified('expert_intermediate_size') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="expert_intermediate_size" label="专家FFN维度" />
                          {isMoeFieldModified('expert_intermediate_size') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={64} value={modelConfig.moe_config.expert_intermediate_size} onChange={(v) => updateMoeField('expert_intermediate_size', v)} placeholder="同FFN" />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMoeFieldModified('first_k_dense_replace') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="first_k_dense_replace" label="前K层Dense" />
                          {isMoeFieldModified('first_k_dense_replace') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={0} value={modelConfig.moe_config.first_k_dense_replace || 0} onChange={(v) => updateMoeField('first_k_dense_replace', v || 0)} />
                      </div>
                    </div>
                  </>
                )}
          </BaseCard>

          {/* 注意力配置 */}
          <BaseCard
            title="注意力配置"
            collapsible
            expanded={openSections.attention}
            onExpandChange={() => toggleSection('attention')}
            contentClassName="p-2"
            gradient
          >
                <div className="grid grid-cols-3 gap-2">
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('num_attention_heads') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="num_attention_heads" label="注意力头数" />
                      {isModelFieldModified('num_attention_heads') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={modelConfig.num_attention_heads} onChange={(v) => updateModelField('num_attention_heads', v || 32)} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('num_kv_heads') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="num_kv_heads" label="KV头数" />
                      {isModelFieldModified('num_kv_heads') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={modelConfig.num_kv_heads} onChange={(v) => updateModelField('num_kv_heads', v || 8)} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('attention_type') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="attention_type" label="Attention类型" />
                      {isModelFieldModified('attention_type') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <Select value={modelConfig.attention_type || 'mha'} onValueChange={(v) => {
                      if (v === 'mla' && !modelConfig.mla_config) onModelChange({ ...modelConfig, attention_type: v, mla_config: { kv_lora_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128, variant: 'mla' } })
                      else updateModelField('attention_type', v as any)
                    }}>
                      <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                      <SelectContent><SelectItem value="mha">MHA</SelectItem><SelectItem value="gqa">GQA</SelectItem><SelectItem value="mqa">MQA</SelectItem><SelectItem value="mla">MLA</SelectItem></SelectContent>
                    </Select>
                  </div>
                </div>
                {modelConfig.attention_type === 'mla' && modelConfig.mla_config && (
                  <>
                    <div className="my-2 border-t border-dashed border-gray-200 pt-2"><span className="text-xs text-gray-500">MLA 参数</span></div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('variant') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="mla_variant" label="MLA 变体" />
                          {isMlaFieldModified('variant') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <Select value={modelConfig.mla_config.variant || 'mla'} onValueChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, variant: v as any } })}>
                          <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                          <SelectContent><SelectItem value="mla">MLA</SelectItem><SelectItem value="mla_v32">MLA V3.2</SelectItem><SelectItem value="mla_absorb">MLA Absorb</SelectItem><SelectItem value="mla_absorb_v32">MLA Absorb V3.2</SelectItem></SelectContent>
                        </Select>
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('kv_lora_rank') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="kv_lora_rank" label="KV LoRA Rank" />
                          {isMlaFieldModified('kv_lora_rank') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={64} value={modelConfig.mla_config.kv_lora_rank} onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, kv_lora_rank: v || 512 } })} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('q_lora_rank') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="q_lora_rank" label="Q LoRA Rank" />
                          {isMlaFieldModified('q_lora_rank') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={64} value={modelConfig.mla_config.q_lora_rank} onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, q_lora_rank: v || 1536 } })} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('qk_nope_head_dim') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="qk_nope_head_dim" label="QK Nope维度" />
                          {isMlaFieldModified('qk_nope_head_dim') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={32} value={modelConfig.mla_config.qk_nope_head_dim} onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, qk_nope_head_dim: v || 128 } })} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('qk_rope_head_dim') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="qk_rope_head_dim" label="QK RoPE维度" />
                          {isMlaFieldModified('qk_rope_head_dim') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={32} value={modelConfig.mla_config.qk_rope_head_dim} onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, qk_rope_head_dim: v || 64 } })} />
                      </div>
                      <div className={`p-2 rounded -m-2 mb-0 ${isMlaFieldModified('v_head_dim') ? 'bg-blue-50/50' : ''}`}>
                        <div className="mb-1 flex items-center gap-1.5">
                          <ConfigLabel name="v_head_dim" label="V 头维度" />
                          {isMlaFieldModified('v_head_dim') && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                          )}
                        </div>
                        <NumberInput min={32} value={modelConfig.mla_config.v_head_dim} onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, v_head_dim: v || 128 } })} />
                      </div>
                    </div>
                  </>
                )}
          </BaseCard>

          {/* 精度配置 */}
          <BaseCard
            title="精度配置"
            collapsible
            expanded={openSections.precision}
            onExpandChange={() => toggleSection('precision')}
            contentClassName="p-2"
            gradient
          >
            <div className="grid grid-cols-2 gap-2">
              <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('weight_dtype') ? 'bg-blue-50/50' : ''}`}>
                <div className="mb-1 flex items-center gap-1.5">
                  <ConfigLabel name="weight_dtype" label="权重精度" />
                  {isModelFieldModified('weight_dtype') && (
                    <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                  )}
                </div>
                <Select value={modelConfig.weight_dtype} onValueChange={(v) => updateModelField('weight_dtype', v as any)}>
                  <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="fp32">FP32</SelectItem><SelectItem value="bf16">BF16</SelectItem><SelectItem value="fp16">FP16</SelectItem><SelectItem value="fp8">FP8</SelectItem><SelectItem value="int8">INT8</SelectItem><SelectItem value="int4">INT4</SelectItem></SelectContent>
                </Select>
              </div>
              <div className={`p-2 rounded -m-2 mb-0 ${isModelFieldModified('activation_dtype') ? 'bg-blue-50/50' : ''}`}>
                <div className="mb-1 flex items-center gap-1.5">
                  <ConfigLabel name="activation_dtype" label="激活精度" />
                  {isModelFieldModified('activation_dtype') && (
                    <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                  )}
                </div>
                <Select value={modelConfig.activation_dtype} onValueChange={(v) => updateModelField('activation_dtype', v as any)}>
                  <SelectTrigger className="w-full h-7"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="fp32">FP32</SelectItem><SelectItem value="bf16">BF16</SelectItem><SelectItem value="fp16">FP16</SelectItem><SelectItem value="fp8">FP8</SelectItem><SelectItem value="int8">INT8</SelectItem><SelectItem value="int4">INT4</SelectItem></SelectContent>
                </Select>
              </div>
            </div>
          </BaseCard>

          {/* 推理参数 */}
          <BaseCard
            title="推理参数"
            collapsible
            expanded={openSections.inference}
            onExpandChange={() => toggleSection('inference')}
            contentClassName="p-2"
            gradient
          >
                <div className="grid grid-cols-2 gap-2">
                  <div className={`p-2 rounded -m-2 mb-0 ${isInferenceFieldModified('batch_size') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="batch_size" label="Batch Size" />
                      {isInferenceFieldModified('batch_size') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={inferenceConfig.batch_size} onChange={(v) => onInferenceChange({ ...inferenceConfig, batch_size: v || 1 })} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isInferenceFieldModified('input_seq_length') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="input_seq_length" label="输入序列长度" />
                      {isInferenceFieldModified('input_seq_length') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={inferenceConfig.input_seq_length} onChange={(v) => onInferenceChange({ ...inferenceConfig, input_seq_length: v || 512 })} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isInferenceFieldModified('output_seq_length') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="output_seq_length" label="输出序列长度" />
                      {isInferenceFieldModified('output_seq_length') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={1} value={inferenceConfig.output_seq_length} onChange={(v) => onInferenceChange({ ...inferenceConfig, output_seq_length: v || 256 })} />
                  </div>
                  <div className={`p-2 rounded -m-2 mb-0 ${isInferenceFieldModified('max_seq_length') ? 'bg-blue-50/50' : ''}`}>
                    <div className="mb-1 flex items-center gap-1.5">
                      <ConfigLabel name="max_seq_length" label="最大序列长度" />
                      {isInferenceFieldModified('max_seq_length') && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-blue-100 text-blue-700 border-blue-300">已修改</Badge>
                      )}
                    </div>
                    <NumberInput min={inferenceConfig.input_seq_length + inferenceConfig.output_seq_length} value={inferenceConfig.max_seq_length} onChange={(v) => onInferenceChange({ ...inferenceConfig, max_seq_length: v || 768 })} />
                  </div>
                </div>
          </BaseCard>
        </div>

        {/* 估算参数量 */}
        <div className="p-2 bg-gray-50 rounded mb-3 text-[13px]">
          估算参数量: <b className="text-gray-800">{paramsStr}</b>
        </div>

        {/* 操作按钮 */}
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleSave}><Save className="h-3.5 w-3.5 mr-1" />保存</Button>
          <Button variant="outline" size="sm" onClick={handleSaveAs}><Copy className="h-3.5 w-3.5 mr-1" />另存为</Button>
          <Button variant="outline" size="sm" onClick={handleReset}><RefreshCw className="h-3.5 w-3.5 mr-1" />重置</Button>
        </div>
      </div>
  )
}

// ============================================
// 硬件配置选择器
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

  const totalChips = value.board.chips_per_board * value.pod.racks_per_pod * value.rack.boards_per_rack

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
          <span className="text-gray-500">显存: <b>{value.chip.memory_capacity_gb}GB</b>/卡</span>
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-gray-500">算力: {value.chip.compute_tflops_bf16} TFLOPs</span>
          <span className="text-gray-500">Board内: {value.board.b2b_bandwidth_gbps} GB/s</span>
        </div>
      </div>
    </div>
  )
}
