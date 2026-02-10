/**
 * BenchmarkEditor - Benchmark 配置编辑器
 *
 * 合并模型+推理参数的 Benchmark 配置面板，
 * 支持预设加载、字段编辑、修改追踪、保存/另存为/重置。
 *
 * 从 ConfigSelectors.tsx 的 BenchmarkConfigSelector 提取而来。
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { BaseCard } from '@/components/common/BaseCard'
import {
  LLMModelConfig,
  InferenceConfig,
} from '../../../utils/llmDeployment/types'
import { calculateModelParams } from '../../../api/model'
import { useDebouncedValue } from '@/hooks/useDebouncedCallback'
import {
  getModelList,
  getModelPreset,
} from '../../../utils/llmDeployment/presets'
import {
  getBenchmarks as apiBenchmarkList,
  getBenchmark as apiBenchmarkDetail,
  createBenchmark as apiCreateBenchmark,
} from '../../../api/math_model'
import { ConfigLabel, getDtypeBits, generateBenchmarkName } from './ConfigSelectors'

// ============================================
// Props
// ============================================

interface BenchmarkConfigSelectorProps {
  modelConfig: LLMModelConfig
  onModelChange: (config: LLMModelConfig) => void
  inferenceConfig: InferenceConfig
  onInferenceChange: (config: InferenceConfig) => void
  onBenchmarkSelect?: (benchmarkName: string | undefined) => void
}

// ============================================
// 常量
// ============================================

const LAST_BENCHMARK_KEY = 'llm_last_benchmark_id'

interface CustomBenchmark {
  id: string
  name: string
  model: LLMModelConfig
  inference: InferenceConfig
}

// ============================================
// 组件
// ============================================

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

  // 另存为弹窗状态
  const [saveAsDialogOpen, setSaveAsDialogOpen] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')

  // 原始配置快照（用于修改追踪）
  const [originalConfig, setOriginalConfig] = useState<{
    model: LLMModelConfig | null
    inference: InferenceConfig | null
  }>({ model: null, inference: null })

  const modelList = getModelList()

  // 使用防抖值减少 API 调用频率（300ms 延迟）
  const debouncedModelConfig = useDebouncedValue(modelConfig, 300)

  useEffect(() => {
    let cancelled = false
    calculateModelParams(debouncedModelConfig)
      .then((res) => { if (!cancelled) setParamsStr(res.formatted) })
      .catch(() => { if (!cancelled) setParamsStr('--') })
    return () => { cancelled = true }
  }, [debouncedModelConfig])

  useEffect(() => {
    apiBenchmarkList().then(async ({ benchmarks: summaries }) => {
      // 逐个获取完整配置
      const details = await Promise.all(
        summaries.map(s => apiBenchmarkDetail(s.id).catch(() => null))
      )
      const mapped = details
        .filter((b): b is NonNullable<typeof b> => b !== null)
        .map(b => ({
          id: b.id ?? b.name ?? '',
          name: b.name ?? '',
          model: b.model as unknown as LLMModelConfig,
          inference: b.inference as unknown as InferenceConfig,
        }))
      setCustomBenchmarks(mapped)
      if (mapped.length > 0) {
        const lastBenchmarkId = localStorage.getItem(LAST_BENCHMARK_KEY)
        const initialId = (lastBenchmarkId && mapped.find(b => b.id === lastBenchmarkId)) ? lastBenchmarkId : mapped[0].id
        setPresetId(initialId)
        onBenchmarkSelect?.(initialId)
        const initialBenchmark = mapped.find(b => b.id === initialId)
        if (initialBenchmark) {
          onModelChange(initialBenchmark.model)
          onInferenceChange(initialBenchmark.inference)
          setOriginalConfig({
            model: { ...initialBenchmark.model },
            inference: { ...initialBenchmark.inference },
          })
        }
      }
    }).catch(err => console.error('加载 Benchmark 列表失败:', err))
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
    onBenchmarkSelect?.(id)
    const match = customBenchmarks.find(c => c.id === id)
    if (match) {
      onModelChange(match.model)
      onInferenceChange(match.inference)
      setOriginalConfig({
        model: { ...match.model },
        inference: { ...match.inference },
      })
    }
  }

  const handleSave = async () => {
    if (isConfigModified()) {
      const newBenchmark: CustomBenchmark = { id: currentBenchmarkName, name: currentBenchmarkName, model: { ...modelConfig }, inference: { ...inferenceConfig } }
      try {
        await apiCreateBenchmark({ name: currentBenchmarkName, model: modelConfig as unknown as Record<string, unknown>, inference: inferenceConfig as unknown as Record<string, unknown> })
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
        onBenchmarkSelect?.(currentBenchmarkName)
        toast.success(`已保存: ${currentBenchmarkName}`)
      } catch {
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

  const handleOpenSaveAsDialog = () => {
    setSaveAsName(currentBenchmarkName)
    setSaveAsDialogOpen(true)
  }

  const handleConfirmSaveAs = async () => {
    if (!saveAsName.trim()) {
      toast.warning('请输入配置名称')
      return
    }
    const trimmedName = saveAsName.trim()
    const newBenchmark: CustomBenchmark = { id: trimmedName, name: trimmedName, model: { ...modelConfig }, inference: { ...inferenceConfig } }
    try {
      await apiCreateBenchmark({ name: trimmedName, model: modelConfig as unknown as Record<string, unknown>, inference: inferenceConfig as unknown as Record<string, unknown> })
      const existingIndex = customBenchmarks.findIndex(b => b.id === trimmedName)
      if (existingIndex >= 0) {
        const updated = [...customBenchmarks]
        updated[existingIndex] = newBenchmark
        setCustomBenchmarks(updated)
      } else {
        setCustomBenchmarks([...customBenchmarks, newBenchmark])
      }
      setPresetId(trimmedName)
      localStorage.setItem(LAST_BENCHMARK_KEY, trimmedName)
      onBenchmarkSelect?.(trimmedName)
      toast.success(`已保存: ${trimmedName}`)
      setSaveAsDialogOpen(false)
      setSaveAsName('')
    } catch {
      toast.error('保存失败')
    }
  }

  const dropdownOptions = customBenchmarks.map(c => ({ value: c.id, label: c.name }))
  const updateModelField = <K extends keyof LLMModelConfig>(field: K, val: LLMModelConfig[K]) => { onModelChange({ ...modelConfig, [field]: val }) }
  const updateMoeField = <K extends keyof NonNullable<LLMModelConfig['moe_config']>>(field: K, val: NonNullable<LLMModelConfig['moe_config']>[K]) => {
    if (modelConfig.moe_config) onModelChange({ ...modelConfig, moe_config: { ...modelConfig.moe_config, [field]: val } })
  }

  const toggleSection = (key: string) => { setOpenSections(prev => ({ ...prev, [key]: !prev[key] })) }

  const isModelFieldModified = (fieldName: keyof LLMModelConfig): boolean => {
    if (!originalConfig.model) return false
    return modelConfig[fieldName] !== originalConfig.model[fieldName]
  }

  const isInferenceFieldModified = (fieldName: keyof InferenceConfig): boolean => {
    if (!originalConfig.inference) return false
    return inferenceConfig[fieldName] !== originalConfig.inference[fieldName]
  }

  const isMoeFieldModified = (fieldName: string): boolean => {
    if (!originalConfig.model || !originalConfig.model.moe_config || !modelConfig.moe_config) return false
    return (modelConfig.moe_config as any)[fieldName] !== (originalConfig.model.moe_config as any)[fieldName]
  }

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
          <Button variant="outline" size="sm" onClick={handleOpenSaveAsDialog}><Copy className="h-3.5 w-3.5 mr-1" />另存为</Button>
          <Button variant="outline" size="sm" onClick={handleReset}><RefreshCw className="h-3.5 w-3.5 mr-1" />重置</Button>
        </div>

        {/* 另存为弹窗 */}
        <Dialog open={saveAsDialogOpen} onOpenChange={setSaveAsDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>另存为新配置</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <label className="text-sm font-medium mb-2 block">配置名称</label>
              <Input
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                placeholder="请输入配置名称"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && saveAsName.trim()) {
                    handleConfirmSaveAs()
                  }
                }}
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => {
                setSaveAsDialogOpen(false)
                setSaveAsName('')
              }}>
                取消
              </Button>
              <Button onClick={handleConfirmSaveAs} disabled={!saveAsName.trim()}>
                保存
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
  )
}
