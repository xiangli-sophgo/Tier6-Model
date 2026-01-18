/**
 * 配置选择器组件
 *
 * 包含：ModelConfigSelector, InferenceConfigSelector, HardwareConfigSelector
 */

import React, { useState, useCallback } from 'react'
import {
  Typography,
  Button,
  InputNumber,
  Select,
  Tag,
  Tooltip,
} from 'antd'
import {
  LLMModelConfig,
  InferenceConfig,
  HardwareConfig,
} from '../../../utils/llmDeployment/types'
import { calculateModelParams } from '../../../utils/llmDeployment/modelCalculator'
import {
  getModelList,
  getChipList,
  getModelPreset,
  getInferencePreset,
  createHardwareConfig,
} from '../../../utils/llmDeployment/presets'

const { Text } = Typography

// ============================================
// 样式常量
// ============================================

export const colors = {
  primary: '#5E6AD2',
  primaryLight: '#E8EAFC',
  success: '#52c41a',
  successLight: '#f6ffed',
  warning: '#faad14',
  warningLight: '#fffbe6',
  error: '#ff4d4f',
  errorLight: '#fff2f0',
  border: '#E5E5E5',
  borderLight: '#F0F0F0',
  background: '#FAFAFA',
  backgroundDark: '#F5F5F5',
  text: '#1A1A1A',
  textSecondary: '#666666',
}

export const configRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 10,
}

export const sectionCardStyle: React.CSSProperties = {
  background: '#fff',
  borderRadius: 10,
  padding: 16,
  marginBottom: 12,
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.06)',
  border: `1px solid ${colors.borderLight}`,
}

export const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: colors.text,
  marginBottom: 12,
  display: 'flex',
  alignItems: 'center',
  gap: 6,
}

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
  const modelList = getModelList()

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

  const paramRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  }

  const isCustomModel = presetId.startsWith('custom_')

  const estimateParams = () => {
    const total = calculateModelParams(value)
    const billions = total / 1e9
    const result = billions >= 1 ? `${billions.toFixed(1)}B` : `${(total / 1e6).toFixed(0)}M`
    return { value: result, breakdown: `总参数量: ${result}` }
  }

  return (
    <div>
      <div style={configRowStyle}>
        <Text>模型选择</Text>
        <Select
          size="small"
          value={presetId}
          onChange={handlePresetChange}
          style={{ width: '100%', maxWidth: 280 }}
          options={allModelOptions}
          optionRender={(option) => (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{option.label}</span>
              {option.data.isCustom && (
                <Button
                  type="text"
                  size="small"
                  danger
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteCustomModel(option.value as string)
                  }}
                  style={{ padding: '0 4px', height: 20, fontSize: 11 }}
                >
                  删除
                </Button>
              )}
            </div>
          )}
        />
      </div>

      {editMode ? (
        <div style={{ padding: 8, background: '#f5f5f5', borderRadius: 6, fontSize: 12 }}>
          <div style={paramRowStyle}>
            <Tooltip title="自定义模型显示名称"><Text style={{ fontSize: 11, cursor: 'help' }}>模型名称</Text></Tooltip>
            <input
              value={value.model_name}
              onChange={(e) => updateField('model_name', e.target.value)}
              style={{ width: 180, fontSize: 11, padding: '2px 6px', border: '1px solid #d9d9d9', borderRadius: 4 }}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Dense: 标准密集模型; MoE: 混合专家稀疏模型"><Text style={{ fontSize: 11, cursor: 'help' }}>模型类型</Text></Tooltip>
            <Select
              size="small"
              value={value.model_type}
              onChange={(v) => {
                if (v === 'moe' && !value.moe_config) {
                  onChange({ ...value, model_type: v, moe_config: { num_experts: 8, num_experts_per_tok: 2, expert_capacity_factor: 1.25 } })
                } else {
                  updateField('model_type', v)
                }
              }}
              style={{ width: 80 }}
              options={[
                { value: 'dense', label: 'Dense' },
                { value: 'moe', label: 'MoE' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Hidden Size: 每个token的向量表示维度"><Text style={{ fontSize: 11, cursor: 'help' }}>隐藏层维度</Text></Tooltip>
            <InputNumber size="small" min={64} max={65536} value={value.hidden_size}
              onChange={(v) => updateField('hidden_size', v || 4096)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Layers: Transformer层数"><Text style={{ fontSize: 11, cursor: 'help' }}>层数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_layers}
              onChange={(v) => updateField('num_layers', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Attention Heads: 多头注意力的Query头数"><Text style={{ fontSize: 11, cursor: 'help' }}>注意力头数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_attention_heads}
              onChange={(v) => updateField('num_attention_heads', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num KV Heads: Key-Value头数，GQA时小于注意力头数"><Text style={{ fontSize: 11, cursor: 'help' }}>KV头数</Text></Tooltip>
            <InputNumber size="small" min={1} max={256} value={value.num_kv_heads}
              onChange={(v) => updateField('num_kv_heads', v || 8)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Intermediate Size: FFN层中间维度，通常是hidden_size的2.5-4倍"><Text style={{ fontSize: 11, cursor: 'help' }}>FFN维度</Text></Tooltip>
            <InputNumber size="small" min={64} max={131072} value={value.intermediate_size}
              onChange={(v) => updateField('intermediate_size', v || 11008)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Vocab Size: 词表大小，影响Embedding层参数量"><Text style={{ fontSize: 11, cursor: 'help' }}>词表大小</Text></Tooltip>
            <InputNumber size="small" min={1000} max={500000} value={value.vocab_size}
              onChange={(v) => updateField('vocab_size', v || 32000)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="权重精度: 模型权重的存储精度，影响权重显存占用"><Text style={{ fontSize: 11, cursor: 'help' }}>权重精度</Text></Tooltip>
            <Select size="small" value={value.weight_dtype} onChange={(v) => updateField('weight_dtype', v)} style={{ width: 70 }}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="激活精度: 计算过程中的激活值和KV Cache精度"><Text style={{ fontSize: 11, cursor: 'help' }}>激活精度</Text></Tooltip>
            <Select size="small" value={value.activation_dtype} onChange={(v) => updateField('activation_dtype', v)} style={{ width: 70 }}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'int8', label: 'INT8' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Max Sequence Length: 模型支持的最大上下文长度"><Text style={{ fontSize: 11, cursor: 'help' }}>最大序列长度</Text></Tooltip>
            <InputNumber size="small" min={128} max={1048576} value={value.max_seq_length}
              onChange={(v) => updateField('max_seq_length', v || 4096)} style={{ width: 80 }} />
          </div>

          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
              <Tag color="purple" style={{ marginBottom: 6 }}>MoE 参数</Tag>
              <div style={paramRowStyle}>
                <Tooltip title="Num Experts: FFN层的专家总数"><Text style={{ fontSize: 11, cursor: 'help' }}>专家数量</Text></Tooltip>
                <InputNumber size="small" min={2} max={1024} value={value.moe_config.num_experts}
                  onChange={(v) => updateMoeField('num_experts', v || 8)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Top-K: 每个token激活的专家数量"><Text style={{ fontSize: 11, cursor: 'help' }}>激活专家数</Text></Tooltip>
                <InputNumber size="small" min={1} max={64} value={value.moe_config.num_experts_per_tok}
                  onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Shared Experts: 所有token共享的专家数量（DeepSeek特有）"><Text style={{ fontSize: 11, cursor: 'help' }}>共享专家数</Text></Tooltip>
                <InputNumber size="small" min={0} max={16} value={value.moe_config.num_shared_experts || 0}
                  onChange={(v) => updateMoeField('num_shared_experts', v || 0)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Expert FFN Size: 每个专家的FFN中间维度（不设置则使用上方的FFN维度）"><Text style={{ fontSize: 11, cursor: 'help' }}>专家FFN维度</Text></Tooltip>
                <InputNumber size="small" min={64} max={65536} value={value.moe_config.expert_intermediate_size}
                  onChange={(v) => updateMoeField('expert_intermediate_size', v || undefined)} style={{ width: 80 }}
                  placeholder="同FFN" />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="First K Dense: 前K层使用Dense FFN而非MoE（DeepSeek V3=3）"><Text style={{ fontSize: 11, cursor: 'help' }}>前K层Dense</Text></Tooltip>
                <InputNumber size="small" min={0} max={100} value={value.moe_config.first_k_dense_replace || 0}
                  onChange={(v) => updateMoeField('first_k_dense_replace', v || 0)} style={{ width: 80 }} />
              </div>
            </div>
          )}

          {value.attention_type === 'mla' && value.mla_config && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
              <Tag color="cyan" style={{ marginBottom: 6 }}>MLA 并行度</Tag>
              <div style={paramRowStyle}>
                <Tooltip title="MLA TP: MLA (Attention) 部分的张量并行度，不设置则使用全局 TP"><Text style={{ fontSize: 11, cursor: 'help' }}>MLA TP</Text></Tooltip>
                <InputNumber size="small" min={1} max={64} value={value.mla_config.mla_tp || undefined}
                  onChange={(v) => updateMlaField('mla_tp', v || undefined)} style={{ width: 80 }}
                  placeholder="同TP" />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="MLA DP: MLA (Attention) 部分的数据并行度，不设置则使用全局 DP"><Text style={{ fontSize: 11, cursor: 'help' }}>MLA DP</Text></Tooltip>
                <InputNumber size="small" min={1} max={64} value={value.mla_config.mla_dp || undefined}
                  onChange={(v) => updateMlaField('mla_dp', v || undefined)} style={{ width: 80 }}
                  placeholder="同DP" />
              </div>
              <Text type="secondary" style={{ fontSize: 10 }}>约束: MLA_TP × MLA_DP = TP × DP</Text>
            </div>
          )}
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
              <Text type="secondary" style={{ cursor: 'help' }}>估算参数量: <b>{estimateParams().value}</b></Text>
            </Tooltip>
          </div>
        </div>
      ) : (
        <div style={{ padding: 8, background: '#f5f5f5', borderRadius: 6, fontSize: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <Tooltip title="Hidden Size: 每个token的向量表示维度，决定模型的表示能力">
              <Text type="secondary" style={{ cursor: 'help' }}>隐藏层: {value.hidden_size}</Text>
            </Tooltip>
            <Tooltip title="Num Layers: Transformer层数，层数越多模型越深">
              <Text type="secondary" style={{ cursor: 'help' }}>层数: {value.num_layers}</Text>
            </Tooltip>
            <Tooltip title="Num Attention Heads: 多头注意力的头数，用于并行计算不同的注意力模式">
              <Text type="secondary" style={{ cursor: 'help' }}>注意力头: {value.num_attention_heads}</Text>
            </Tooltip>
            <Tooltip title="Num KV Heads: Key-Value头数，GQA/MQA时小于注意力头数可减少KV Cache">
              <Text type="secondary" style={{ cursor: 'help' }}>KV头: {value.num_kv_heads}</Text>
            </Tooltip>
            <Tooltip title="Intermediate Size: FFN层的中间维度，通常是隐藏层的2.5-4倍">
              <Text type="secondary" style={{ cursor: 'help' }}>FFN: {value.intermediate_size}</Text>
            </Tooltip>
            <Tooltip title="精度: W=权重精度, A=激活精度，如W4A16表示权重INT4、激活FP16">
              <Text type="secondary" style={{ cursor: 'help' }}>精度: W{value.weight_dtype === 'int4' ? '4' : value.weight_dtype === 'int8' ? '8' : '16'}A{value.activation_dtype === 'int8' ? '8' : '16'}</Text>
            </Tooltip>
          </div>
          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #e8e8e8' }}>
              <Tooltip title="Mixture of Experts: 稀疏激活架构，每次只激活部分专家，提高模型容量的同时控制计算量">
                <Tag color="purple" style={{ cursor: 'help' }}>MoE</Tag>
              </Tooltip>
              <Tooltip title={`总共${value.moe_config.num_experts}个专家，每个token激活${value.moe_config.num_experts_per_tok}个`}>
                <Text type="secondary" style={{ fontSize: 11, cursor: 'help' }}>
                  {value.moe_config.num_experts}专家 × {value.moe_config.num_experts_per_tok}激活
                  {value.moe_config.num_shared_experts ? ` + ${value.moe_config.num_shared_experts}共享` : ''}
                </Text>
              </Tooltip>
            </div>
          )}
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #e8e8e8', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
              <Text type="secondary" style={{ cursor: 'help' }}>估算参数量: <b>{estimateParams().value}</b></Text>
            </Tooltip>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
        {editMode && (
          <Button
            size="small"
            onClick={() => {
              if (editBackup) {
                onChange(editBackup)
              }
              setEditMode(false)
              setEditBackup(null)
            }}
            style={{ flex: 1 }}
          >
            取消
          </Button>
        )}
        <Button
          size="small"
          type={editMode ? 'primary' : 'default'}
          onClick={() => {
            if (!editMode) {
              setEditBackup({ ...value })
            }
            setEditMode(!editMode)
          }}
          style={{ flex: 1 }}
        >
          {editMode ? '完成编辑' : '编辑模型参数'}
        </Button>
        {isCustomModel && (
          <Button
            size="small"
            type="primary"
            onClick={() => {
              const newModels = { ...customModels, [presetId]: { ...value } }
              setCustomModels(newModels)
              saveCustomModels(newModels)
            }}
            style={{ flex: 1 }}
          >
            保存
          </Button>
        )}
        {editMode && (
          <Button
            size="small"
            onClick={() => {
              setSaveName(value.model_name)
              setSaveModalVisible(true)
            }}
            style={{ flex: 1 }}
          >
            另存为
          </Button>
        )}
        {isCustomModel && (
          <Button
            size="small"
            danger
            onClick={() => handleDeleteCustomModel(presetId)}
          >
            删除
          </Button>
        )}
      </div>

      {saveModalVisible && (
        <div style={{
          padding: 12,
          background: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: 6,
          marginTop: 8,
        }}>
          <div style={{ marginBottom: 8 }}>
            <Text style={{ fontSize: 12 }}>输入自定义模型名称：</Text>
          </div>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="如: My-Custom-Model"
            style={{
              width: '100%',
              padding: '6px 8px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              marginBottom: 8,
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSaveModel()}
          />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <Button size="small" onClick={() => setSaveModalVisible(false)}>取消</Button>
            <Button size="small" type="primary" onClick={handleSaveModel}>保存</Button>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// 推理配置选择器
// ============================================

interface InferenceConfigSelectorProps {
  value: InferenceConfig
  onChange: (config: InferenceConfig) => void
}

const CUSTOM_INFERENCE_KEY = 'llm_custom_inference'

function loadCustomInference(): Record<string, InferenceConfig & { name: string }> {
  try {
    const data = localStorage.getItem(CUSTOM_INFERENCE_KEY)
    return data ? JSON.parse(data) : {}
  } catch {
    return {}
  }
}

function saveCustomInference(configs: Record<string, InferenceConfig & { name: string }>) {
  localStorage.setItem(CUSTOM_INFERENCE_KEY, JSON.stringify(configs))
}

export const InferenceConfigSelector: React.FC<InferenceConfigSelectorProps> = ({ value, onChange }) => {
  const [presetId, setPresetId] = useState<string>('standard')
  const [customConfigs, setCustomConfigs] = useState(loadCustomInference)
  const [saveModalVisible, setSaveModalVisible] = useState(false)
  const [saveName, setSaveName] = useState('')

  const presetOptions = [
    { value: 'low-latency', label: '低延迟交互', isCustom: false },
    { value: 'standard', label: '标准推理', isCustom: false },
    { value: 'high-throughput', label: '高吞吐批处理', isCustom: false },
    { value: 'long-context', label: '长上下文', isCustom: false },
    { value: 'code-gen', label: '代码生成', isCustom: false },
  ]

  const allOptions = [
    ...Object.entries(customConfigs).map(([id, cfg]) => ({
      value: id,
      label: `[自定义] ${cfg.name}`,
      isCustom: true,
    })),
    ...presetOptions,
  ]

  const handlePresetChange = (id: string) => {
    setPresetId(id)
    if (customConfigs[id]) {
      const { name: _, ...config } = customConfigs[id]
      onChange(config)
    } else {
      onChange(getInferencePreset(id))
    }
  }

  const handleSave = () => {
    if (!saveName.trim()) return
    const customId = `custom_inf_${saveName.trim().toLowerCase().replace(/\s+/g, '_')}`
    const newConfigs = { ...customConfigs, [customId]: { ...value, name: saveName.trim() } }
    setCustomConfigs(newConfigs)
    saveCustomInference(newConfigs)
    setPresetId(customId)
    setSaveModalVisible(false)
    setSaveName('')
  }

  const handleDelete = (id: string) => {
    const newConfigs = { ...customConfigs }
    delete newConfigs[id]
    setCustomConfigs(newConfigs)
    saveCustomInference(newConfigs)
    if (presetId === id) {
      setPresetId('standard')
      onChange(getInferencePreset('standard'))
    }
  }

  const isCustom = presetId.startsWith('custom_inf_')

  return (
    <div>
      <div style={configRowStyle}>
        <Text>场景预设</Text>
        <Select
          size="small"
          value={presetId}
          onChange={handlePresetChange}
          style={{ width: 160 }}
          options={allOptions}
          optionRender={(option) => (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{option.label}</span>
              {option.data.isCustom && (
                <Button
                  type="text"
                  size="small"
                  danger
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(option.value as string)
                  }}
                  style={{ padding: '0 4px', height: 20, fontSize: 11 }}
                >
                  删除
                </Button>
              )}
            </div>
          )}
        />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
        <div style={configRowStyle}>
          <Tooltip title="同时处理的请求数量，影响吞吐量和延迟">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Batch Size</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={128}
            value={value.batch_size}
            onChange={(v) => onChange({ ...value, batch_size: v || 1 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="输入提示词的 Token 数量（Prefill 阶段处理）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Input SeqLen</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={131072}
            value={value.input_seq_length}
            onChange={(v) => onChange({ ...value, input_seq_length: v || 512 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="生成输出的 Token 数量（Decode 阶段逐个生成）">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Output SeqLen</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={1}
            max={32768}
            value={value.output_seq_length}
            onChange={(v) => onChange({ ...value, output_seq_length: v || 256 })}
            style={{ width: 70 }}
          />
        </div>
        <div style={configRowStyle}>
          <Tooltip title="KV Cache 预分配的最大长度，通常 ≥ 输入长度 + 输出长度">
            <Text style={{ fontSize: 12, cursor: 'help' }}>Max SeqLen</Text>
          </Tooltip>
          <InputNumber
            size="small"
            min={value.input_seq_length + value.output_seq_length}
            max={131072}
            value={value.max_seq_length}
            onChange={(v) => onChange({ ...value, max_seq_length: v || 768 })}
            style={{ width: 70 }}
          />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
        {isCustom && (
          <Button
            size="small"
            type="primary"
            onClick={() => {
              const newConfigs = { ...customConfigs, [presetId]: { ...value, name: customConfigs[presetId].name } }
              setCustomConfigs(newConfigs)
              saveCustomInference(newConfigs)
            }}
            style={{ flex: 1 }}
          >
            保存
          </Button>
        )}
        <Button
          size="small"
          onClick={() => {
            setSaveName('')
            setSaveModalVisible(true)
          }}
          style={{ flex: 1 }}
        >
          另存为
        </Button>
      </div>

      {saveModalVisible && (
        <div style={{
          padding: 12,
          background: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: 6,
          marginTop: 8,
        }}>
          <div style={{ marginBottom: 8 }}>
            <Text style={{ fontSize: 12 }}>输入配置名称：</Text>
          </div>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="如: RAG场景"
            style={{
              width: '100%',
              padding: '6px 8px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              marginBottom: 8,
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <Button size="small" onClick={() => setSaveModalVisible(false)}>取消</Button>
            <Button size="small" type="primary" onClick={handleSave}>保存</Button>
          </div>
        </div>
      )}
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
  const [nodeId, setNodeId] = useState<string>('dgx-h100')
  const [numNodes, setNumNodes] = useState<number>(1)

  const chipList = getChipList()
  const nodeOptions = [
    { value: 'dgx-h100', label: 'DGX H100 (8卡 NVLink)' },
    { value: 'dgx-a100', label: 'DGX A100 (8卡 NVLink)' },
    { value: 'pcie-8gpu', label: '通用 PCIe (8卡)' },
  ]

  const handleConfigChange = useCallback((newChipId: string, newNodeId: string, newNumNodes: number) => {
    const config = createHardwareConfig(newChipId, newNodeId, newNumNodes, 400)
    onChange(config)
  }, [onChange])

  const handleChipChange = (id: string) => {
    setChipId(id)
    handleConfigChange(id, nodeId, numNodes)
  }

  const handleNodeChange = (id: string) => {
    setNodeId(id)
    handleConfigChange(chipId, id, numNodes)
  }

  const handleNumNodesChange = (n: number) => {
    setNumNodes(n)
    handleConfigChange(chipId, nodeId, n)
  }

  const totalChips = value.node.chips_per_node * value.cluster.num_nodes

  return (
    <div>
      <div style={configRowStyle}>
        <Text>芯片类型</Text>
        <Select
          size="small"
          value={chipId}
          onChange={handleChipChange}
          style={{ width: 140 }}
          options={chipList.map(c => ({
            value: c.id,
            label: `${c.name}`,
          }))}
        />
      </div>
      <div style={configRowStyle}>
        <Text>节点类型</Text>
        <Select
          size="small"
          value={nodeId}
          onChange={handleNodeChange}
          style={{ width: 160 }}
          options={nodeOptions}
        />
      </div>
      <div style={configRowStyle}>
        <Text>节点数量</Text>
        <InputNumber
          size="small"
          min={1}
          max={64}
          value={numNodes}
          onChange={(v) => handleNumNodesChange(v || 1)}
          style={{ width: 70 }}
        />
      </div>
      <div style={{ padding: 8, background: '#f0f5ff', borderRadius: 6, fontSize: 12, marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Text type="secondary">总芯片数: <b>{totalChips}</b></Text>
          <Text type="secondary">显存: <b>{value.chip.memory_gb}GB</b>/卡</Text>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <Text type="secondary">算力: {value.chip.compute_tflops_fp16} TFLOPs</Text>
          <Text type="secondary">节点内: {value.node.intra_node_bandwidth_gbps} GB/s</Text>
        </div>
      </div>
    </div>
  )
}
