/**
 * 配置选择器组件
 *
 * 包含：ModelConfigSelector, BenchmarkConfigSelector, HardwareConfigSelector
 */

import React, { useState, useCallback, useEffect } from 'react'
import {
  Typography,
  Button,
  InputNumber,
  Select,
  Tag,
  Tooltip,
  Space,
  message,
  Collapse,
  Row,
  Col,
  Divider,
} from 'antd'
import { SaveOutlined, CopyOutlined, ReloadOutlined, DownOutlined, UpOutlined } from '@ant-design/icons'
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

const { Text } = Typography

// ============================================
// 样式常量
// ============================================

export const colors = {
  // 主色（紫蓝色 - 保留用于大卡片标题等）
  primary: '#5E6AD2',
  primaryLight: '#E8EAFC',

  // 交互色（Ant Design标准蓝 - 统一用于可点击卡片）
  interactive: '#1890ff',
  interactiveLight: '#e6f7ff',
  interactiveShadow: 'rgba(24, 144, 255, 0.15)',

  // 状态色
  success: '#52c41a',
  successLight: '#f6ffed',
  warning: '#faad14',
  warningLight: '#fffbe6',
  error: '#ff4d4f',
  errorLight: '#fff2f0',

  // 中性色
  border: '#e8e8e8',
  borderLight: '#F0F0F0',
  background: '#FAFAFA',
  backgroundDark: '#F5F5F5',
  cardBg: '#fff',
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
// Benchmark 参数提示
// ============================================

const BENCHMARK_TOOLTIPS: Record<string, string> = {
  // 基础参数
  model_name: '选择预设模型',
  model_type: 'Dense: 标准密集模型; MoE: 混合专家稀疏模型',
  hidden_size: 'Hidden Size: 隐藏层维度',
  num_layers: 'Num Layers: Transformer 层数',
  intermediate_size: 'Intermediate Size: FFN 中间层维度',
  vocab_size: 'Vocab Size: 词表大小',
  // 注意力配置
  num_attention_heads: 'Num Attention Heads: 注意力头数',
  num_kv_heads: 'Num KV Heads: KV 头数 (GQA)',
  attention_type: 'Attention 类型: MHA/GQA/MQA/MLA',
  // 精度配置
  weight_dtype: '权重存储精度',
  activation_dtype: '激活/KV Cache 精度',
  // MLA 参数
  mla_variant: 'MLA 实现变体',
  kv_lora_rank: 'KV LoRA Rank: KV 压缩后的隐维度',
  q_lora_rank: 'Q LoRA Rank: Query 的 LoRA rank',
  qk_nope_head_dim: 'QK Nope Head Dim: 非 RoPE 头维度',
  qk_rope_head_dim: 'QK RoPE Head Dim: RoPE 头维度',
  v_head_dim: 'V Head Dim: Value 头维度',
  // MoE 参数
  num_experts: 'Num Experts: 专家总数',
  num_experts_per_tok: 'Top-K: 每个 token 激活的专家数',
  num_shared_experts: 'Shared Experts: 共享专家数量',
  expert_intermediate_size: 'Expert FFN Size: 专家 FFN 维度',
  first_k_dense_replace: 'First K Dense: 前 K 层使用 Dense 替代 MoE',
  // 推理参数
  batch_size: 'Batch Size: 批处理大小',
  input_seq_length: '输入 Token 数量',
  output_seq_length: '输出 Token 数量',
  max_seq_length: 'KV Cache 最大长度',
}

// ConfigLabel 组件
interface ConfigLabelProps {
  name: string
  label?: string
}

const ConfigLabel: React.FC<ConfigLabelProps> = ({ name, label }) => (
  <Tooltip title={BENCHMARK_TOOLTIPS[name] || name}>
    <Text type="secondary" style={{ cursor: 'help', fontSize: 12 }}>{label || name}</Text>
  </Tooltip>
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

  // 异步获取模型参数量
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

  const paramRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  }

  const isCustomModel = presetId.startsWith('custom_')

  // 使用状态中的参数量字符串
  const estimateParams = () => {
    return { value: paramsStr, breakdown: `总参数量: ${paramsStr}` }
  }

  return (
    <div>
      <div style={configRowStyle}>
        <Text>模型选择</Text>
        <Select
          size="middle"
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
                  size="middle"
                  danger
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteCustomModel(option.value as string)
                  }}
                  style={{ padding: '0 4px', height: 20, fontSize: 13 }}
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
            <Tooltip title="自定义模型显示名称"><Text style={{ fontSize: 13, cursor: 'help' }}>模型名称</Text></Tooltip>
            <input
              value={value.model_name}
              onChange={(e) => updateField('model_name', e.target.value)}
              style={{ width: 180, fontSize: 13, padding: '2px 6px', border: '1px solid #d9d9d9', borderRadius: 4 }}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Dense: 标准密集模型; MoE: 混合专家稀疏模型"><Text style={{ fontSize: 13, cursor: 'help' }}>模型类型</Text></Tooltip>
            <Select
              size="middle"
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
            <Tooltip title="Hidden Size: 每个token的向量表示维度"><Text style={{ fontSize: 13, cursor: 'help' }}>隐藏层维度</Text></Tooltip>
            <InputNumber size="middle" min={64} max={65536} value={value.hidden_size}
              onChange={(v) => updateField('hidden_size', v || 4096)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Layers: Transformer层数"><Text style={{ fontSize: 13, cursor: 'help' }}>层数</Text></Tooltip>
            <InputNumber size="middle" min={1} max={256} value={value.num_layers}
              onChange={(v) => updateField('num_layers', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num Attention Heads: 多头注意力的Query头数"><Text style={{ fontSize: 13, cursor: 'help' }}>注意力头数</Text></Tooltip>
            <InputNumber size="middle" min={1} max={256} value={value.num_attention_heads}
              onChange={(v) => updateField('num_attention_heads', v || 32)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Num KV Heads: Key-Value头数，GQA时小于注意力头数"><Text style={{ fontSize: 13, cursor: 'help' }}>KV头数</Text></Tooltip>
            <InputNumber size="middle" min={1} max={256} value={value.num_kv_heads}
              onChange={(v) => updateField('num_kv_heads', v || 8)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Intermediate Size: FFN层中间维度，通常是hidden_size的2.5-4倍"><Text style={{ fontSize: 13, cursor: 'help' }}>FFN维度</Text></Tooltip>
            <InputNumber size="middle" min={64} max={131072} value={value.intermediate_size}
              onChange={(v) => updateField('intermediate_size', v || 11008)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Vocab Size: 词表大小，影响Embedding层参数量"><Text style={{ fontSize: 13, cursor: 'help' }}>词表大小</Text></Tooltip>
            <InputNumber size="middle" min={1000} max={500000} value={value.vocab_size}
              onChange={(v) => updateField('vocab_size', v || 32000)} style={{ width: 80 }} />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="权重精度: 模型权重的存储精度，影响权重显存占用"><Text style={{ fontSize: 13, cursor: 'help' }}>权重精度</Text></Tooltip>
            <Select size="middle" value={value.weight_dtype} onChange={(v) => updateField('weight_dtype', v)} style={{ width: 90 }}
              options={[
                { value: 'fp32', label: 'FP32' },
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp8', label: 'FP8' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="激活精度: 计算过程中的激活值和KV Cache精度"><Text style={{ fontSize: 13, cursor: 'help' }}>激活精度</Text></Tooltip>
            <Select size="middle" value={value.activation_dtype} onChange={(v) => updateField('activation_dtype', v)} style={{ width: 90 }}
              options={[
                { value: 'fp32', label: 'FP32' },
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp8', label: 'FP8' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </div>
          <div style={paramRowStyle}>
            <Tooltip title="Max Sequence Length: 模型支持的最大上下文长度"><Text style={{ fontSize: 13, cursor: 'help' }}>最大序列长度</Text></Tooltip>
            <InputNumber size="middle" min={128} max={1048576} value={value.max_seq_length}
              onChange={(v) => updateField('max_seq_length', v || 4096)} style={{ width: 80 }} />
          </div>

          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
              <Tag color="purple" style={{ marginBottom: 6 }}>MoE 参数</Tag>
              <div style={paramRowStyle}>
                <Tooltip title="Num Experts: FFN层的专家总数"><Text style={{ fontSize: 13, cursor: 'help' }}>专家数量</Text></Tooltip>
                <InputNumber size="middle" min={2} max={1024} value={value.moe_config.num_experts}
                  onChange={(v) => updateMoeField('num_experts', v || 8)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Top-K: 每个token激活的专家数量"><Text style={{ fontSize: 13, cursor: 'help' }}>激活专家数</Text></Tooltip>
                <InputNumber size="middle" min={1} max={64} value={value.moe_config.num_experts_per_tok}
                  onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Shared Experts: 所有token共享的专家数量（DeepSeek特有）"><Text style={{ fontSize: 13, cursor: 'help' }}>共享专家数</Text></Tooltip>
                <InputNumber size="middle" min={0} max={16} value={value.moe_config.num_shared_experts || 0}
                  onChange={(v) => updateMoeField('num_shared_experts', v || 0)} style={{ width: 80 }} />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="Expert FFN Size: 每个专家的FFN中间维度（不设置则使用上方的FFN维度）"><Text style={{ fontSize: 13, cursor: 'help' }}>专家FFN维度</Text></Tooltip>
                <InputNumber size="middle" min={64} max={65536} value={value.moe_config.expert_intermediate_size}
                  onChange={(v) => updateMoeField('expert_intermediate_size', v || undefined)} style={{ width: 80 }}
                  placeholder="同FFN" />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="First K Dense: 前K层使用Dense FFN而非MoE（DeepSeek V3=3）"><Text style={{ fontSize: 13, cursor: 'help' }}>前K层Dense</Text></Tooltip>
                <InputNumber size="middle" min={0} max={100} value={value.moe_config.first_k_dense_replace || 0}
                  onChange={(v) => updateMoeField('first_k_dense_replace', v || 0)} style={{ width: 80 }} />
              </div>
            </div>
          )}

          {value.attention_type === 'mla' && value.mla_config && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
              <Tag color="cyan" style={{ marginBottom: 6 }}>MLA 并行度</Tag>
              <div style={paramRowStyle}>
                <Tooltip title="MLA TP: MLA (Attention) 部分的张量并行度，不设置则使用全局 TP"><Text style={{ fontSize: 13, cursor: 'help' }}>MLA TP</Text></Tooltip>
                <InputNumber size="middle" min={1} max={64} value={value.mla_config.mla_tp || undefined}
                  onChange={(v) => updateMlaField('mla_tp', v || undefined)} style={{ width: 80 }}
                  placeholder="同TP" />
              </div>
              <div style={paramRowStyle}>
                <Tooltip title="MLA DP: MLA (Attention) 部分的数据并行度，不设置则使用全局 DP"><Text style={{ fontSize: 13, cursor: 'help' }}>MLA DP</Text></Tooltip>
                <InputNumber size="middle" min={1} max={64} value={value.mla_config.mla_dp || undefined}
                  onChange={(v) => updateMlaField('mla_dp', v || undefined)} style={{ width: 80 }}
                  placeholder="同DP" />
              </div>
              <Text type="secondary" style={{ fontSize: 10 }}>约束: MLA_TP × MLA_DP = TP × DP</Text>
            </div>
          )}
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #e8e8e8' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 13, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
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
              <Text type="secondary" style={{ cursor: 'help' }}>精度: W{getDtypeBits(value.weight_dtype)}A{getDtypeBits(value.activation_dtype)}</Text>
            </Tooltip>
          </div>
          {value.model_type === 'moe' && value.moe_config && (
            <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #e8e8e8' }}>
              <Tooltip title="Mixture of Experts: 稀疏激活架构，每次只激活部分专家，提高模型容量的同时控制计算量">
                <Tag color="purple" style={{ cursor: 'help' }}>MoE</Tag>
              </Tooltip>
              <Tooltip title={`总共${value.moe_config.num_experts}个专家，每个token激活${value.moe_config.num_experts_per_tok}个`}>
                <Text type="secondary" style={{ fontSize: 13, cursor: 'help' }}>
                  {value.moe_config.num_experts}专家 × {value.moe_config.num_experts_per_tok}激活
                  {value.moe_config.num_shared_experts ? ` + ${value.moe_config.num_shared_experts}共享` : ''}
                </Text>
              </Tooltip>
            </div>
          )}
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #e8e8e8', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tooltip title={<pre style={{ margin: 0, fontSize: 13, whiteSpace: 'pre-wrap' }}>{estimateParams().breakdown}</pre>}>
              <Text type="secondary" style={{ cursor: 'help' }}>估算参数量: <b>{estimateParams().value}</b></Text>
            </Tooltip>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
        {editMode && (
          <Button
            size="middle"
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
          size="middle"
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
            size="middle"
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
            size="middle"
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
            size="middle"
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
            <Button size="middle" onClick={() => setSaveModalVisible(false)}>取消</Button>
            <Button size="middle" type="primary" onClick={handleSaveModel}>保存</Button>
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
}

/** 记录上次使用的 Benchmark */
const LAST_BENCHMARK_KEY = 'llm_last_benchmark_id'

/** 格式化序列长度 */
function formatSeqLen(len: number): string {
  if (len >= 1024 && len % 1024 === 0) {
    return `${len / 1024}K`
  }
  return String(len)
}

/** 获取数据类型的位数 */
function getDtypeBits(dtype: string): number {
  const bitsMap: Record<string, number> = {
    'fp32': 32, 'fp16': 16, 'bf16': 16, 'fp8': 8, 'int8': 8, 'int4': 4
  }
  return bitsMap[dtype] || 16
}

/**
 * 解析模型名称和参数量
 * "DeepSeek-V3-671B" → { name: "DeepSeek-V3", size: "671B" }
 */
function parseModelName(modelName: string): { name: string; size: string } {
  const sizeMatch = modelName.match(/(\d+\.?\d*)[BMK]/i)

  if (sizeMatch) {
    const size = sizeMatch[1] + sizeMatch[0].slice(-1).toUpperCase()
    const name = modelName
      .replace(/[-_]?\d+\.?\d*[BMK][-_]?/i, '')
      .replace(/-+$/, '')
      .replace(/^-+/, '')
      .replace(/-Instruct|-Chat|-Base/i, '')
      .trim()
    return { name, size }
  }

  return { name: modelName, size: '' }
}

/** 生成 Benchmark 名称: DeepSeek-V3-671B-S4K-O512-W16A16-B8 */
export function generateBenchmarkName(model: LLMModelConfig, inference: InferenceConfig): string {
  const { name, size } = parseModelName(model.model_name)
  const seqIn = formatSeqLen(inference.input_seq_length)
  const seqOut = formatSeqLen(inference.output_seq_length)
  const wBits = getDtypeBits(model.weight_dtype)
  const aBits = getDtypeBits(model.activation_dtype)

  // 构建名称
  const parts = [
    size ? `${name}-${size}` : name,
    `S${seqIn}`,
    `O${seqOut}`,
    `W${wBits}A${aBits}`,
    `B${inference.batch_size}`
  ]

  return parts.join('-')
}

/** Benchmark 类型 */
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
}) => {
  const [presetId, setPresetId] = useState<string>('')
  const [customBenchmarks, setCustomBenchmarks] = useState<CustomBenchmark[]>([])
  const [activeKeys, setActiveKeys] = useState<string[]>(['basic']) // 默认展开基础参数
  const [paramsStr, setParamsStr] = useState<string>('--')

  const modelList = getModelList()

  // 异步获取模型参数量
  useEffect(() => {
    let cancelled = false
    calculateModelParams(modelConfig)
      .then((res) => {
        if (!cancelled) setParamsStr(res.formatted)
      })
      .catch(() => {
        if (!cancelled) setParamsStr('--')
      })
    return () => { cancelled = true }
  }, [modelConfig])

  // 从后端加载自定义 benchmarks
  useEffect(() => {
    listBenchmarks().then((benchmarks) => {
      const mapped = benchmarks.map(b => ({
        id: b.id,
        name: b.name,
        model: b.model as unknown as LLMModelConfig,
        inference: b.inference as unknown as InferenceConfig,
      }))
      setCustomBenchmarks(mapped)

      // 设置初始 benchmark：优先使用上次的，否则使用第一个
      if (mapped.length > 0) {
        const lastBenchmarkId = localStorage.getItem(LAST_BENCHMARK_KEY)
        const initialId = (lastBenchmarkId && mapped.find(b => b.id === lastBenchmarkId))
          ? lastBenchmarkId
          : mapped[0].id

        setPresetId(initialId)
        const initialBenchmark = mapped.find(b => b.id === initialId)
        if (initialBenchmark) {
          onModelChange(initialBenchmark.model)
          onInferenceChange(initialBenchmark.inference)
        }
      }
    })
  }, [])

  // 当前 Benchmark 名称
  const currentBenchmarkName = generateBenchmarkName(modelConfig, inferenceConfig)

  // 检查当前配置是否与选中的 benchmark 匹配
  const isConfigModified = useCallback(() => {
    const match = customBenchmarks.find(c => c.id === presetId)
    if (match) {
      return generateBenchmarkName(match.model, match.inference) !== currentBenchmarkName
    }
    return true
  }, [presetId, currentBenchmarkName, customBenchmarks])

  // 选择 benchmark
  const handlePresetChange = (id: string) => {
    setPresetId(id)
    // 保存到 localStorage，记录本次选择
    localStorage.setItem(LAST_BENCHMARK_KEY, id)
    const match = customBenchmarks.find(c => c.id === id)
    if (match) {
      onModelChange(match.model)
      onInferenceChange(match.inference)
    }
  }

  // 保存当前配置
  const handleSave = async () => {
    // 如果配置已修改，创建新的 benchmark（使用 benchmark 名称作为 ID）
    if (isConfigModified()) {
      const newBenchmark: CustomBenchmark = {
        id: currentBenchmarkName,
        name: currentBenchmarkName,
        model: { ...modelConfig },
        inference: { ...inferenceConfig },
      }
      // 保存到后端
      const success = await createBenchmark({
        id: currentBenchmarkName,
        name: currentBenchmarkName,
        model: modelConfig as unknown as Record<string, unknown>,
        inference: inferenceConfig as unknown as Record<string, unknown>,
      })
      if (success) {
        // 检查是否已存在同名 benchmark，如果存在则更新，否则添加
        const existingIndex = customBenchmarks.findIndex(b => b.id === currentBenchmarkName)
        if (existingIndex >= 0) {
          const updated = [...customBenchmarks]
          updated[existingIndex] = newBenchmark
          setCustomBenchmarks(updated)
        } else {
          setCustomBenchmarks([...customBenchmarks, newBenchmark])
        }
        setPresetId(currentBenchmarkName)
        // 保存到 localStorage，记录本次选择
        localStorage.setItem(LAST_BENCHMARK_KEY, currentBenchmarkName)
        message.success(`已保存: ${currentBenchmarkName}`)
      } else {
        message.error('保存失败')
      }
    }
  }

  // 重置到选中 Benchmark 的原始配置
  const handleReset = () => {
    const match = customBenchmarks.find(c => c.id === presetId)
    if (match) {
      onModelChange(match.model)
      onInferenceChange(match.inference)
      message.info('已重置到原始配置')
    }
  }

  // 另存为 Benchmark（使用 benchmark 名称作为 ID）
  const handleSaveAs = async () => {
    const newBenchmark: CustomBenchmark = {
      id: currentBenchmarkName,
      name: currentBenchmarkName,
      model: { ...modelConfig },
      inference: { ...inferenceConfig },
    }
    // 保存到后端
    const success = await createBenchmark({
      id: currentBenchmarkName,
      name: currentBenchmarkName,
      model: modelConfig as unknown as Record<string, unknown>,
      inference: inferenceConfig as unknown as Record<string, unknown>,
    })
    if (success) {
      // 检查是否已存在同名 benchmark
      const existingIndex = customBenchmarks.findIndex(b => b.id === currentBenchmarkName)
      if (existingIndex >= 0) {
        const updated = [...customBenchmarks]
        updated[existingIndex] = newBenchmark
        setCustomBenchmarks(updated)
      } else {
        setCustomBenchmarks([...customBenchmarks, newBenchmark])
      }
      setPresetId(currentBenchmarkName)
      // 保存到 localStorage，记录本次选择
      localStorage.setItem(LAST_BENCHMARK_KEY, currentBenchmarkName)
      message.success(`已保存: ${currentBenchmarkName}`)
    } else {
      message.error('保存失败')
    }
  }

  // 生成下拉选项：全部从后端读取
  const dropdownOptions = customBenchmarks.map(c => ({
    value: c.id,
    label: c.name,
  }))

  // 更新模型字段
  const updateModelField = <K extends keyof LLMModelConfig>(field: K, val: LLMModelConfig[K]) => {
    onModelChange({ ...modelConfig, [field]: val })
  }

  // 更新 MoE 字段
  const updateMoeField = <K extends keyof NonNullable<LLMModelConfig['moe_config']>>(
    field: K,
    val: NonNullable<LLMModelConfig['moe_config']>[K]
  ) => {
    if (modelConfig.moe_config) {
      onModelChange({ ...modelConfig, moe_config: { ...modelConfig.moe_config, [field]: val } })
    }
  }

  // 使用状态中的参数量字符串
  const estimateParams = () => paramsStr

  // 构建 Collapse items
  const collapseItems = [
    // 基础参数（包含 MoE）
    {
      key: 'basic',
      label: '基础参数',
      children: (
        <div>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="model_name" label="模型选择" /></div>
              <Select
                size="small"
                value={modelConfig.model_name}
                onChange={(name) => {
                  const preset = modelList.find(m => m.name === name || m.id === name)
                  if (preset) {
                    onModelChange(getModelPreset(preset.id))
                  }
                }}
                style={{ width: '100%' }}
                popupMatchSelectWidth={false}
                options={modelList.map(m => ({ value: m.id, label: m.params ? `${m.name} (${m.params})` : m.name }))}
              />
            </Col>
            <Col span={12}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="model_type" label="模型类型" /></div>
              <Select
                size="small"
                value={modelConfig.model_type}
                onChange={(v) => {
                  if (v === 'moe' && !modelConfig.moe_config) {
                    onModelChange({ ...modelConfig, model_type: v, moe_config: { num_experts: 8, num_experts_per_tok: 2, expert_capacity_factor: 1.25 } })
                  } else {
                    updateModelField('model_type', v)
                  }
                }}
                style={{ width: '100%' }}
                options={[
                  { value: 'dense', label: 'Dense' },
                  { value: 'moe', label: 'MoE' },
                ]}
              />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="hidden_size" label="隐藏层维度" /></div>
              <InputNumber size="small" min={64} max={65536} value={modelConfig.hidden_size}
                onChange={(v) => updateModelField('hidden_size', v || 4096)} style={{ width: '100%' }} />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="num_layers" label="层数" /></div>
              <InputNumber size="small" min={1} max={256} value={modelConfig.num_layers}
                onChange={(v) => updateModelField('num_layers', v || 32)} style={{ width: '100%' }} />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="intermediate_size" label="FFN维度" /></div>
              <InputNumber size="small" min={64} max={131072} value={modelConfig.intermediate_size}
                onChange={(v) => updateModelField('intermediate_size', v || 11008)} style={{ width: '100%' }} />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="vocab_size" label="词表大小" /></div>
              <InputNumber size="small" min={1000} max={500000} value={modelConfig.vocab_size}
                onChange={(v) => updateModelField('vocab_size', v || 32000)} style={{ width: '100%' }} />
            </Col>
          </Row>
          {/* MoE 参数（条件显示） */}
          {modelConfig.model_type === 'moe' && modelConfig.moe_config && (
            <>
              <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>MoE 参数</Divider>
              <Row gutter={[16, 8]}>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="num_experts" label="专家数量" /></div>
                  <InputNumber size="small" min={2} max={1024} value={modelConfig.moe_config?.num_experts}
                    onChange={(v) => updateMoeField('num_experts', v || 8)} style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="num_experts_per_tok" label="激活专家数" /></div>
                  <InputNumber size="small" min={1} max={64} value={modelConfig.moe_config?.num_experts_per_tok}
                    onChange={(v) => updateMoeField('num_experts_per_tok', v || 2)} style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="num_shared_experts" label="共享专家数" /></div>
                  <InputNumber size="small" min={0} max={16} value={modelConfig.moe_config?.num_shared_experts || 0}
                    onChange={(v) => updateMoeField('num_shared_experts', v || 0)} style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="expert_intermediate_size" label="专家FFN维度" /></div>
                  <InputNumber size="small" min={64} max={65536} value={modelConfig.moe_config?.expert_intermediate_size}
                    onChange={(v) => updateMoeField('expert_intermediate_size', v || undefined)} style={{ width: '100%' }}
                    placeholder="同FFN" />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="first_k_dense_replace" label="前K层Dense" /></div>
                  <InputNumber size="small" min={0} max={100} value={modelConfig.moe_config?.first_k_dense_replace || 0}
                    onChange={(v) => updateMoeField('first_k_dense_replace', v || 0)} style={{ width: '100%' }} />
                </Col>
              </Row>
            </>
          )}
        </div>
      ),
    },
    // 注意力配置（包含 MLA）
    {
      key: 'attention',
      label: '注意力配置',
      children: (
        <div>
          <Row gutter={[16, 8]}>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="num_attention_heads" label="注意力头数" /></div>
              <InputNumber size="small" min={1} max={256} value={modelConfig.num_attention_heads}
                onChange={(v) => updateModelField('num_attention_heads', v || 32)} style={{ width: '100%' }} />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="num_kv_heads" label="KV头数" /></div>
              <InputNumber size="small" min={1} max={256} value={modelConfig.num_kv_heads}
                onChange={(v) => updateModelField('num_kv_heads', v || 8)} style={{ width: '100%' }} />
            </Col>
            <Col span={8}>
              <div style={{ marginBottom: 4 }}><ConfigLabel name="attention_type" label="Attention类型" /></div>
              <Select size="small" value={modelConfig.attention_type || 'mha'} onChange={(v) => {
                if (v === 'mla' && !modelConfig.mla_config) {
                  onModelChange({ ...modelConfig, attention_type: v, mla_config: {
                    kv_lora_rank: 512,
                    q_lora_rank: 1536,
                    qk_nope_head_dim: 128,
                    qk_rope_head_dim: 64,
                    v_head_dim: 128,
                    variant: 'mla',
                  }})
                } else {
                  updateModelField('attention_type', v)
                }
              }} style={{ width: '100%' }}
                options={[
                  { value: 'mha', label: 'MHA' },
                  { value: 'gqa', label: 'GQA' },
                  { value: 'mqa', label: 'MQA' },
                  { value: 'mla', label: 'MLA' },
                ]}
              />
            </Col>
          </Row>
          {/* MLA 参数（条件显示） */}
          {modelConfig.attention_type === 'mla' && modelConfig.mla_config && (
            <>
              <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0 8px', fontSize: 12 }}>MLA 参数</Divider>
              <Row gutter={[16, 8]}>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="mla_variant" label="MLA 变体" /></div>
                  <Select size="small" value={modelConfig.mla_config?.variant || 'mla'}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, variant: v } })}
                    style={{ width: '100%' }}
                    options={[
                      { value: 'mla', label: 'MLA' },
                      { value: 'mla_v32', label: 'MLA V3.2' },
                      { value: 'mla_absorb', label: 'MLA Absorb' },
                      { value: 'mla_absorb_v32', label: 'MLA Absorb V3.2' },
                    ]}
                  />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="kv_lora_rank" label="KV LoRA Rank" /></div>
                  <InputNumber size="small" min={64} max={4096} value={modelConfig.mla_config?.kv_lora_rank}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, kv_lora_rank: v || 512 } })}
                    style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="q_lora_rank" label="Q LoRA Rank" /></div>
                  <InputNumber size="small" min={64} max={4096} value={modelConfig.mla_config?.q_lora_rank}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, q_lora_rank: v || 1536 } })}
                    style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="qk_nope_head_dim" label="QK Nope维度" /></div>
                  <InputNumber size="small" min={32} max={512} value={modelConfig.mla_config?.qk_nope_head_dim}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, qk_nope_head_dim: v || 128 } })}
                    style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="qk_rope_head_dim" label="QK RoPE维度" /></div>
                  <InputNumber size="small" min={32} max={512} value={modelConfig.mla_config?.qk_rope_head_dim}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, qk_rope_head_dim: v || 64 } })}
                    style={{ width: '100%' }} />
                </Col>
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="v_head_dim" label="V 头维度" /></div>
                  <InputNumber size="small" min={32} max={512} value={modelConfig.mla_config?.v_head_dim}
                    onChange={(v) => onModelChange({ ...modelConfig, mla_config: { ...modelConfig.mla_config!, v_head_dim: v || 128 } })}
                    style={{ width: '100%' }} />
                </Col>
              </Row>
            </>
          )}
        </div>
      ),
    },
    // 精度配置
    {
      key: 'precision',
      label: '精度配置',
      children: (
        <Row gutter={[16, 8]}>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="weight_dtype" label="权重精度" /></div>
            <Select size="small" value={modelConfig.weight_dtype} onChange={(v) => updateModelField('weight_dtype', v)} style={{ width: '100%' }}
              options={[
                { value: 'fp32', label: 'FP32' },
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp8', label: 'FP8' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="activation_dtype" label="激活精度" /></div>
            <Select size="small" value={modelConfig.activation_dtype} onChange={(v) => updateModelField('activation_dtype', v)} style={{ width: '100%' }}
              options={[
                { value: 'fp32', label: 'FP32' },
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp8', label: 'FP8' },
                { value: 'int8', label: 'INT8' },
                { value: 'int4', label: 'INT4' },
              ]}
            />
          </Col>
        </Row>
      ),
    },
    // 推理参数
    {
      key: 'inference',
      label: '推理参数',
      children: (
        <Row gutter={[16, 8]}>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="batch_size" label="Batch Size" /></div>
            <InputNumber size="small" min={1} max={512} value={inferenceConfig.batch_size}
              onChange={(v) => onInferenceChange({ ...inferenceConfig, batch_size: v || 1 })} style={{ width: '100%' }} />
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="input_seq_length" label="输入序列长度" /></div>
            <InputNumber size="small" min={1} max={131072} value={inferenceConfig.input_seq_length}
              onChange={(v) => onInferenceChange({ ...inferenceConfig, input_seq_length: v || 512 })} style={{ width: '100%' }} />
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="output_seq_length" label="输出序列长度" /></div>
            <InputNumber size="small" min={1} max={32768} value={inferenceConfig.output_seq_length}
              onChange={(v) => onInferenceChange({ ...inferenceConfig, output_seq_length: v || 256 })} style={{ width: '100%' }} />
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 4 }}><ConfigLabel name="max_seq_length" label="最大序列长度" /></div>
            <InputNumber size="small" min={inferenceConfig.input_seq_length + inferenceConfig.output_seq_length} max={131072}
              value={inferenceConfig.max_seq_length}
              onChange={(v) => onInferenceChange({ ...inferenceConfig, max_seq_length: v || 768 })} style={{ width: '100%' }} />
          </Col>
        </Row>
      ),
    },
  ]

  return (
    <div>
      {/* Benchmark 选择下拉框 */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ marginBottom: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            <span style={{ color: '#ff4d4f' }}>*</span> Benchmark 配置文件
          </Text>
          <Button
            type="link"
            size="small"
            icon={activeKeys.length === collapseItems.length ? <UpOutlined /> : <DownOutlined />}
            onClick={() => {
              if (activeKeys.length === collapseItems.length) {
                setActiveKeys([])
              } else {
                setActiveKeys(collapseItems.map(item => item.key))
              }
            }}
            style={{ padding: 0, height: 'auto', fontSize: 12 }}
          >
            {activeKeys.length === collapseItems.length ? '全部折叠' : '全部展开'}
          </Button>
        </div>
        <Select
          size="middle"
          value={presetId}
          onChange={handlePresetChange}
          style={{ width: '100%' }}
          popupMatchSelectWidth={false}
          options={dropdownOptions}
        />
      </div>

      {/* 折叠面板参数配置 */}
      <Collapse
        size="small"
        style={{ marginBottom: 12, background: 'transparent' }}
        items={collapseItems}
        activeKey={activeKeys}
        onChange={(keys) => setActiveKeys(keys as string[])}
        expandIconPosition="start"
        className="benchmark-collapse"
      />
      <style>{`
        .benchmark-collapse .ant-collapse-header {
          background: #f0f0f0 !important;
          border-radius: 4px !important;
          font-weight: 500;
        }
        .benchmark-collapse .ant-collapse-content {
          background: #fff;
        }
      `}</style>

      {/* 估算参数量 */}
      <div style={{ padding: '8px 12px', background: '#fafafa', borderRadius: 6, marginBottom: 12, fontSize: 13 }}>
        估算参数量: <b style={{ color: '#333' }}>{estimateParams()}</b>
      </div>

      {/* 操作按钮 */}
      <Space>
        <Button size="small" icon={<SaveOutlined />} onClick={handleSave}>
          保存
        </Button>
        <Button size="small" icon={<CopyOutlined />} onClick={handleSaveAs}>
          另存为
        </Button>
        <Button size="small" icon={<ReloadOutlined />} onClick={handleReset}>
          重置
        </Button>
      </Space>
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
          size="middle"
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
          size="middle"
          value={nodeId}
          onChange={handleNodeChange}
          style={{ width: 160 }}
          options={nodeOptions}
        />
      </div>
      <div style={configRowStyle}>
        <Text>节点数量</Text>
        <InputNumber
          size="middle"
          min={1}
          max={64}
          value={numNodes}
          onChange={(v) => handleNumNodesChange(v || 1)}
          style={{ width: 90 }}
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
