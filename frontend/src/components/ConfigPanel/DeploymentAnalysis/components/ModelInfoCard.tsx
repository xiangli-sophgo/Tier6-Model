/**
 * 模型架构可视化卡片
 * 左右布局：左边架构图，右边详情面板
 */

import React, { useState, useMemo } from 'react'
import { Tag } from 'antd'
import { LLMModelConfig, InferenceConfig } from '../../../../utils/llmDeployment/types'
import { calculateModelParams } from '../../../../utils/llmDeployment/modelCalculator'

interface ModelInfoCardProps {
  model: LLMModelConfig
  inference?: InferenceConfig
}

// 浅色配色 - 与整体风格搭配
const COLORS = {
  embedding: { bg: '#e6f4ff', border: '#91caff', text: '#0958d9' },
  attention: { bg: '#f9f0ff', border: '#d3adf7', text: '#722ed1' },
  ffn: { bg: '#f6ffed', border: '#b7eb8f', text: '#389e0d' },
  moe: { bg: '#fff0f6', border: '#ffadd2', text: '#c41d7f' },
  output: { bg: '#fff7e6', border: '#ffd591', text: '#d46b08' },
  wire: '#d9d9d9',
  wireActive: '#1677ff',
  text: '#262626',
  textSecondary: '#8c8c8c',
  bg: '#fafafa',
}

// 格式化数字
const formatNum = (n: number): string => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return n.toString()
}

// FLOPs 计算
const calculateFLOPs = (model: LLMModelConfig, inference?: InferenceConfig) => {
  const B = inference?.batch_size || 1
  const S = inference?.input_seq_length || 1024
  const H = model.hidden_size
  const I = model.intermediate_size
  const L = model.num_layers
  const n_h = model.num_attention_heads
  const n_kv = model.num_kv_heads
  const d_h = H / n_h
  const V = model.vocab_size

  const qkvProj = 2 * B * S * H * (H + 2 * (n_kv * d_h))
  const attnScore = 2 * B * n_h * S * S * d_h
  const attnOut = 2 * B * S * H * H
  const attnTotal = qkvProj + attnScore + attnOut

  let ffnTotal = 2 * 2 * B * S * H * I + 2 * B * S * I * H

  if (model.model_type === 'moe' && model.moe_config) {
    const expertI = model.moe_config.expert_intermediate_size || I
    const topK = model.moe_config.num_experts_per_tok
    const shared = model.moe_config.num_shared_experts || 0
    const firstKDense = model.moe_config.first_k_dense_replace || 0
    
    // MoE 层的 FFN FLOPs (激活专家数 * 单专家计算量 + Router)
    const moeFFN = (topK + shared) * (2 * 2 * B * S * H * expertI + 2 * B * S * expertI * H)
                 + 2 * B * S * H * model.moe_config.num_experts
    
    // Dense 层的 FFN FLOPs
    const denseFFN = 2 * 2 * B * S * H * I + 2 * B * S * I * H
    
    // 加权平均（考虑 Dense 和 MoE 层的比例）
    const numDenseLayers = Math.min(firstKDense, L)
    const numMoELayers = L - numDenseLayers
    ffnTotal = L > 0 ? (denseFFN * numDenseLayers + moeFFN * numMoELayers) / L : moeFFN
  }

  const embFLOPs = 2 * B * S * V * H
  const outFLOPs = 2 * B * S * H * V

  return {
    attention: attnTotal,
    ffn: ffnTotal,
    perLayer: attnTotal + ffnTotal,
    embedding: embFLOPs,
    output: outFLOPs,
    total: embFLOPs + L * (attnTotal + ffnTotal) + outFLOPs,
  }
}

// 参数量计算 - 使用统一的 calculateModelParams 计算总量
const calculateParams = (model: LLMModelConfig) => {
  const H = model.hidden_size
  const I = model.intermediate_size
  const L = model.num_layers
  const V = model.vocab_size
  const n_kv = model.num_kv_heads
  const d_h = H / model.num_attention_heads

  const embParams = V * H
  const attnParams = H * H + 2 * (n_kv * d_h) * H + H * H

  let ffnParams = 3 * H * I
  if (model.model_type === 'moe' && model.moe_config) {
    const E = model.moe_config.num_experts
    const S = model.moe_config.num_shared_experts || 0
    const expertI = model.moe_config.expert_intermediate_size || I
    const firstKDense = model.moe_config.first_k_dense_replace || 0

    // 区分 Dense 层和 MoE 层
    const numDenseLayers = Math.min(firstKDense, L)
    const numMoELayers = L - numDenseLayers
    const denseFFN = 3 * H * I
    const moeFFN = (E + S) * 3 * H * expertI + H * E

    // 加权平均每层 FFN 参数
    ffnParams = L > 0 ? (denseFFN * numDenseLayers + moeFFN * numMoELayers) / L : moeFFN
  }

  const outParams = H * V

  return {
    embedding: embParams,
    attention: attnParams * L,
    ffn: ffnParams * L,
    output: outParams,
    total: calculateModelParams(model),
  }
}

// 小节标题样式
const SubSectionTitle: React.FC<{ title: string }> = ({ title }) => (
  <div style={{
    fontSize: 12,
    fontWeight: 600,
    color: COLORS.text,
    marginBottom: 6,
    marginTop: 10,
    paddingTop: 8,
    borderTop: '1px dashed #e8e8e8',
  }}>
    {title}
  </div>
)

// 两列参数网格
const ParamGrid: React.FC<{ items: { label: string; value: string | number }[]; title?: string }> = ({ items, title }) => (
  <div>
    {title && <SubSectionTitle title={title} />}
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px', marginBottom: 8 }}>
      {items.map((item, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '2px 0' }}>
          <span style={{ color: COLORS.textSecondary }}>{item.label}:</span>
          <span style={{ fontWeight: 500, color: COLORS.text, fontFamily: 'ui-monospace, monospace' }}>{item.value}</span>
        </div>
      ))}
    </div>
  </div>
)

// 详情面板
const DetailSection: React.FC<{ title: string; color: typeof COLORS.embedding; children: React.ReactNode }> = ({ title, color, children }) => (
  <div style={{ marginBottom: 12 }}>
    <div style={{
      fontSize: 14,
      fontWeight: 600,
      color: color.text,
      marginBottom: 6,
      paddingBottom: 4,
      borderBottom: `2px solid ${color.border}`,
    }}>
      {title}
    </div>
    {children}
  </div>
)

export const ModelInfoCard: React.FC<ModelInfoCardProps> = ({ model, inference }) => {
  const [selectedBlock, setSelectedBlock] = useState<string>('overview')

  const isMoE = model.model_type === 'moe' && model.moe_config
  const isMLA = model.attention_type === 'mla' && model.mla_config
  const params = useMemo(() => calculateParams(model), [model])
  const flops = useMemo(() => calculateFLOPs(model, inference), [model, inference])

  const H = model.hidden_size
  const I = model.intermediate_size
  const n_h = model.num_attention_heads
  const n_kv = model.num_kv_heads
  const d_h = H / n_h

  // SVG 尺寸 - 根据内容调整高度
  const svgWidth = 500
  const svgHeight = isMoE ? 625 : 565
  const centerX = svgWidth / 2

  // 块样式
  const getBlockStyle = (key: string, color: typeof COLORS.embedding) => ({
    fill: color.bg,
    stroke: selectedBlock === key ? COLORS.wireActive : color.border,
    strokeWidth: selectedBlock === key ? 2 : 1,
    cursor: 'pointer',
  })

  // 操作步骤组件 - 更详细的说明
  const StepList: React.FC<{ items: { name: string; desc: string; detail?: string }[]; title?: string }> = ({ items, title = '操作流程' }) => (
    <div>
      <SubSectionTitle title={title} />
      <div style={{ fontSize: 12 }}>
        {items.map((item, i) => (
          <div key={i} style={{ marginBottom: 6, paddingLeft: 4 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start' }}>
              <span style={{ color: '#1677ff', fontWeight: 600, minWidth: 20 }}>{i + 1}.</span>
              <div>
                <b style={{ color: COLORS.text }}>{item.name}</b>
                <span style={{ color: COLORS.textSecondary }}>：{item.desc}</span>
                {item.detail && <div style={{ color: '#999', marginTop: 2, fontSize: 11 }}>{item.detail}</div>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  // 详情内容
  const detailContent: Record<string, React.ReactNode> = {
    embedding: (
      <DetailSection title="Embedding Layer" color={COLORS.embedding}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          将离散的 Token ID 映射为连续的高维向量表示，是模型理解文本的第一步。
          <div style={{ marginTop: 4, fontFamily: 'ui-monospace, monospace', color: COLORS.text }}>
            维度变化：[B, S] → [B, S, H]
          </div>
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '词表大小 V', value: formatNum(model.vocab_size) },
          { label: '隐藏维度 H', value: formatNum(H) },
          { label: '位置编码', value: 'RoPE' },
          { label: '参数量', value: formatNum(params.embedding) },
        ]} />
        <StepList items={[
          { name: 'Token Embedding', desc: '查表映射', detail: '输入 Token ID (整数)，从 V×H 的嵌入矩阵中查找对应的 H 维向量' },
          { name: 'RoPE 位置编码', desc: '旋转位置编码', detail: '通过旋转变换将位置信息编码到向量中，使模型能够区分不同位置的 Token' },
        ]} />
      </DetailSection>
    ),
    attention: (
      <DetailSection title={`${isMLA ? 'MLA' : model.attention_type?.toUpperCase() || 'GQA'} Attention`} color={COLORS.attention}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          {isMLA
            ? 'Multi-head Latent Attention（多头潜在注意力）：DeepSeek 独创的注意力机制，通过低秩压缩技术将 KV Cache 压缩数倍，在保持模型性能的同时大幅降低推理时的显存占用。'
            : `GQA（Grouped Query Attention，分组查询注意力）：将多个 Query 头共享同一组 Key-Value 头，在保持模型表达能力的同时减少 KV Cache 的显存占用和计算量。本模型使用 ${n_h} 个 Q 头共享 ${n_kv} 个 KV 头。`}
          <div style={{ marginTop: 4, fontFamily: 'ui-monospace, monospace', color: COLORS.text }}>
            维度变化：[B, S, H] → [B, S, H]（维度不变）
          </div>
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '注意力头', value: n_h },
          { label: 'KV 头', value: n_kv },
          { label: '头维度', value: d_h },
          { label: '参数量/层', value: formatNum(params.attention / model.num_layers) },
          ...(isMLA && model.mla_config ? [
            { label: 'Q LoRA', value: model.mla_config.q_lora_rank },
            { label: 'KV LoRA', value: model.mla_config.kv_lora_rank },
            { label: 'KV 压缩比', value: `${Math.round(H / model.mla_config.kv_lora_rank)}×` },
          ] : []),
        ]} />
        {isMLA ? (
          <StepList items={[
            { name: 'RMSNorm', desc: '输入归一化', detail: '对上一层的输出进行均方根归一化，消除不同样本间的数值差异，使训练更加稳定' },
            { name: 'Q LoRA 投影', desc: '低秩查询生成', detail: '通过低秩分解技术生成 Query 向量，先压缩再扩展，在减少参数量的同时保持表达能力' },
            { name: 'KV 压缩', desc: '键值缓存压缩', detail: '将 Key 和 Value 投影到低维空间存储，推理时可节省数倍显存，是 MLA 的核心创新' },
            { name: '注意力计算', desc: '相似度加权', detail: '计算当前位置与所有历史位置的相关性得分，决定应该关注哪些上下文信息' },
            { name: 'V 解压 + 输出投影', desc: '恢复并输出', detail: '将压缩的 Value 解压并与注意力权重加权求和，再通过线性变换生成最终输出' },
            { name: '+ Residual', desc: '残差连接', detail: '将注意力输出与原始输入相加，让梯度能够直接回传，解决深层网络训练困难的问题' },
          ]} />
        ) : (
          <StepList items={[
            { name: 'RMSNorm', desc: '输入归一化', detail: '对上一层的输出进行均方根归一化，消除不同样本间的数值差异，使训练更加稳定' },
            { name: 'QKV 投影', desc: '生成查询/键/值', detail: 'Query 用于表示"我要查找什么"，Key 用于表示"我有什么信息"，Value 是实际要传递的内容' },
            { name: '注意力计算', desc: '相似度加权', detail: '计算 Query 和所有 Key 的相似度得分，通过 Softmax 归一化后作为权重，对 Value 进行加权求和' },
            { name: '输出投影', desc: '多头融合输出', detail: '将多个注意力头捕获的不同模式信息拼接起来，通过线性变换融合成统一的表示' },
            { name: '+ Residual', desc: '残差连接', detail: '将注意力输出与原始输入相加，让梯度能够直接回传，解决深层网络训练困难的问题' },
          ]} />
        )}
      </DetailSection>
    ),
    ffn: (
      <DetailSection title="FFN 前馈网络" color={COLORS.ffn}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          FFN（Feed-Forward Network，前馈神经网络）：对每个位置的表示独立进行非线性变换。本模型采用 SwiGLU 变体，相比传统 FFN 有更好的性能。研究表明 FFN 层是 Transformer 存储事实知识的主要位置。
          <div style={{ marginTop: 4, fontFamily: 'ui-monospace, monospace', color: COLORS.text }}>
            维度变化：[B, S, H] → [B, S, I] → [B, S, H]
          </div>
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '隐藏维度 H', value: formatNum(H) },
          { label: '中间维度 I', value: formatNum(I) },
          { label: '扩展倍数', value: `${(I / H).toFixed(1)}×` },
          { label: 'FFN 类型', value: 'SwiGLU' },
          { label: '参数量/层', value: formatNum(params.ffn / model.num_layers) },
        ]} />
        <StepList items={[
          { name: 'RMSNorm', desc: '输入归一化', detail: '对注意力层的输出进行归一化，确保数值稳定，为后续计算提供一致的输入分布' },
          { name: 'Gate 投影', desc: 'H → I', detail: '将输入线性变换到中间维度 I，这个分支的输出将经过激活函数处理，用于控制信息流通' },
          { name: 'Up 投影', desc: 'H → I', detail: '将输入线性变换到中间维度 I，这个分支承载实际的特征信息，将与门控信号相乘' },
          { name: 'SiLU × Up', desc: '门控激活', detail: 'Gate 分支经过 SiLU 激活函数后与 Up 分支逐元素相乘，维度保持 I 不变' },
          { name: 'Down 投影', desc: 'I → H', detail: '将中间维度 I 压缩回隐藏维度 H，完成"扩展-压缩"的信息处理流程' },
          { name: '+ Residual', desc: '残差连接', detail: '将 FFN 输出与输入相加，确保原始信息不丢失，同时融入新学到的特征' },
        ]} />
      </DetailSection>
    ),
    moe: model.moe_config && (
      <DetailSection title="MoE 混合专家层" color={COLORS.moe}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          MoE（Mixture of Experts，混合专家）：用多个专家网络替代单一 FFN，每个 Token 只激活少量专家进行计算。这种稀疏激活机制使模型能够拥有巨大的参数量（存储更多知识），同时保持较低的计算成本。
          <div style={{ marginTop: 4, fontFamily: 'ui-monospace, monospace', color: COLORS.text }}>
            维度变化：[B, S, H] → [B, S, H]（维度不变）
          </div>
        </div>
        {model.model_name?.toLowerCase().includes('deepseek') && (
          <div style={{ background: '#fff0f6', border: '1px solid #ffadd2', borderRadius: 4, padding: '6px 8px', marginBottom: 8, fontSize: 12 }}>
            <b style={{ color: COLORS.moe.text }}>DeepSeek 层分布：</b>
            <span style={{ color: COLORS.textSecondary }}>Layer 0-2 使用 Dense FFN，Layer 3-{model.num_layers - 1} 使用 MoE</span>
          </div>
        )}
        <ParamGrid title="关键参数" items={[
          { label: '专家总数', value: model.moe_config.num_experts },
          { label: '激活专家', value: `Top-${model.moe_config.num_experts_per_tok}` },
          { label: '共享专家', value: model.moe_config.num_shared_experts || 0 },
          { label: '专家维度', value: formatNum(model.moe_config.expert_intermediate_size || I) },
          { label: '参数量/层', value: formatNum(params.ffn / model.num_layers) },
        ]} />
        <StepList items={[
          { name: 'RMSNorm', desc: '输入归一化', detail: '对注意力层的输出进行归一化，为路由决策和专家计算提供稳定的输入' },
          { name: '路由计算', desc: '专家选择决策', detail: '路由网络根据输入内容计算每个专家的匹配分数，决定当前 Token 应该由哪些专家处理' },
          { name: 'Top-K 选择', desc: '稀疏激活', detail: '只选择得分最高的 K 个专家参与计算，其他专家不激活，大幅减少计算量' },
          { name: 'AllToAll 分发', desc: '跨设备传输', detail: '在分布式训练中，将 Token 发送到对应专家所在的 GPU，实现专家并行' },
          { name: '路由专家计算', desc: '专家独立处理', detail: '每个专家是一个独立的 FFN 网络，专门处理路由给它的 Token，不同专家学习不同类型的知识' },
          { name: '共享专家计算', desc: '通用特征提取', detail: '共享专家处理所有 Token，提取通用特征，与路由专家互补，提升模型整体表现' },
          { name: 'AllToAll 收集', desc: '结果汇总', detail: '将分散在各 GPU 上的专家计算结果收集回来，准备进行汇总' },
          { name: '加权求和 + 残差', desc: '融合输出', detail: '按路由分数对各专家输出加权求和，加上共享专家的贡献，最后与输入残差连接' },
        ]} />
      </DetailSection>
    ),
    output: (
      <DetailSection title="LM Head 语言模型头" color={COLORS.output}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          LM Head（Language Model Head，语言模型头）：将 Transformer 最后一层输出的隐藏状态转换为词表上的概率分布，用于预测下一个 Token。这是模型从"理解"到"生成"的关键转换步骤。
          <div style={{ marginTop: 4, fontFamily: 'ui-monospace, monospace', color: COLORS.text }}>
            维度变化：[B, S, H] → [B, S, V]
          </div>
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '输入维度 H', value: formatNum(H) },
          { label: '输出维度 V', value: formatNum(model.vocab_size) },
          { label: '权重共享', value: '是' },
          { label: '参数量', value: formatNum(params.output) },
        ]} />
        <StepList items={[
          { name: 'Final RMSNorm', desc: '最终归一化', detail: '对 Transformer 最后一层的输出进行归一化，确保输入到分类器的数值稳定' },
          { name: '线性投影', desc: 'H → V', detail: '将隐藏状态投影到词表维度 V，通常与输入 Embedding 共享权重以减少参数量并提升效果' },
          { name: 'Softmax', desc: '概率分布生成', detail: '将投影得到的原始分数转换为概率分布，每个位置表示对应词的预测概率' },
        ]} />
      </DetailSection>
    ),
    // 整体流程概览（默认视图）
    overview: (
      <DetailSection title="模型架构概览" color={{ bg: '#e6f7ff', border: '#91d5ff', text: '#0050b3' }}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 10, lineHeight: 1.6 }}>
          {model.model_name} 是一个 {model.num_layers} 层的大型语言模型，采用 {isMLA ? 'MLA (Multi-head Latent Attention)' : 'GQA (Grouped Query Attention)'} 注意力机制
          {isMoE && `和 MoE (Mixture of Experts) 稀疏架构`}。
        </div>
        <div style={{ background: '#f0f5ff', border: '1px solid #adc6ff', borderRadius: 4, padding: '6px 10px', marginBottom: 10, fontSize: 11 }}>
          <div style={{ fontWeight: 600, color: '#1d39c4', marginBottom: 4 }}>符号说明</div>
          <div style={{ color: COLORS.textSecondary, lineHeight: 1.6 }}>
            <span style={{ marginRight: 12 }}><b>B</b>=批次大小</span>
            <span style={{ marginRight: 12 }}><b>S</b>=序列长度</span>
            <span style={{ marginRight: 12 }}><b>H</b>=隐藏维度({formatNum(H)})</span>
            <span style={{ marginRight: 12 }}><b>V</b>=词表大小({formatNum(model.vocab_size)})</span>
            <span><b>I</b>=中间维度({formatNum(I)})</span>
          </div>
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '总参数量', value: formatNum(params.total) },
          { label: '隐藏维度 H', value: formatNum(H) },
          { label: '层数 L', value: model.num_layers },
          { label: '词表大小 V', value: formatNum(model.vocab_size) },
          { label: '注意力头', value: n_h },
          { label: 'KV 头', value: n_kv },
          ...(isMoE && model.moe_config ? [
            { label: '专家数', value: model.moe_config.num_experts },
            { label: '激活专家', value: model.moe_config.num_experts_per_tok },
          ] : []),
        ]} />
        <StepList items={[
          { name: 'Embedding', desc: '[B,S] → [B,S,H]', detail: '将 Token ID 映射为 H 维向量，加入 RoPE 位置编码' },
          { name: 'Transformer ×' + model.num_layers, desc: '[B,S,H] → [B,S,H]', detail: `每层包含 ${isMLA ? 'MLA' : 'Attention'} 和 ${isMoE ? 'MoE' : 'FFN'}，维度保持不变` },
          { name: 'Final RMSNorm', desc: '归一化', detail: '对最后一层输出进行 RMSNorm 归一化' },
          { name: 'LM Head', desc: '[B,S,H] → [B,S,V]', detail: '映射到词表空间，预测下一个 Token 的概率分布' },
        ]} />
      </DetailSection>
    ),
    // Transformer 层说明
    transformer: (
      <DetailSection title="Transformer Layer" color={{ bg: '#f0f0f0', border: '#d9d9d9', text: '#595959' }}>
        <div style={{ fontSize: 13, color: COLORS.textSecondary, marginBottom: 8, lineHeight: 1.5 }}>
          Transformer 层是模型的核心组件，由注意力机制和前馈网络组成，共 {model.num_layers} 层堆叠。
        </div>
        <ParamGrid title="关键参数" items={[
          { label: '层数', value: model.num_layers },
          { label: '隐藏维度', value: formatNum(H) },
          { label: '注意力类型', value: isMLA ? 'MLA' : (model.attention_type?.toUpperCase() || 'GQA') },
          { label: 'FFN 类型', value: isMoE ? 'MoE' : 'Dense' },
        ]} />
        <StepList items={[
          { name: 'Pre-LN 架构', desc: '归一化在前', detail: '每个子层前先做 RMSNorm，比 Post-LN 更稳定' },
          { name: '注意力子层', desc: '自注意力机制', detail: `${isMLA ? 'MLA' : 'GQA'} 注意力，捕获序列中的依赖关系` },
          { name: 'FFN 子层', desc: isMoE ? 'MoE 稀疏计算' : 'SwiGLU FFN', detail: isMoE ? '稀疏专家混合，大容量低计算' : '全连接前馈网络，存储知识' },
          { name: '残差连接', desc: '信息直通', detail: '每个子层都有残差连接，将输入直接加到输出上，帮助梯度流动' },
        ]} />
      </DetailSection>
    ),
  }

  // 头部信息
  const headerContent = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: COLORS.text }}>{model.model_name}</span>
        <Tag color="blue" style={{ margin: 0 }}>{model.attention_type?.toUpperCase() || 'GQA'}</Tag>
        {isMoE && <Tag color="magenta" style={{ margin: 0 }}>MoE</Tag>}
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 12, color: COLORS.textSecondary }}>
        <span><b style={{ color: '#1677ff' }}>{formatNum(params.total)}</b> Params</span>
        <span><b style={{ color: '#52c41a' }}>{formatNum(flops.total)}</b> FLOPs</span>
        <span>{model.num_layers} Layers</span>
      </div>
    </div>
  )

  const cardContent = (
    <div style={{ display: 'flex', gap: 24 }}>
      {/* 左侧：架构图 - 占更大比例 */}
      <div style={{ flex: '0 0 55%', minWidth: 0 }}>
        <svg
          width="100%"
          height={svgHeight}
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ display: 'block', fontFamily: '"Times New Roman", Times, serif' }}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker id="arrow" markerWidth="12" markerHeight="10" refX="6" refY="5" orient="auto" markerUnits="userSpaceOnUse">
              <polygon points="0 0, 12 5, 0 10" fill={COLORS.wire} />
            </marker>
          </defs>

          {/* 背景点击区域 - 点击空白处返回整体流程 */}
          <rect
            x={0} y={0} width={svgWidth} height={svgHeight}
            fill="transparent"
            onClick={() => setSelectedBlock('overview')}
            style={{ cursor: selectedBlock !== 'overview' ? 'pointer' : 'default' }}
          />

          {/* 符号说明 */}
          <g transform="translate(12, 12)">
            <text x={0} y={0} fontSize={10} fill={COLORS.textSecondary}>
              <tspan x={0} dy={0}>B=批次 S=序列长度</tspan>
              <tspan x={0} dy={12}>H=隐藏维度 V=词表</tspan>
            </text>
          </g>

          {/* Input */}
          <text x={centerX} y={24} textAnchor="middle" fontSize={15} fontWeight={500} fill={COLORS.text}>
            输入 Token IDs [B, S]
          </text>

          {/* Arrow - 线段 + 三角形 */}
          <line x1={centerX} y1={30} x2={centerX} y2={46} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},54 ${centerX - 6},44 ${centerX + 6},44`} fill={COLORS.wire} />

          {/* Embedding */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('embedding') }} style={{ cursor: 'pointer' }}>
            <rect x={centerX - 130} y={54} width={260} height={54} rx={6} {...getBlockStyle('embedding', COLORS.embedding)} />
            <text x={centerX} y={76} textAnchor="middle" fontSize={16} fontWeight={600} fill={COLORS.embedding.text}>
              Embedding
            </text>
            <text x={centerX} y={94} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>
              [B, S] → [B, S, H]
            </text>
          </g>

          {/* Arrow */}
          <line x1={centerX} y1={108} x2={centerX} y2={124} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},132 ${centerX - 6},122 ${centerX + 6},122`} fill={COLORS.wire} />

          {/* Transformer Layer Box */}
          <rect x={20} y={132} width={svgWidth - 40} height={isMoE ? 350 : 290} rx={8} fill="none" stroke={COLORS.wire} strokeWidth={1.5} strokeDasharray="6,3" />
          <text x={35} y={156} fontSize={14} fontWeight={500} fill={COLORS.textSecondary}>
            Transformer × {model.num_layers}
          </text>

          {/* Attention - MLA 或标准 GQA */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('attention') }} style={{ cursor: 'pointer' }}>
            <rect x={35} y={168} width={200} height={isMoE ? 300 : 230} rx={6} {...getBlockStyle('attention', COLORS.attention)} />
            {/* 标题 */}
            <text x={135} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.attention.text}>
              {isMLA ? 'MLA' : model.attention_type?.toUpperCase() || 'GQA'}
            </text>
            {/* Pre-LN: RMSNorm */}
            <rect x={53} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
            <text x={135} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

            {isMLA && model.mla_config ? (
              /* MLA 完整流程 - 分叉数据流，间距加大适配 300 高度 */
              <>
                {/* RMSNorm 后分叉箭头 - 左边Q路径(中心100)，右边KV路径(中心186) */}
                <line x1={100} y1={220} x2={100} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="100,238 96,228 104,228" fill={COLORS.wire} />
                <line x1={186} y1={220} x2={186} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="186,238 182,228 190,228" fill={COLORS.wire} />

                {/* Q LoRA: 低秩Q生成 */}
                <g transform="translate(53, 238)">
                  <rect width={94} height={60} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={47} y={20} textAnchor="middle" fontSize={11} fontWeight={500} fill={COLORS.attention.text}>Q LoRA</text>
                  <text x={47} y={38} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>低秩压缩</text>
                  <text x={47} y={52} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>生成 Q</text>
                </g>

                {/* KV 压缩 */}
                <g transform="translate(155, 238)">
                  <rect width={62} height={60} rx={4} fill={COLORS.attention.bg} stroke={COLORS.attention.border} strokeWidth={2} />
                  <text x={31} y={18} textAnchor="middle" fontSize={11} fontWeight={600} fill={COLORS.attention.text}>KV</text>
                  <text x={31} y={36} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>压缩</text>
                  <text x={31} y={52} textAnchor="middle" fontSize={10} fill={COLORS.attention.text}>{Math.round(H / (model.mla_config.kv_lora_rank || 512))}×</text>
                </g>

                {/* 汇合箭头 - Q和KV汇合到Attention */}
                <line x1={100} y1={298} x2={100} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={186} y1={298} x2={186} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={100} y1={312} x2={186} y2={312} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={135} y1={312} x2={135} y2={322} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,330 131,320 139,320" fill={COLORS.wire} />

                {/* 注意力计算 */}
                <g transform="translate(53, 330)">
                  <rect width={164} height={30} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={20} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>注意力计算</text>
                </g>

                {/* 垂直流动箭头 */}
                <line x1={135} y1={360} x2={135} y2={372} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,380 131,370 139,370" fill={COLORS.wire} />

                {/* V 解压 + 输出投影 */}
                <g transform="translate(53, 380)">
                  <rect width={164} height={30} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={20} textAnchor="middle" fontSize={12} fill={COLORS.attention.text}>V 解压 + 输出投影</text>
                </g>

                {/* 垂直流动箭头 */}
                <line x1={135} y1={410} x2={135} y2={422} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,430 131,420 139,420" fill={COLORS.wire} />

                {/* Residual Add */}
                <g transform="translate(53, 430)">
                  <rect width={164} height={26} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                  <text x={82} y={17} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
                </g>

              </>
            ) : (
              /* 标准 GQA/MHA - 带数据流箭头 */
              <>
                {/* RMSNorm 到 QKV 的流动箭头 */}
                <line x1={135} y1={220} x2={135} y2={226} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,232 131,224 139,224" fill={COLORS.wire} />

                {/* Q K V 投影 */}
                <g transform="translate(53, 232)">
                  {['Q', 'K', 'V'].map((label, i) => (
                    <g key={label} transform={`translate(${i * 55}, 0)`}>
                      <rect width={50} height={30} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                      <text x={25} y={20} textAnchor="middle" fontSize={13} fontWeight={500} fill={COLORS.attention.text}>{label}</text>
                    </g>
                  ))}
                </g>

                {/* QKV 汇聚到 Attention 的箭头 */}
                <line x1={78} y1={262} x2={78} y2={272} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={135} y1={262} x2={135} y2={272} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={192} y1={262} x2={192} y2={272} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={78} y1={272} x2={192} y2={272} stroke={COLORS.wire} strokeWidth={1.5} />
                <line x1={135} y1={272} x2={135} y2={280} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,286 131,278 139,278" fill={COLORS.wire} />

                {/* Dot-Product Attention */}
                <g transform="translate(53, 286)">
                  <rect width={164} height={28} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={19} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>Scaled Dot-Product</text>
                </g>

                {/* Attention 到 Output Proj 的箭头 */}
                <line x1={135} y1={314} x2={135} y2={322} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,328 131,320 139,320" fill={COLORS.wire} />

                {/* Output Projection */}
                <g transform="translate(53, 328)">
                  <rect width={164} height={28} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                  <text x={82} y={19} textAnchor="middle" fontSize={12} fill={COLORS.attention.text}>输出投影</text>
                </g>

                {/* Output Proj 到 Residual 的箭头 */}
                <line x1={135} y1={356} x2={135} y2={364} stroke={COLORS.wire} strokeWidth={1.5} />
                <polygon points="135,370 131,362 139,362" fill={COLORS.wire} />

                {/* Residual Add */}
                <g transform="translate(53, 370)">
                  <rect width={164} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                  <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
                </g>
              </>
            )}
          </g>

          {/* Arrow between Attention and FFN - 线段 + 三角形 */}
          <line x1={235} y1={isMoE ? 330 : 300} x2={250} y2={isMoE ? 330 : 300} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`260,${isMoE ? 330 : 300} 250,${isMoE ? 324 : 294} 250,${isMoE ? 336 : 306}`} fill={COLORS.wire} />

          {/* FFN / MoE */}
          {isMoE && model.moe_config ? (
            <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('moe') }} style={{ cursor: 'pointer' }}>
              <rect x={260} y={168} width={200} height={300} rx={6} {...getBlockStyle('moe', COLORS.moe)} />
              {/* 标题 */}
              <text x={360} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.moe.text}>
                MoE
              </text>
              {/* Pre-LN: RMSNorm */}
              <rect x={278} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
              <text x={360} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

              {/* RMSNorm 到 Router 的流动箭头 */}
              <line x1={360} y1={220} x2={360} y2={226} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,232 356,224 364,224" fill={COLORS.wire} />

              {/* 路由 */}
              <g transform="translate(278, 232)">
                <rect width={164} height={28} rx={4} fill="#fff" stroke={COLORS.moe.border} strokeWidth={1.5} />
                <text x={82} y={19} textAnchor="middle" fontSize={12} fill={COLORS.moe.text}>
                  路由 → Top-{model.moe_config.num_experts_per_tok}
                </text>
              </g>

              {/* Router 到 AllToAll 的流动箭头 */}
              <line x1={360} y1={260} x2={360} y2={266} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,272 356,264 364,264" fill={COLORS.wire} />

              {/* AllToAll 分发 */}
              <g transform="translate(278, 272)">
                <rect width={164} height={24} rx={4} fill={COLORS.bg} stroke={COLORS.wire} strokeDasharray="4,2" strokeWidth={1.5} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>AllToAll 分发</text>
              </g>

              {/* AllToAll 到 Experts 的流动箭头 */}
              <line x1={360} y1={296} x2={360} y2={302} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,308 356,300 364,300" fill={COLORS.wire} />

              {/* 专家层 - 并行分支 */}
              <g transform="translate(278, 308)">
                {/* 分支箭头 - 从中间分出两路 */}
                {(model.moe_config.num_shared_experts || 0) > 0 ? (
                  <>
                    <line x1={82} y1={0} x2={42} y2={12} stroke={COLORS.wire} strokeWidth={1.5} />
                    <line x1={82} y1={0} x2={122} y2={12} stroke={COLORS.wire} strokeWidth={1.5} />
                  </>
                ) : null}

                {/* 激活专家 */}
                <rect x={0} y={(model.moe_config.num_shared_experts || 0) > 0 ? 14 : 0} width={(model.moe_config.num_shared_experts || 0) > 0 ? 80 : 164} height={44} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={(model.moe_config.num_shared_experts || 0) > 0 ? 40 : 82} y={(model.moe_config.num_shared_experts || 0) > 0 ? 32 : 18} textAnchor="middle" fontSize={11} fontWeight={600} fill={COLORS.ffn.text}>激活专家</text>
                <text x={(model.moe_config.num_shared_experts || 0) > 0 ? 40 : 82} y={(model.moe_config.num_shared_experts || 0) > 0 ? 48 : 36} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>Top-{model.moe_config.num_experts_per_tok} / {model.moe_config.num_experts}</text>

                {/* 共享专家 */}
                {(model.moe_config.num_shared_experts || 0) > 0 && (
                  <>
                    <rect x={84} y={14} width={80} height={44} rx={4} fill="#fff" stroke={COLORS.attention.border} strokeWidth={1.5} />
                    <text x={124} y={32} textAnchor="middle" fontSize={11} fontWeight={600} fill={COLORS.attention.text}>共享专家</text>
                    <text x={124} y={48} textAnchor="middle" fontSize={10} fill={COLORS.textSecondary}>×{model.moe_config.num_shared_experts}</text>
                  </>
                )}

                {/* 汇聚箭头 - 两路合并 */}
                {(model.moe_config.num_shared_experts || 0) > 0 ? (
                  <>
                    <line x1={42} y1={58} x2={82} y2={68} stroke={COLORS.wire} strokeWidth={1.5} />
                    <line x1={122} y1={58} x2={82} y2={68} stroke={COLORS.wire} strokeWidth={1.5} />
                    <polygon points="82,74 78,66 86,66" fill={COLORS.wire} />
                  </>
                ) : (
                  <>
                    <line x1={82} y1={44} x2={82} y2={50} stroke={COLORS.wire} strokeWidth={1.5} />
                    <polygon points="82,56 78,48 86,48" fill={COLORS.wire} />
                  </>
                )}
              </g>

              {/* 专家输出到 AllToAll 的流动箭头 */}
              <line x1={360} y1={(model.moe_config.num_shared_experts || 0) > 0 ? 382 : 364} x2={360} y2={(model.moe_config.num_shared_experts || 0) > 0 ? 388 : 370} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points={`360,${(model.moe_config.num_shared_experts || 0) > 0 ? 394 : 376} 356,${(model.moe_config.num_shared_experts || 0) > 0 ? 386 : 368} 364,${(model.moe_config.num_shared_experts || 0) > 0 ? 386 : 368}`} fill={COLORS.wire} />

              {/* AllToAll 收集 */}
              <g transform={`translate(278, ${(model.moe_config.num_shared_experts || 0) > 0 ? 394 : 376})`}>
                <rect width={164} height={24} rx={4} fill={COLORS.bg} stroke={COLORS.wire} strokeDasharray="4,2" strokeWidth={1.5} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>AllToAll 收集</text>
              </g>

              {/* AllToAll 到 Sum 的流动箭头 */}
              <line x1={360} y1={(model.moe_config.num_shared_experts || 0) > 0 ? 418 : 400} x2={360} y2={(model.moe_config.num_shared_experts || 0) > 0 ? 424 : 406} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points={`360,${(model.moe_config.num_shared_experts || 0) > 0 ? 430 : 412} 356,${(model.moe_config.num_shared_experts || 0) > 0 ? 422 : 404} 364,${(model.moe_config.num_shared_experts || 0) > 0 ? 422 : 404}`} fill={COLORS.wire} />

              {/* 加权求和 + 残差 */}
              <g transform={`translate(278, ${(model.moe_config.num_shared_experts || 0) > 0 ? 430 : 412})`}>
                <rect width={164} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>求和 + 残差</text>
              </g>
            </g>
          ) : (
            <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('ffn') }} style={{ cursor: 'pointer' }}>
              <rect x={260} y={168} width={200} height={230} rx={6} {...getBlockStyle('ffn', COLORS.ffn)} />
              {/* 标题 */}
              <text x={360} y={188} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.ffn.text}>
                FFN (SwiGLU)
              </text>
              {/* Pre-LN: RMSNorm */}
              <rect x={278} y={198} width={164} height={22} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
              <text x={360} y={213} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>RMSNorm</text>

              {/* RMSNorm 分叉到 Gate 和 Up */}
              <line x1={360} y1={220} x2={360} y2={224} stroke={COLORS.wire} strokeWidth={1.5} />
              <line x1={317} y1={224} x2={403} y2={224} stroke={COLORS.wire} strokeWidth={1.5} />
              <line x1={317} y1={224} x2={317} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="317,236 313,228 321,228" fill={COLORS.wire} />
              <line x1={403} y1={224} x2={403} y2={230} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="403,236 399,228 407,228" fill={COLORS.wire} />

              {/* Gate 投影 */}
              <g transform="translate(278, 236)">
                <rect width={78} height={30} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={39} y={20} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Gate</text>
              </g>
              {/* Up 投影 */}
              <g transform="translate(364, 236)">
                <rect width={78} height={30} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={39} y={20} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Up</text>
              </g>

              {/* Gate 和 Up 汇聚到 SiLU */}
              <line x1={317} y1={266} x2={317} y2={274} stroke={COLORS.wire} strokeWidth={1.5} />
              <line x1={403} y1={266} x2={403} y2={274} stroke={COLORS.wire} strokeWidth={1.5} />
              <line x1={317} y1={274} x2={403} y2={274} stroke={COLORS.wire} strokeWidth={1.5} />
              <line x1={360} y1={274} x2={360} y2={280} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,286 356,278 364,278" fill={COLORS.wire} />

              {/* SiLU 激活 */}
              <g transform="translate(295, 286)">
                <rect width={130} height={26} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={65} y={18} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>SiLU(Gate) ⊙ Up</text>
              </g>

              {/* SiLU 到 Down 的箭头 */}
              <line x1={360} y1={312} x2={360} y2={320} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,326 356,318 364,318" fill={COLORS.wire} />

              {/* Down 投影 */}
              <g transform="translate(295, 326)">
                <rect width={130} height={28} rx={4} fill="#fff" stroke={COLORS.ffn.border} strokeWidth={1.5} />
                <text x={65} y={19} textAnchor="middle" fontSize={12} fill={COLORS.ffn.text}>Down</text>
              </g>

              {/* Down 到 Residual 的箭头 */}
              <line x1={360} y1={354} x2={360} y2={362} stroke={COLORS.wire} strokeWidth={1.5} />
              <polygon points="360,368 356,360 364,360" fill={COLORS.wire} />

              {/* Residual Add */}
              <g transform="translate(278, 368)">
                <rect width={164} height={24} rx={3} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
                <text x={82} y={16} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>+ Residual</text>
              </g>
            </g>
          )}

          {/* Arrow: Transformer → Final RMSNorm */}
          <line x1={centerX} y1={isMoE ? 482 : 422} x2={centerX} y2={isMoE ? 498 : 438} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},${isMoE ? 506 : 446} ${centerX - 6},${isMoE ? 496 : 436} ${centerX + 6},${isMoE ? 496 : 436}`} fill={COLORS.wire} />

          {/* Final RMSNorm */}
          <g transform={`translate(${centerX - 80}, ${isMoE ? 506 : 446})`}>
            <rect width={160} height={26} rx={4} fill="#fafafa" stroke={COLORS.wire} strokeWidth={1} />
            <text x={80} y={18} textAnchor="middle" fontSize={12} fill={COLORS.textSecondary}>Final RMSNorm</text>
          </g>

          {/* Arrow: Final RMSNorm → LM Head */}
          <line x1={centerX} y1={isMoE ? 532 : 472} x2={centerX} y2={isMoE ? 548 : 488} stroke={COLORS.wire} strokeWidth={2} />
          <polygon points={`${centerX},${isMoE ? 556 : 496} ${centerX - 6},${isMoE ? 546 : 486} ${centerX + 6},${isMoE ? 546 : 486}`} fill={COLORS.wire} />

          {/* Output */}
          <g onClick={(e) => { e.stopPropagation(); setSelectedBlock('output') }} style={{ cursor: 'pointer' }}>
            <rect x={centerX - 110} y={isMoE ? 556 : 496} width={220} height={54} rx={6} {...getBlockStyle('output', COLORS.output)} />
            <text x={centerX} y={isMoE ? 580 : 518} textAnchor="middle" fontSize={15} fontWeight={600} fill={COLORS.output.text}>
              LM Head
            </text>
            <text x={centerX} y={isMoE ? 598 : 536} textAnchor="middle" fontSize={11} fill={COLORS.textSecondary}>
              [B, S, H] → [B, S, V]
            </text>
          </g>
        </svg>
      </div>

      {/* 右侧：详情面板 */}
      <div style={{ flex: '1 1 45%', minWidth: 0, padding: '0 8px' }}>
        {detailContent[selectedBlock] || detailContent.overview}
      </div>
    </div>
  )

  return (
    <div style={{ fontFamily: '"Times New Roman", Times, serif, "Microsoft YaHei", "PingFang SC", sans-serif' }}>
      {headerContent}
      {cardContent}
    </div>
  )
}

export default ModelInfoCard
