/**
 * 并行策略介绍组件
 * 参考 NVIDIA Megatron-LM 和 NeMo 的专业可视化风格
 */

import React from 'react'

export type ParallelismType = 'dp' | 'tp' | 'pp' | 'ep' | 'sp'

interface ParallelismInfoProps {
  type: ParallelismType
}

// 统一配色 - 所有策略使用相同色调
const COLORS: Record<ParallelismType, { primary: string; light: string; dark: string }> = {
  dp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  tp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  pp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  ep: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
  sp: { primary: '#1890ff', light: '#e6f7ff', dark: '#0050b3' },
}

const INFO: Record<ParallelismType, {
  name: string
  fullName: string
  shortDesc: string
  definition: string
  keyPoints: string[]
  communication: string
  bestFor: string
  steps?: string[]  // 可选的流程步骤
}> = {
  dp: {
    name: 'DP',
    fullName: 'Data Parallelism',
    shortDesc: '数据并行',
    definition: '每个 GPU 持有完整模型副本，将 Batch 切分后并行处理，通过 AllReduce 同步梯度。',
    keyPoints: ['模型完整复制', 'Batch 切分', '通信频率低'],
    communication: 'AllReduce (每 Step)',
    bestFor: '模型可放入单 GPU，需要扩展吞吐量',
    steps: [
      'Global Batch 切分到各 GPU',
      '各 GPU 独立前向计算',
      '各 GPU 独立反向计算',
      'AllReduce 同步梯度',
      '各 GPU 更新参数',
    ],
  },
  tp: {
    name: 'TP',
    fullName: 'Tensor Parallelism',
    shortDesc: '张量并行',
    definition: '将权重矩阵按列/行切分到多个 GPU，每层计算后需要 AllReduce/AllGather 同步。',
    keyPoints: ['矩阵列切分', '每层通信', '节点内高效'],
    communication: 'AllReduce (每层 2-4 次)',
    bestFor: '单层参数过大，需要 NVLink 高速互联',
    steps: [
      '输入 X 复制到各 GPU',
      '各 GPU 并行计算 X × W[i]',
      'AllReduce 同步部分结果',
      '每层重复 (Attn + FFN)',
    ],
  },
  pp: {
    name: 'PP',
    fullName: 'Pipeline Parallelism',
    shortDesc: '流水线并行',
    definition: '将模型按层分成多个 Stage，分配到不同 GPU，数据以流水线方式依次处理。',
    keyPoints: ['层间切分', '点对点通信', '存在 Bubble'],
    communication: 'P2P (每 Micro-batch)',
    bestFor: '跨节点扩展，模型层数多',
    steps: [
      '模型按层划分到多个 Stage',
      'Micro-batch 进入 Stage 0',
      'P2P 传递激活值到下一 Stage',
      '多 Micro-batch 流水线并行',
    ],
  },
  ep: {
    name: 'EP',
    fullName: 'Expert Parallelism',
    shortDesc: '专家并行',
    definition: 'MoE 专用，将专家网络分布到不同 GPU，通过 AllToAll 路由 Token 到对应专家。',
    keyPoints: ['专家分布', 'AllToAll 路由', '稀疏激活'],
    communication: 'AllToAll (每 MoE 层)',
    bestFor: 'MoE 模型，大规模专家扩展',
    steps: [
      'Router 计算 Token → Expert 分配',
      'AllToAll 发送 Token 到目标 GPU',
      '各 GPU 执行本地专家计算',
      'AllToAll 返回结果到原 GPU',
    ],
  },
  sp: {
    name: 'SP',
    fullName: 'Sequence Parallelism',
    shortDesc: '序列并行',
    definition: '与 TP 配合使用，在 LayerNorm/Dropout 处沿序列维度切分激活值，减少激活显存。',
    keyPoints: ['与 TP 配合', '激活显存 ÷TP', 'LayerNorm/Dropout'],
    communication: 'AllGather + ReduceScatter',
    bestFor: '长序列推理，激活显存受限',
    steps: [
      'LayerNorm: 输入序列切分 (S/TP)',
      'AllGather: 收集完整序列',
      'Attention/FFN: 使用 TP 计算',
      'ReduceScatter: 输出回序列切分',
      'Dropout: 继续序列切分状态',
    ],
  },
}

// ============================================
// SVG 图示 - 统一配色，简洁专业
// ============================================

// 统一色彩
const C = {
  primary: '#1890ff',
  primaryLight: '#e6f7ff',
  border: '#d9d9d9',
  text: '#262626',
  textSec: '#8c8c8c',
  bg: '#fafafa',
}

const DiagramDP: React.FC = () => (
  <svg width="360" height="195" viewBox="0 0 360 195" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
    {/* 顶部：Global Batch */}
    <rect x="80" y="8" width="200" height="28" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="180" y="27" textAnchor="middle" fontSize="13" fill={C.primary} fontWeight="600">Global Batch</text>

    {/* 分发箭头 */}
    <defs>
      <marker id="arrowDown" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill={C.border} />
      </marker>
      <marker id="arrowGreen" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#13c2c2" />
      </marker>
    </defs>
    <path d="M130 38 L75 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown)" />
    <path d="M180 38 L180 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown)" />
    <path d="M230 38 L285 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown)" />

    {/* GPU 方块 - 居中对称 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 115}, 62)`}>
        <rect width="85" height="78" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="42" y="18" textAnchor="middle" fontSize="12" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="24" x2="85" y2="24" stroke={C.border} strokeWidth="1" />
        {/* 模型 */}
        <rect x="8" y="32" width="69" height="20" rx="3" fill={C.bg} stroke={C.border} strokeWidth="1" />
        <text x="42" y="46" textAnchor="middle" fontSize="11" fill={C.textSec}>Model (完整)</text>
        {/* 数据 */}
        <rect x="8" y="56" width="69" height="20" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1" />
        <text x="42" y="70" textAnchor="middle" fontSize="11" fill={C.primary}>Data 1/{3}</text>
      </g>
    ))}

    {/* 底部：AllReduce 通信 */}
    <rect x="80" y="160" width="200" height="28" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
    <text x="180" y="179" textAnchor="middle" fontSize="13" fill="#13c2c2" fontWeight="600">AllReduce 同步梯度</text>

    {/* 连接箭头 - 梯度流向 */}
    <path d="M62 140 L125 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
    <path d="M180 140 L180 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
    <path d="M298 140 L235 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
  </svg>
)

const DiagramTP: React.FC = () => (
  <svg width="360" height="195" viewBox="0 0 360 195" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
    {/* 箭头定义 */}
    <defs>
      <marker id="arrowDownTP" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill={C.border} />
      </marker>
      <marker id="arrowGreenTP" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#13c2c2" />
      </marker>
    </defs>

    {/* 顶部：Weight Matrix */}
    <rect x="80" y="8" width="200" height="28" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="180" y="27" textAnchor="middle" fontSize="13" fill={C.primary} fontWeight="600">Weight [H × 4H]</text>

    {/* 分发箭头 */}
    <path d="M130 38 L75 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDownTP)" />
    <path d="M180 38 L180 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDownTP)" />
    <path d="M230 38 L285 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDownTP)" />

    {/* GPU 方块 - 居中对称 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 115}, 62)`}>
        <rect width="85" height="78" rx="5" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="42" y="18" textAnchor="middle" fontSize="12" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="24" x2="85" y2="24" stroke={C.border} strokeWidth="1" />
        {/* 权重分片 */}
        <rect x="10" y="32" width="65" height="38" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
        <text x="42" y="56" textAnchor="middle" fontSize="13" fill={C.primary} fontWeight="600">W[{i}]</text>
      </g>
    ))}

    {/* 底部：AllReduce 通信 */}
    <rect x="80" y="160" width="200" height="28" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
    <text x="180" y="179" textAnchor="middle" fontSize="13" fill="#13c2c2" fontWeight="600">AllReduce (每层)</text>

    {/* 连接箭头 - 结果同步 */}
    <path d="M62 140 L125 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreenTP)" />
    <path d="M180 140 L180 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreenTP)" />
    <path d="M298 140 L235 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowGreenTP)" />
  </svg>
)

const DiagramPP: React.FC = () => (
  <svg width="400" height="210" viewBox="0 0 400 210" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
    {/* 箭头定义 - 更小的箭头头部 */}
    <defs>
      <marker id="arrowP2P" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#13c2c2" />
      </marker>
    </defs>

    {/* Pipeline Stages - 居中对称，间距增大 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${15 + i * 130}, 10)`}>
        <rect width="95" height="145" rx="6" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="47" y="20" textAnchor="middle" fontSize="13" fill={C.text} fontWeight="600">Stage {i}</text>
        <line x1="0" y1="28" x2="95" y2="28" stroke={C.border} strokeWidth="1" />
        <text x="47" y="44" textAnchor="middle" fontSize="12" fill={C.textSec}>GPU {i}</text>
        {/* 层 - 带标签 */}
        {[0, 1, 2, 3].map(j => (
          <g key={j}>
            <rect x="10" y={52 + j * 22} width="75" height="18" rx="3" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
            <text x="47" y={65 + j * 22} textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="500">Layer {i * 4 + j}</text>
          </g>
        ))}
      </g>
    ))}

    {/* P2P 箭头 - Stage 间通信，位置调整避免重叠 */}
    <g transform="translate(114, 82)">
      <line x1="0" y1="0" x2="26" y2="0" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowP2P)" />
      <text x="13" y="-6" textAnchor="middle" fontSize="10" fill="#13c2c2" fontWeight="600">P2P</text>
    </g>
    <g transform="translate(244, 82)">
      <line x1="0" y1="0" x2="26" y2="0" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowP2P)" />
      <text x="13" y="-6" textAnchor="middle" fontSize="10" fill="#13c2c2" fontWeight="600">P2P</text>
    </g>

    {/* 底部：Micro-batch 流水线 */}
    <rect x="85" y="173" width="230" height="26" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
    <text x="200" y="190" textAnchor="middle" fontSize="12" fill="#13c2c2" fontWeight="600">Micro-batch 流水线执行</text>
  </svg>
)

const DiagramEP: React.FC = () => (
  <svg width="360" height="195" viewBox="0 0 360 195" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
    {/* 箭头定义 */}
    <defs>
      <marker id="arrowEP" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#13c2c2" />
      </marker>
      <marker id="arrowDown2" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill={C.border} />
      </marker>
    </defs>

    {/* 顶部：Router */}
    <rect x="100" y="8" width="160" height="28" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
    <text x="180" y="27" textAnchor="middle" fontSize="13" fill={C.primary} fontWeight="600">Router (Top-K)</text>

    {/* 分发箭头 */}
    <path d="M140 38 L75 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown2)" />
    <path d="M180 38 L180 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown2)" />
    <path d="M220 38 L285 55" stroke={C.border} strokeWidth="1.5" markerEnd="url(#arrowDown2)" />

    {/* GPU + Experts - 居中对称 */}
    {[0, 1, 2].map(i => (
      <g key={i} transform={`translate(${20 + i * 115}, 62)`}>
        <rect width="85" height="78" rx="6" fill="#fff" stroke={C.border} strokeWidth="1.5" />
        <text x="42" y="18" textAnchor="middle" fontSize="12" fill={C.text} fontWeight="600">GPU {i}</text>
        <line x1="0" y1="24" x2="85" y2="24" stroke={C.border} strokeWidth="1" />
        {/* 专家 */}
        <rect x="8" y="32" width="33" height="40" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
        <text x="24" y="56" textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="600">E{i*2}</text>
        <rect x="44" y="32" width="33" height="40" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
        <text x="60" y="56" textAnchor="middle" fontSize="12" fill={C.primary} fontWeight="600">E{i*2+1}</text>
      </g>
    ))}

    {/* 底部：AllToAll 通信 */}
    <rect x="80" y="160" width="200" height="28" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
    <text x="180" y="179" textAnchor="middle" fontSize="13" fill="#13c2c2" fontWeight="600">AllToAll 路由</text>

    {/* 连接箭头 - 双向 AllToAll */}
    <path d="M62 140 L125 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowEP)" />
    <path d="M180 140 L180 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowEP)" />
    <path d="M298 140 L235 157" stroke="#13c2c2" strokeWidth="1.5" markerEnd="url(#arrowEP)" />
  </svg>
)

const DiagramSP: React.FC = () => (
  <svg width="420" height="170" viewBox="0 0 420 170" style={{ fontFamily: '"Times New Roman", Times, serif' }}>
    {/* 左侧：LayerNorm (序列切分) */}
    <g transform="translate(10, 10)">
      <rect width="90" height="150" rx="6" fill="#fff" stroke={C.border} strokeWidth="1.5" />
      <text x="45" y="20" textAnchor="middle" fontSize="13" fill={C.text} fontWeight="600">LayerNorm</text>
      <line x1="0" y1="28" x2="90" y2="28" stroke={C.border} strokeWidth="1" />
      {/* 序列切分 - 3个GPU各持有1/3 */}
      {[0, 1, 2].map(i => (
        <g key={i}>
          <rect x="10" y={38 + i * 36} width="70" height="30" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
          <text x="45" y={57 + i * 36} textAnchor="middle" fontSize="11" fill={C.primary} fontWeight="500">S/TP @ GPU{i}</text>
        </g>
      ))}
    </g>

    {/* AllGather 箭头 */}
    <g transform="translate(108, 75)">
      <line x1="0" y1="10" x2="50" y2="10" stroke={C.primary} strokeWidth="2" />
      <polygon points="50,10 43,4 43,16" fill={C.primary} />
      <text x="25" y="-2" textAnchor="middle" fontSize="11" fill={C.primary} fontWeight="600">AllGather</text>
    </g>

    {/* 中间：Attention (完整序列 + TP权重) */}
    <g transform="translate(165, 20)">
      <rect width="80" height="130" rx="6" fill="#fff" stroke={C.primary} strokeWidth="2" />
      <text x="40" y="20" textAnchor="middle" fontSize="13" fill={C.primary} fontWeight="600">Attention</text>
      <line x1="0" y1="28" x2="80" y2="28" stroke={C.border} strokeWidth="1" />
      {/* 完整序列 */}
      <rect x="8" y="38" width="64" height="26" rx="4" fill={C.primaryLight} stroke={C.primary} strokeWidth="1.5" />
      <text x="40" y="55" textAnchor="middle" fontSize="11" fill={C.primary} fontWeight="500">Full Seq</text>
      {/* TP 权重 */}
      <rect x="8" y="70" width="64" height="26" rx="4" fill={C.bg} stroke={C.border} strokeWidth="1.5" />
      <text x="40" y="87" textAnchor="middle" fontSize="11" fill={C.textSec}>W (TP切分)</text>
      {/* 输出 */}
      <rect x="8" y="102" width="64" height="20" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
      <text x="40" y="116" textAnchor="middle" fontSize="10" fill="#13c2c2" fontWeight="500">Output</text>
    </g>

    {/* ReduceScatter 箭头 */}
    <g transform="translate(253, 75)">
      <line x1="0" y1="10" x2="60" y2="10" stroke="#13c2c2" strokeWidth="2" />
      <polygon points="60,10 53,4 53,16" fill="#13c2c2" />
      <text x="30" y="-2" textAnchor="middle" fontSize="11" fill="#13c2c2" fontWeight="600">ReduceScatter</text>
    </g>

    {/* 右侧：Dropout (序列切分) */}
    <g transform="translate(320, 10)">
      <rect width="90" height="150" rx="6" fill="#fff" stroke={C.border} strokeWidth="1.5" />
      <text x="45" y="20" textAnchor="middle" fontSize="13" fill={C.text} fontWeight="600">Dropout</text>
      <line x1="0" y1="28" x2="90" y2="28" stroke={C.border} strokeWidth="1" />
      {/* 序列切分 */}
      {[0, 1, 2].map(i => (
        <g key={i}>
          <rect x="10" y={38 + i * 36} width="70" height="30" rx="4" fill="#e6fffb" stroke="#13c2c2" strokeWidth="1.5" />
          <text x="45" y={57 + i * 36} textAnchor="middle" fontSize="11" fill="#13c2c2" fontWeight="500">S/TP @ GPU{i}</text>
        </g>
      ))}
    </g>
  </svg>
)

const DIAGRAMS: Record<ParallelismType, React.FC> = {
  dp: DiagramDP,
  tp: DiagramTP,
  pp: DiagramPP,
  ep: DiagramEP,
  sp: DiagramSP,
}

// ============================================
// 主组件 - 简洁的左右布局
// ============================================

export const ParallelismInfo: React.FC<ParallelismInfoProps> = ({ type }) => {
  const info = INFO[type]
  const color = COLORS[type]
  const Diagram = DIAGRAMS[type]

  return (
    <div style={{
      display: 'flex',
      gap: 200,
      padding: 12,
    }}>
      {/* 左侧：图示 */}
      <div style={{
        flexShrink: 0,
        background: '#fff',
        borderRadius: 6,
        padding: 15,
        marginLeft: 120,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <Diagram />
      </div>

      {/* 右侧：说明 */}
      <div style={{ flex: '0 1 auto', minWidth: 0, maxWidth: 400 }}>
        {/* 标题 */}
        <div style={{ marginBottom: 10 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: color.dark }}>
            {info.fullName}
          </span>
          <span style={{ fontSize: 13, color: '#8c8c8c', marginLeft: 8 }}>
            {info.shortDesc}
          </span>
        </div>

        {/* 定义 */}
        <div style={{ fontSize: 13, color: '#595959', lineHeight: 1.7, marginBottom: 12 }}>
          {info.definition}
        </div>

        {/* 关键特点 - 横向标签 */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {info.keyPoints.map((point, i) => (
            <span key={i} style={{
              padding: '3px 10px',
              background: '#fff',
              border: `1px solid ${color.primary}66`,
              borderRadius: 12,
              fontSize: 12,
              color: color.dark,
            }}>
              {point}
            </span>
          ))}
        </div>

        {/* 流程步骤 */}
        {info.steps && (
          <div style={{ marginBottom: 12, fontSize: 12, lineHeight: 1.9 }}>
            <span style={{ color: '#8c8c8c' }}>流程: </span>
            <div style={{ marginTop: 6, paddingLeft: 24 }}>
              {info.steps.map((step, i) => (
                <div key={i} style={{ color: '#595959' }}>
                  <span style={{ color: color.primary, fontWeight: 600 }}>{'①②③④⑤'[i]} </span>
                  {step}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 通信和适用场景（SP 有流程时不显示） */}
        {!info.steps && (
          <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
            <div>
              <span style={{ color: '#8c8c8c' }}>通信: </span>
              <span style={{ color: '#262626', fontWeight: 500 }}>{info.communication}</span>
            </div>
            <div style={{ flex: 1 }}>
              <span style={{ color: '#8c8c8c' }}>适用: </span>
              <span style={{ color: '#262626' }}>{info.bestFor}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================
// 并行策略卡片组件
// ============================================

interface ParallelismCardProps {
  type: ParallelismType
  value: number
  selected: boolean
  onClick: () => void
}

export const ParallelismCard: React.FC<ParallelismCardProps> = ({ type, value, selected, onClick }) => {
  const info = INFO[type]
  const color = COLORS[type]

  return (
    <div
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: 90,
        padding: '8px 12px',
        background: selected ? color.light : '#fff',
        border: `1.5px solid ${selected ? color.primary : '#e8e8e8'}`,
        borderRadius: 8,
        cursor: 'pointer',
        transition: 'all 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontFamily: '"Times New Roman", Times, serif',
      }}
    >
      <div style={{
        fontSize: 28,
        fontWeight: 700,
        color: selected ? color.primary : '#262626',
        lineHeight: 1,
        minWidth: 32,
        textAlign: 'center',
      }}>
        {value}
      </div>
      <div style={{ flex: 1 }}>
        <div style={{
          fontSize: 16,
          fontWeight: 700,
          color: selected ? color.primary : '#262626',
          lineHeight: 1.2,
        }}>
          {info.name}
        </div>
        <div style={{
          fontSize: 11,
          color: selected ? color.primary : '#8c8c8c',
          marginTop: 2,
        }}>
          {info.shortDesc}
        </div>
      </div>
    </div>
  )
}

export default ParallelismInfo
