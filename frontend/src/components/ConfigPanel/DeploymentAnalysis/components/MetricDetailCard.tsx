/**
 * 指标详情卡片组件
 * 参考 Notion 的简洁设计风格
 * 每个公式参数都有详细说明
 */

import React from 'react'
import { Typography } from 'antd'
import {
  FormulaCard,
  VariableList,
  CalculationSteps,
} from './FormulaDisplay'
import { PlanAnalysisResult } from '../../../../utils/llmDeployment/types'

const { Text } = Typography

export type MetricType = 'ttft' | 'tpot' | 'throughput' | 'tps_batch' | 'tps_chip' | 'mfu' | 'mbu' | 'cost' | 'percentiles' | 'bottleneck' | 'e2e' | 'chips' | 'memory'

interface MetricDetailCardProps {
  metric: MetricType
  result: PlanAnalysisResult
}

// 内嵌详情区域样式
const detailWrapperStyle: React.CSSProperties = {
  background: '#fafafa',
  borderRadius: 8,
  padding: 16,
}

// 小节标题样式
const sectionTitleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: '#374151',
  marginBottom: 10,
}

// 说明文字样式
const descStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  lineHeight: 1.6,
}

export const MetricDetailCard: React.FC<MetricDetailCardProps> = ({ metric, result }) => {
  const { plan, memory, latency, throughput } = result

  switch (metric) {
    case 'ttft':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff', marginBottom: 12 }}>
            First Token Latency (FTL)
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              首Token延迟，即从请求发送到生成第一个输出Token的时间。
              对应Prefill阶段，处理全部输入序列。MLPerf要求P99 ≤ 450ms。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{FTL} = \frac{T_{\text{compute}} + T_{\text{comm}}}{1 - \beta}`}
            result={latency.prefill_total_latency_ms.toFixed(2)}
            unit="ms"
            resultColor="#1890ff"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{FLOPs}_{\\text{prefill}}',
                name: 'Prefill计算量',
                description: '线性部分：$2 \\times (B \\times S) \\times P_{active}$；Attention部分：$O(S^2)$',
              },
              {
                symbol: 'P_{\\text{active}}',
                name: '激活参数量',
                description: 'MoE模型实际参与计算的参数，如DeepSeek-V3约37B（总参671B）',
              },
              {
                symbol: '\\text{Peak}',
                name: '峰值算力',
                description: '单芯片理论峰值 × TP，如H100 SXM = 989 TFLOPs (BF16)',
              },
              {
                symbol: '\\text{MFU}',
                name: '硬件利用率',
                description: 'Model FLOPs Utilization，Prefill阶段通常可达50-60%',
              },
              {
                symbol: 'T_{\\text{comm}}',
                name: '通信延迟',
                description: 'TP AllReduce：$2 \\times L \\times (B \\times S) \\times H \\times dtype / BW$',
              },
              {
                symbol: '\\beta',
                name: '气泡比',
                description: 'PP导致的空闲时间占比，$\\beta = \\frac{PP-1}{MB+PP-1}$（GPipe调度）',
              },
              {
                symbol: '\\text{TP}',
                name: '张量并行度',
                description: '单层内切分设备数，减少单卡计算量但增加AllReduce通信',
              },
              {
                symbol: '\\text{PP}',
                name: '流水线并行度',
                description: '层间切分阶段数，PP=1时无气泡开销',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '\\text{FLOPs}_{\\text{prefill}}',
                formula: '\\text{FLOPs}_{\\text{prefill}} = 2 \\times (B \\times S) \\times P_{active} + O(S^2)',
                value: latency.prefill_flops ? (latency.prefill_flops / 1e12).toFixed(1) : '-',
                unit: 'TFLOPs',
              },
              {
                label: 'T_{\\text{compute}}',
                formula: 'T_{\\text{compute}} = \\frac{\\text{FLOPs}_{\\text{prefill}}}{\\text{Peak} \\times \\text{MFU}}',
                value: latency.prefill_compute_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: 'T_{\\text{comm}}',
                formula: 'T_{\\text{comm}} = 2 \\times L \\times \\frac{(B \\times S) \\times H \\times dtype}{BW}',
                value: latency.prefill_comm_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: '\\beta',
                formula: '\\beta = \\frac{PP - 1}{MB + PP - 1}',
                value: (latency.pipeline_bubble_ratio * 100).toFixed(1),
                unit: '%',
              },
            ]}
          />
        </div>
      )

    case 'tpot':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff', marginBottom: 12 }}>
            Time Per Output Token (TPOT)
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              单Token延迟，即Decode阶段生成每个输出Token的时间。
              是memory-bound，瓶颈在显存带宽。MLPerf要求P99 ≤ 40ms。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{TPOT} = \max(T_{\text{compute}}, T_{\text{memory}}) + T_{\text{comm}}`}
            result={latency.decode_per_token_latency_ms.toFixed(3)}
            unit="ms"
            resultColor="#13c2c2"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: 'T_{\\text{compute}}',
                name: '计算延迟',
                description: '单token前向传播计算时间',
              },
              {
                symbol: 'T_{\\text{memory}}',
                name: '访存延迟',
                description: '每token需读取全部权重，Decode阶段瓶颈',
              },
              {
                symbol: 'T_{\\text{comm}}',
                name: '通信延迟',
                description: 'TP AllReduce通信开销',
              },
              {
                symbol: '\\text{FLOPs}_{\\text{decode}}',
                name: '每Token计算量',
                description: 'Decode阶段单token计算量',
              },
              {
                symbol: 'P',
                name: '参数量',
                description: '模型总参数数量',
              },
              {
                symbol: 'M_{\\text{model}}',
                name: '模型显存',
                description: '模型权重占用的显存大小（GB）',
              },
              {
                symbol: 'M_{\\text{KV}}',
                name: 'KV缓存显存',
                description: 'Key/Value占用的显存（GB）',
              },
              {
                symbol: 'B',
                name: '批次大小',
                description: '同时处理的请求数量',
              },
              {
                symbol: 'H',
                name: '隐藏维度',
                description: '模型隐藏层维度',
              },
              {
                symbol: '\\text{BW}',
                name: '带宽',
                description: '访存用HBM带宽，通信用Link带宽（来自拓扑配置）',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '\\text{FLOPs}_{\\text{decode}}',
                formula: '\\text{FLOPs}_{\\text{decode}} \\approx 2 \\times P',
                value: (2 * 70e9 / 1e9).toFixed(0),
                unit: 'GFLOPs',
              },
              {
                label: 'T_{\\text{compute}}',
                formula: 'T_{\\text{compute}} = \\frac{\\text{FLOPs}_{\\text{decode}}}{\\text{Peak} \\times \\text{TP}}',
                value: latency.decode_compute_latency_ms.toFixed(3),
                unit: 'ms',
              },
              {
                label: 'T_{\\text{memory}}',
                formula: 'T_{\\text{memory}} = \\frac{M_{\\text{model}} + M_{\\text{KV}}}{\\text{BW}}',
                value: (memory.model_memory_gb / 3.35).toFixed(2),
                unit: 'ms',
              },
              {
                label: 'T_{\\text{comm}}',
                formula: 'T_{\\text{comm}} = \\frac{2 \\times B \\times H}{\\text{BW}}',
                value: latency.decode_comm_latency_ms.toFixed(3),
                unit: 'ms',
              },
            ]}
          />
        </div>
      )

    case 'throughput':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#52c41a', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Total TPS</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>集群总吞吐 · 系统整体处理能力</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              集群每秒生成的Token总数，衡量系统整体处理能力。
              Total TPS = TPS per Chip × 芯片数。是容量规划和成本计算的基础。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{Total TPS} = \text{TPS}_{\text{chip}} \times N_{\text{chips}}`}
            description="集群总吞吐 = 单芯片吞吐 × 芯片数"
            result={throughput.tokens_per_second.toFixed(0)}
            unit="tok/s"
            resultColor="#52c41a"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{TPS}_{\\text{chip}}',
                name: '单芯片吞吐',
                description: '每芯片每秒生成的token数，$= B \\times \\text{TPS}_{\\text{batch}}$',
              },
              {
                symbol: 'N_{\\text{chips}}',
                name: '芯片数',
                description: '$= \\text{DP} \\times \\text{TP} \\times \\text{PP} \\times \\text{EP}$',
              },
              {
                symbol: '\\text{TPS}_{\\text{batch}}',
                name: '单Batch吞吐',
                description: '$= 1000 / \\text{TPOT}(ms)$，用户体验指标',
              },
              {
                symbol: 'B',
                name: '批次大小',
                description: '同时处理的请求数量',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: 'TPS_{\\text{batch}}',
                formula: '\\text{TPS}_{\\text{batch}} = \\frac{1000}{\\text{TPOT}(ms)}',
                value: throughput.tps_per_batch.toFixed(1),
                unit: 'tok/s',
              },
              {
                label: 'TPS_{\\text{chip}}',
                formula: '\\text{TPS}_{\\text{chip}} = B \\times \\text{TPS}_{\\text{batch}}',
                value: throughput.tps_per_chip.toFixed(0),
                unit: 'tok/s',
              },
              {
                label: 'N_{\\text{chips}}',
                formula: 'N_{\\text{chips}} = \\text{DP} \\times \\text{TP} \\times \\text{PP} \\times \\text{EP}',
                value: plan.total_chips.toString(),
                unit: 'chips',
              },
            ]}
          />
        </div>
      )

    case 'tps_batch':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>TPS per Batch</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>用户体验指标 · SLO约束 ≥10</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              单Batch每秒生成的Token数，是用户体验的核心指标。
              TPS per Batch = 1 / DecodeTime(s)。SLO要求 ≥10，即 DecodeTime ≤ 100ms。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{TPS}_{\text{batch}} = \frac{1000}{\text{TPOT}(ms)} = \frac{1}{\text{DecodeTime}(s)}`}
            description="单Batch吞吐 = 1000 / 单Token延迟(ms)"
            result={throughput.tps_per_batch.toFixed(1)}
            unit="tok/s"
            resultColor={throughput.tps_per_batch >= 10 ? '#52c41a' : '#f5222d'}
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{TPOT}',
                name: '单Token延迟',
                description: 'Time Per Output Token，Decode阶段每token生成时间',
              },
              {
                symbol: '\\text{DecodeTime}',
                name: 'Decode时间',
                description: '与TPOT相同，单位为秒',
              },
              {
                symbol: '\\text{SLO}',
                name: '服务质量约束',
                description: 'TPS per Batch ≥ 10，保证用户体验',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '\\text{TPOT}',
                formula: '\\text{TPOT} = \\max(T_{\\text{comp}}, T_{\\text{mem}}) + T_{\\text{comm}}',
                value: latency.decode_per_token_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: '\\text{TPS}_{\\text{batch}}',
                formula: '\\text{TPS}_{\\text{batch}} = \\frac{1000}{\\text{TPOT}}',
                value: throughput.tps_per_batch.toFixed(1),
                unit: 'tok/s',
              },
            ]}
          />

          <div style={{
            marginTop: 16,
            padding: '10px 14px',
            background: throughput.tps_per_batch >= 10 ? '#f6ffed' : '#fff2f0',
            borderRadius: 8,
            fontSize: 13,
            color: throughput.tps_per_batch >= 10 ? '#52c41a' : '#f5222d',
            textAlign: 'center',
            border: `1px solid ${throughput.tps_per_batch >= 10 ? '#b7eb8f' : '#ffa39e'}`,
          }}>
            {throughput.tps_per_batch >= 10 ? '✓ 满足SLO约束' : '⚠ 不满足SLO约束'} ·
            TPS/Batch = <strong>{throughput.tps_per_batch.toFixed(1)}</strong> tok/s ·
            要求 ≥ 10 tok/s
          </div>
        </div>
      )

    case 'tps_chip':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#fa8c16', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>TPS per Chip</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>成本效益指标 · 优化目标</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              单芯片每秒生成的Token数，是成本效益的核心指标，也是优化的主要目标。
              TPS per Chip = Batch × TPS per Batch。在满足SLO的前提下，最大化此指标。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{TPS}_{\text{chip}} = B \times \text{TPS}_{\text{batch}} = \frac{B}{\text{DecodeTime}(s)}`}
            description="单芯片吞吐 = 批次大小 × 单Batch吞吐"
            result={throughput.tps_per_chip.toFixed(0)}
            unit="tok/s"
            resultColor="#fa8c16"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: 'B',
                name: '批次大小',
                description: '同时处理的请求数量，增大B可提高TPS per Chip',
              },
              {
                symbol: '\\text{TPS}_{\\text{batch}}',
                name: '单Batch吞吐',
                description: '$= 1000 / \\text{TPOT}(ms)$，受SLO约束',
              },
              {
                symbol: '\\text{DecodeTime}',
                name: 'Decode时间',
                description: '单token生成时间，与TPOT相同',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: 'B',
                formula: '\\text{Batch Size} = \\frac{\\text{TPS}_{\\text{chip}}}{\\text{TPS}_{\\text{batch}}}',
                value: Math.round(throughput.tps_per_chip / throughput.tps_per_batch).toString(),
                unit: '',
              },
              {
                label: '\\text{TPS}_{\\text{batch}}',
                formula: '\\frac{1000}{\\text{TPOT}(ms)}',
                value: throughput.tps_per_batch.toFixed(1),
                unit: 'tok/s',
              },
              {
                label: '\\text{TPS}_{\\text{chip}}',
                formula: 'B \\times \\text{TPS}_{\\text{batch}}',
                value: throughput.tps_per_chip.toFixed(0),
                unit: 'tok/s',
              },
            ]}
          />

          <div style={{
            marginTop: 16,
            padding: '10px 14px',
            background: '#fff7e6',
            borderRadius: 8,
            fontSize: 12,
            color: '#ad6800',
            textAlign: 'center',
          }}>
            💡 优化目标: 在满足 TPS/Batch ≥ 10 的前提下，最大化 TPS/Chip
          </div>
        </div>
      )

    case 'mfu':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#52c41a', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Model FLOPs Utilization (MFU)</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>算力利用率 · Prefill效率指标</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              实际用于模型计算的算力占硬件峰值算力的比例。MFU越高说明硬件利用越充分。
              Prefill阶段是compute-bound，MFU是衡量其效率的关键指标。
              参考值：Prefill 40-60%（优秀），Decode 20-40%（正常，因memory-bound）。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{MFU} = \frac{\text{Achieved FLOPs}}{\text{Peak FLOPs}} \times 100\%`}
            description="实际算力 / 理论峰值算力"
            result={(throughput.model_flops_utilization * 100).toFixed(2)}
            unit="%"
            resultColor="#faad14"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Achieved}',
                name: '实际算力',
                description: 'Throughput × FLOPs per Token',
              },
              {
                symbol: '\\text{Peak}',
                name: '峰值算力',
                description: '芯片数 × 单芯片峰值算力',
              },
              {
                symbol: '\\text{Throughput}',
                name: '吞吐量',
                description: '每秒生成的token数',
              },
              {
                symbol: '\\text{FLOPs/Token}',
                name: '每Token计算量',
                description: '$\\approx 2 \\times$ 模型参数量',
              },
              {
                symbol: 'N_{\\text{chips}}',
                name: '芯片数',
                description: '部署使用的芯片总数',
              },
              {
                symbol: '\\text{Chip TFLOPs}',
                name: '单芯片算力',
                description: '单个芯片的理论峰值算力',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '实际算力',
                formula: '\\text{Achieved} = \\text{Throughput} \\times \\text{FLOPs/Token}',
                value: (throughput.tokens_per_second * 2 * 70e9 / 1e12).toFixed(2),
                unit: 'TFLOPs',
              },
              {
                label: '峰值算力',
                formula: '\\text{Peak} = N_{\\text{chips}} \\times \\text{Chip TFLOPs}',
                value: `${plan.total_chips} × Peak`,
              },
            ]}
          />
        </div>
      )

    case 'mbu':
      const achievedBW = (memory.model_memory_gb + memory.kv_cache_memory_gb * 0.5) / (latency.decode_per_token_latency_ms / 1000)
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#52c41a', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Memory Bandwidth Utilization (MBU)</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>带宽利用率 · Decode效率指标</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              实际显存带宽使用量占峰值带宽的比例。Decode阶段是memory-bound，
              MBU是衡量其效率的核心指标。MBU越高，TPOT越接近理论极限。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{MBU} = \frac{\text{Achieved BW}}{\text{Peak BW}} \times 100\%`}
            description="实际带宽利用 / 硬件峰值带宽"
            result={(throughput.memory_bandwidth_utilization * 100).toFixed(1)}
            unit="%"
            resultColor="#722ed1"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Achieved BW}',
                name: '实际带宽',
                description: '(模型大小 + KV Cache) / TPOT',
              },
              {
                symbol: '\\text{Peak BW}',
                name: '峰值带宽',
                description: '芯片HBM带宽，如H100 = 3.35 TB/s',
              },
              {
                symbol: '\\text{Model}',
                name: '模型大小',
                description: '模型权重占用的显存',
              },
              {
                symbol: '\\text{KV Cache}',
                name: 'KV缓存',
                description: '存储历史token的Key/Value',
              },
              {
                symbol: '\\text{TPOT}',
                name: '每Token延迟',
                description: 'Decode阶段单token生成时间',
              },
              {
                symbol: '\\text{Data}',
                name: '数据量',
                description: '每token需要读取的总数据量',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '每Token数据量',
                formula: '\\text{Data} = \\text{Model} + \\text{KV Cache}',
                value: `${memory.model_memory_gb.toFixed(2)} + ${memory.kv_cache_memory_gb.toFixed(2)}`,
                unit: 'GB',
              },
              {
                label: '实际带宽',
                formula: '\\text{Achieved BW} = \\frac{\\text{Data}}{\\text{TPOT}}',
                value: achievedBW.toFixed(0),
                unit: 'GB/s',
              },
            ]}
          />
        </div>
      )

    case 'cost':
      const costData = result.cost
      if (!costData) return null
      // 计算每小时处理的token数
      const tokensPerHour = throughput.tokens_per_second * 3600
      // 计算输出/输入成本比
      const outputInputRatio = costData.input_cost_per_million_tokens > 0
        ? (costData.output_cost_per_million_tokens / costData.input_cost_per_million_tokens).toFixed(1)
        : '-'
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#fa8c16', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Cost Analysis (成本分析)</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>经济性指标 · $/M tokens</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              每百万Token的推理成本，是衡量部署经济性的核心指标。
              成本 = 硬件租用成本 / 吞吐量。输出成本通常是输入成本的3-5倍，
              因为Decode阶段每token需要完整的前向传播，而Prefill可以批量处理。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`\text{Cost}_{\text{/M}} = \frac{\text{Price}_{\text{chip}} \times N_{\text{chips}} \times 10^6}{\text{TPS}_{\text{total}} \times 3600}`}
            description="(单芯片价格 × 芯片数 × 100万) / (总TPS × 3600)"
            result={`$${costData.cost_per_million_tokens.toFixed(4)}`}
            unit="/M tokens"
            resultColor="#fa541c"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{Price}_{\\text{chip}}',
                name: '单芯片租用价格',
                description: `云服务商每小时租用价格，当前 $${costData.hardware_cost_per_hour.toFixed(2)}/h`,
              },
              {
                symbol: 'N_{\\text{chips}}',
                name: '芯片数量',
                description: `$= \\text{DP} \\times \\text{TP} \\times \\text{PP} \\times \\text{EP} = ${plan.total_chips}$`,
              },
              {
                symbol: '\\text{TPS}_{\\text{total}}',
                name: '集群总吞吐',
                description: `$= \\text{TPS}_{\\text{chip}} \\times N_{\\text{chips}} = ${throughput.tokens_per_second.toFixed(0)}$ tok/s`,
              },
              {
                symbol: '\\text{Cost}_{\\text{input}}',
                name: '输入成本',
                description: 'Prefill阶段成本，批量处理效率高',
              },
              {
                symbol: '\\text{Cost}_{\\text{output}}',
                name: '输出成本',
                description: 'Decode阶段成本，逐token生成，通常是输入的3-5倍',
              },
            ]}
          />

          <CalculationSteps
            title="计算分解"
            steps={[
              {
                label: '总硬件成本',
                formula: '\\text{Cost}_{\\text{hw}} = \\text{Price}_{\\text{chip}} \\times N_{\\text{chips}}',
                value: `$${costData.hardware_cost_per_hour.toFixed(2)} × ${plan.total_chips}`,
                unit: `= $${costData.total_hardware_cost_per_hour.toFixed(2)}/h`,
              },
              {
                label: '每小时Token数',
                formula: '\\text{Tokens/h} = \\text{TPS}_{\\text{total}} \\times 3600',
                value: tokensPerHour.toExponential(2),
                unit: 'tokens',
              },
              {
                label: '每Token成本',
                formula: '\\text{Cost}_{\\text{/tok}} = \\frac{\\text{Cost}_{\\text{hw}}}{\\text{Tokens/h}}',
                value: (costData.total_hardware_cost_per_hour / tokensPerHour * 1e6).toFixed(4),
                unit: '$/M tok',
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 16 }}>
            <div style={{
              padding: '14px 12px',
              background: '#fff7e6',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#ad6800', marginBottom: 4 }}>综合成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#fa541c' }}>
                ${costData.cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#ad6800' }}>/M tokens</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#f6ffed',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#389e0d', marginBottom: 4 }}>输入成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>
                ${costData.input_cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#389e0d' }}>/M tokens</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#fff1f0',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#cf1322', marginBottom: 4 }}>输出成本</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f5222d' }}>
                ${costData.output_cost_per_million_tokens.toFixed(4)}
              </div>
              <div style={{ fontSize: 10, color: '#cf1322' }}>/M tokens</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 10 }}>
            <div style={{
              padding: '10px 14px',
              background: '#f5f5f5',
              borderRadius: 8,
              fontSize: 12,
              color: '#1f2937',
              textAlign: 'center',
            }}>
              效率: <strong style={{ color: '#fa541c' }}>{costData.tokens_per_dollar.toExponential(2)}</strong> tokens/$
            </div>
            <div style={{
              padding: '10px 14px',
              background: '#f0f5ff',
              borderRadius: 8,
              fontSize: 12,
              color: '#2f54eb',
              textAlign: 'center',
            }}>
              输出/输入比: <strong>{outputInputRatio}×</strong>
            </div>
          </div>

          <div style={{
            marginTop: 12,
            padding: '10px 14px',
            background: '#fffbe6',
            borderRadius: 8,
            fontSize: 12,
            color: '#ad6800',
          }}>
            💡 <strong>优化建议</strong>：在满足SLO（TPS/Batch ≥ 10）的前提下，
            增大Batch Size可提高TPS/Chip，从而降低单位成本。
          </div>
        </div>
      )

    case 'percentiles':
      const ttftP = latency.ttft_percentiles
      const tpotP = latency.tpot_percentiles
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff', marginBottom: 12 }}>
            Latency Percentiles (延迟分位数)
          </div>

          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              延迟的统计分布，P99表示99%请求延迟低于此值。
              MLPerf要求：FTL P99 ≤ 450ms，TPOT P99 ≤ 40ms。
            </div>
          </div>

          <VariableList
            title="分位数说明"
            variables={[
              {
                symbol: 'P_{50}',
                name: '中位数',
                description: '50%请求低于此延迟，代表典型用户体验',
              },
              {
                symbol: 'P_{90}',
                name: '90分位',
                description: '90%请求低于此延迟，包含大部分用户',
              },
              {
                symbol: 'P_{99}',
                name: '99分位（尾部延迟）',
                description: '99%请求低于此延迟，SLO的关键指标',
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            {/* TTFT 分位数 */}
            <div style={{ padding: 16, background: '#f0f5ff', borderRadius: 10 }}>
              <Text strong style={{ fontSize: 14, color: '#2f54eb', display: 'block', marginBottom: 12 }}>
                TTFT 分位数
              </Text>
              {ttftP && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P50</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{ttftP.p50.toFixed(1)} ms</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P90</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{ttftP.p90.toFixed(1)} ms</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: ttftP.p99 > 450 ? '#fff2f0' : '#f6ffed',
                    borderRadius: 6,
                    border: `1px solid ${ttftP.p99 > 450 ? '#ffa39e' : '#b7eb8f'}`,
                  }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P99</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: ttftP.p99 > 450 ? '#f5222d' : '#52c41a' }}>
                      {ttftP.p99.toFixed(1)} ms
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* TPOT 分位数 */}
            <div style={{ padding: 16, background: '#e6fffb', borderRadius: 10 }}>
              <Text strong style={{ fontSize: 14, color: '#13c2c2', display: 'block', marginBottom: 12 }}>
                TPOT 分位数
              </Text>
              {tpotP && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P50</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{tpotP.p50.toFixed(2)} ms</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: '#fff', borderRadius: 6 }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P90</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>{tpotP.p90.toFixed(2)} ms</span>
                  </div>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: tpotP.p99 > 40 ? '#fff2f0' : '#f6ffed',
                    borderRadius: 6,
                    border: `1px solid ${tpotP.p99 > 40 ? '#ffa39e' : '#b7eb8f'}`,
                  }}>
                    <span style={{ fontSize: 12, color: '#6b7280' }}>P99</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: tpotP.p99 > 40 ? '#f5222d' : '#52c41a' }}>
                      {tpotP.p99.toFixed(2)} ms
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

            <div style={{
              marginTop: 16,
              padding: '10px 14px',
              background: '#f0f5ff',
              borderRadius: 8,
              fontSize: 12,
              color: '#2f54eb',
              textAlign: 'center',
            }}>
              📊 MLPerf SLO标准: FTL P99 ≤ 450ms, TPOT P99 ≤ 40ms
            </div>
        </div>
      )

    case 'bottleneck':
      const bottleneckInfo: Record<string, { name: string; color: string; desc: string; solution: string }> = {
        compute: {
          name: '计算瓶颈',
          color: '#faad14',
          desc: '算力不足，GPU计算单元成为限制因素',
          solution: '增加TP并行度，或使用更强算力的芯片',
        },
        memory: {
          name: '访存瓶颈',
          color: '#1890ff',
          desc: '显存带宽不足，数据读取速度限制了计算',
          solution: '减小batch size，或使用更高带宽的芯片',
        },
        communication: {
          name: '通信瓶颈',
          color: '#722ed1',
          desc: '芯片间通信延迟过高，集合通信成为限制因素',
          solution: '减小TP/PP并行度，或使用更高带宽的互联',
        },
        pipeline_bubble: {
          name: '流水线气泡',
          color: '#13c2c2',
          desc: '流水线并行导致的空闲时间过长',
          solution: '增加micro-batch数量，或减小PP并行度',
        },
      }
      const info = bottleneckInfo[latency.bottleneck_type] || { name: '未知', color: '#666', desc: '', solution: '' }

      return (
        <div style={{ ...detailWrapperStyle, background: '#fffbe6' }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: '#fa8c16', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>⚠️ 性能瓶颈分析</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>{info.name}</span>
          </div>
          <div style={{ ...descStyle, background: '#fff', padding: 12, borderRadius: 8 }}>
            <div style={{ marginBottom: 8 }}>
              <strong style={{ color: '#ad6800' }}>瓶颈原因：</strong>
              {info.desc}
            </div>
            <div>
              <strong style={{ color: '#ad6800' }}>详细信息：</strong>
              {latency.bottleneck_details}
            </div>
          </div>

          <div style={{
            padding: '12px 16px',
            background: '#fff',
            borderRadius: 8,
            marginBottom: 16,
            borderLeft: `4px solid ${info.color}`,
          }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>优化建议</div>
            <div style={{ fontSize: 14, color: info.color, fontWeight: 500 }}>{info.solution}</div>
          </div>

            <CalculationSteps
              title="延迟分解"
              steps={[
                { label: 'Prefill 计算', value: latency.prefill_compute_latency_ms.toFixed(2), unit: 'ms' },
                { label: 'Prefill 通信', value: latency.prefill_comm_latency_ms.toFixed(2), unit: 'ms' },
                { label: 'Decode 计算', value: latency.decode_compute_latency_ms.toFixed(3), unit: 'ms' },
                { label: 'Decode 通信', value: latency.decode_comm_latency_ms.toFixed(3), unit: 'ms' },
                { label: '流水线气泡比', value: (latency.pipeline_bubble_ratio * 100).toFixed(1), unit: '%' },
              ]}
            />
        </div>
      )

    case 'e2e':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff', marginBottom: 12 }}>
            End-to-End Latency (E2E)
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              端到端延迟，即从发送请求到接收完整响应的总时间。
              E2E = FTL + TPOT × 输出Token数。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`T_{\text{e2e}} = \text{FTL} + \text{TPOT} \times N_{\text{output}}`}
            result={(latency.end_to_end_latency_ms / 1000).toFixed(2)}
            unit="秒"
            resultColor="#eb2f96"
          />

          <VariableList
            title="参数说明"
            variables={[
              {
                symbol: '\\text{TTFT}',
                name: '首Token延迟',
                description: 'Prefill阶段处理输入的时间',
              },
              {
                symbol: '\\text{TPOT}',
                name: '每Token延迟',
                description: 'Decode阶段每个token的生成时间',
              },
              {
                symbol: 'N_{\\text{output}}',
                name: '输出Token数',
                description: '生成的输出token数量',
              },
              {
                symbol: '\\text{Prefill}',
                name: 'Prefill阶段',
                description: '处理输入序列，生成KV Cache',
              },
              {
                symbol: '\\text{Decode}',
                name: 'Decode阶段',
                description: '逐token生成输出',
              },
            ]}
          />

          <CalculationSteps
            title="延迟分解"
            steps={[
              {
                label: '\\text{Prefill}_{\\%}',
                formula: '\\frac{\\text{FTL}}{T_{\\text{e2e}}} \\times 100\\%',
                value: (latency.prefill_total_latency_ms / latency.end_to_end_latency_ms * 100).toFixed(1),
                unit: '%',
              },
              {
                label: '\\text{FTL}',
                formula: '\\text{FTL} = \\frac{T_{\\text{compute}} + T_{\\text{comm}}}{1 - \\beta}',
                value: latency.prefill_total_latency_ms.toFixed(2),
                unit: 'ms',
              },
              {
                label: 'T_{\\text{decode}}',
                formula: '\\text{TPOT} \\times N_{\\text{output}}',
                value: (latency.end_to_end_latency_ms - latency.prefill_total_latency_ms).toFixed(1),
                unit: 'ms',
              },
            ]}
          />
        </div>
      )

    case 'chips':
      const { dp, tp, pp, ep } = plan.parallelism
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 15, fontWeight: 600, color: '#fa8c16', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Chip Configuration (芯片配置)</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>资源利用 · 并行策略分解</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              总芯片数由并行策略决定：DP × TP × PP × EP = 总芯片数。
              合理的芯片配置需要平衡延迟、吞吐和成本。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`N_{\text{chips}} = \text{DP} \times \text{TP} \times \text{PP} \times \text{EP}`}
            description="总芯片数 = 数据并行 × 张量并行 × 流水线并行 × 专家并行"
            result={plan.total_chips}
            unit="chips"
            resultColor="#2f54eb"
          />

          <VariableList
            title="并行维度说明"
            variables={[
              {
                symbol: '\\text{DP}',
                name: '数据并行',
                description: '独立处理不同batch的副本数，增加吞吐',
              },
              {
                symbol: '\\text{TP}',
                name: '张量并行',
                description: '单层内切分到多设备，减少单卡显存',
              },
              {
                symbol: '\\text{PP}',
                name: '流水线并行',
                description: '层间切分，适合超大模型',
              },
              {
                symbol: '\\text{EP}',
                name: '专家并行',
                description: 'MoE模型的专家分布',
              },
            ]}
          />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginTop: 16 }}>
              {[
                { label: 'DP', value: dp, color: '#1890ff' },
                { label: 'TP', value: tp, color: '#52c41a' },
                { label: 'PP', value: pp, color: '#fa8c16' },
                { label: 'EP', value: ep, color: '#722ed1' },
              ].map((item) => (
                <div key={item.label} style={{
                  padding: 12,
                  background: `${item.color}10`,
                  borderRadius: 8,
                  textAlign: 'center',
                }}>
                  <div style={{ fontSize: 11, color: item.color }}>{item.label}</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: item.color }}>{item.value}</div>
                </div>
              ))}
            </div>
        </div>
      )

    case 'memory':
      return (
        <div style={detailWrapperStyle}>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#fa8c16', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>Memory Usage (显存占用)</span>
            <span style={{ fontSize: 12, fontWeight: 400, color: '#8c8c8c' }}>资源约束 · 可行性关键指标</span>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitleStyle}>指标定义</div>
            <div style={descStyle}>
              单芯片显存占用包括模型权重、KV Cache和激活值三部分。显存不足会导致OOM，是部署可行性的硬约束。
              TP并行可以减少单卡模型显存，PP并行可以减少单卡激活显存。
            </div>
          </div>

          <FormulaCard
            title="核心公式"
            tex={String.raw`M_{\text{total}} = M_{\text{model}} + M_{\text{KV}} + M_{\text{act}}`}
            description="总显存 = 模型权重 + KV缓存 + 激活值"
            result={memory.total_per_chip_gb.toFixed(2)}
            unit="GB"
            resultColor={memory.is_memory_sufficient ? '#13c2c2' : '#f5222d'}
          />

          <VariableList
            title="显存组成说明"
            variables={[
              {
                symbol: 'M_{\\text{model}}',
                name: '模型权重',
                description: '模型参数占用显存，$M = \\frac{\\text{Params} \\times \\text{bytes}}{\\text{TP}}$',
              },
              {
                symbol: 'M_{\\text{KV}}',
                name: 'KV缓存',
                description: '存储历史token的Key/Value，随序列长度和batch线性增长',
              },
              {
                symbol: 'M_{\\text{act}}',
                name: '激活值',
                description: '前向传播的中间结果，与batch×seq成正比',
              },
              {
                symbol: '\\text{TP}',
                name: '张量并行度',
                description: '模型切分份数，TP越大单卡显存越小',
              },
              {
                symbol: '\\text{PP}',
                name: '流水线并行度',
                description: '层切分份数，PP越大单卡层数越少',
              },
              {
                symbol: '\\text{Params}',
                name: '模型参数量',
                description: '模型总参数数量',
              },
              {
                symbol: '\\text{bytes}',
                name: '参数字节数',
                description: 'FP16=2, BF16=2, FP32=4',
              },
            ]}
          />

          <CalculationSteps
            title="显存分解"
            steps={[
              {
                label: '模型权重',
                formula: 'M_{\\text{model}} = \\frac{\\text{Params} \\times \\text{bytes}}{\\text{TP} \\times \\text{PP}}',
                value: memory.model_memory_gb.toFixed(2),
                unit: 'GB',
              },
              {
                label: 'KV缓存',
                formula: 'M_{\\text{KV}} = 2 \\times L \\times H \\times S \\times B \\times \\text{bytes}',
                value: memory.kv_cache_memory_gb.toFixed(2),
                unit: 'GB',
              },
              {
                label: '激活值',
                formula: 'M_{\\text{act}} = \\text{batch} \\times \\text{seq} \\times H \\times \\text{factor}',
                value: memory.activation_memory_gb.toFixed(2),
                unit: 'GB',
              },
              {
                label: '显存利用率',
                formula: '\\text{Util} = \\frac{M_{\\text{total}}}{M_{\\text{chip}}} \\times 100\\%',
                value: (memory.memory_utilization * 100).toFixed(1),
                unit: '%',
              },
            ]}
          />

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 16 }}>
            <div style={{
              padding: '14px 12px',
              background: '#e6f7ff',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#096dd9', marginBottom: 4 }}>模型权重</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#1890ff' }}>
                {memory.model_memory_gb.toFixed(1)}
              </div>
              <div style={{ fontSize: 10, color: '#096dd9' }}>GB</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#f6ffed',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#389e0d', marginBottom: 4 }}>KV缓存</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>
                {memory.kv_cache_memory_gb.toFixed(1)}
              </div>
              <div style={{ fontSize: 10, color: '#389e0d' }}>GB</div>
            </div>
            <div style={{
              padding: '14px 12px',
              background: '#fff7e6',
              borderRadius: 10,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: '#ad6800', marginBottom: 4 }}>激活值</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#fa8c16' }}>
                {memory.activation_memory_gb.toFixed(1)}
              </div>
              <div style={{ fontSize: 10, color: '#ad6800' }}>GB</div>
            </div>
          </div>

          <div style={{
            marginTop: 12,
            padding: '10px 14px',
            background: memory.is_memory_sufficient ? '#f6ffed' : '#fff2f0',
            borderRadius: 8,
            fontSize: 13,
            color: memory.is_memory_sufficient ? '#52c41a' : '#f5222d',
            textAlign: 'center',
            border: `1px solid ${memory.is_memory_sufficient ? '#b7eb8f' : '#ffa39e'}`,
          }}>
            {memory.is_memory_sufficient ? '✓ 显存充足' : '⚠ 显存不足'} ·
            总占用 <strong>{memory.total_per_chip_gb.toFixed(1)} GB</strong> / 80 GB ·
            利用率 <strong>{(memory.memory_utilization * 100).toFixed(1)}%</strong>
          </div>
        </div>
      )

    default:
      return null
  }
}

export default MetricDetailCard
