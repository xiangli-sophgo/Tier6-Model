/**
 * BaseCard 及性能分析 Card 功能展示页面
 * 展示统一卡片组件的各种用法，以及三种性能分析 Card 风格预览
 */

import React, { useState } from 'react'
import { BaseCard } from '@/components/common/BaseCard'
import {
  Settings,
  Database,
  Gauge,
  ArrowUp,
  ArrowDown,
  Minus,
  Zap,
  Timer,
  Cpu,
  HardDrive,
  Activity,
  TrendingUp,
} from 'lucide-react'
import { getScoreColor, getScoreLabel } from '@/components/ConfigPanel/DeploymentAnalysis/charts/chartTheme'
import {
  TimeBreakdownChart,
  WaterfallChart,
  CostBreakdownChart,
} from '@/components/ConfigPanel/DeploymentAnalysis/charts'

// Mock 数据用于性能分析 Card 预览
const MOCK_PERFORMANCE_DATA = {
  ttft: 85.6,
  tpot: 12.3,
  tps: 1250,
  tpsPerChip: 19.5,
  mfu: 0.42,
  mbu: 0.68,
  memoryUsedGB: 52.4,
  memoryCapacityGB: 80,
  latencyScore: 78,
  throughputScore: 85,
  efficiencyScore: 72,
  balanceScore: 65,
  overallScore: 75,
  bottleneck: 'memory' as const,
  prefillTime: 85.6,
  decodeTime: 1230,
  commRatio: 0.18,
  // 成本相关指标
  totalCost: 275000,           // 总硬件成本 ($)
  serverCost: 180000,          // 服务器成本 ($)
  interconnectCost: 95000,     // 互联成本 ($)
  costPerChip: 4297,           // 单芯片均摊成本 ($)
  costPerMTokens: 0.42,        // 运营成本 ($/M tokens, 3年折旧)
  chipCount: 64,               // 芯片数量

  // ========== 高级图表数据 ==========
  // 时间分解图 - 时间占比 (Treemap)
  timeBreakdownData: {
    name: 'Total Inference',
    value: 1230,
    children: [
      {
        name: 'Prefill (85.6ms)',
        value: 85.6,
        children: [
          { name: 'Attention', value: 45 },
          { name: 'FFN', value: 30 },
          { name: 'Communication', value: 10.6 },
        ],
      },
      {
        name: 'Decode (1144.4ms)',
        value: 1144.4,
        children: [
          { name: 'Attention', value: 800 },
          { name: 'FFN', value: 300 },
          { name: 'Communication', value: 44.4 },
        ],
      },
    ],
  },

  // 瀑布图 - 任务时间线
  waterfallTasks: [
    { name: 'TP0 Compute', start: 0, duration: 50 },
    { name: 'TP1 Compute', start: 0, duration: 50 },
    { name: 'TP2 Compute', start: 0, duration: 48 },
    { name: 'TP3 Compute', start: 0, duration: 50 },
    { name: 'AllReduce', start: 50, duration: 10, deps: ['TP0 Compute', 'TP1 Compute'] },
    { name: 'PP Stage 0', start: 0, duration: 60 },
    { name: 'PP Stage 1', start: 60, duration: 60, deps: ['PP Stage 0'] },
    { name: 'PP Stage 2', start: 120, duration: 60, deps: ['PP Stage 1'] },
  ],

  // 成本分解图 - 成本占比 (Treemap)
  costBreakdownData: {
    name: 'Total Cost',
    value: 275000,
    children: [
      {
        name: 'Hardware',
        value: 180000,
        children: [
          { name: 'Chips', value: 160000 },
          { name: 'Servers', value: 20000 },
        ],
      },
      {
        name: 'Interconnect',
        value: 95000,
        children: [
          { name: 'Intra-Rack', value: 40000 },
          { name: 'Inter-Rack', value: 35000 },
          { name: 'Pod-to-Pod', value: 20000 },
        ],
      },
    ],
  },
}

// 通用的环形进度条组件
const CircularProgress: React.FC<{
  value: number
  max?: number
  size?: number
  strokeWidth?: number
  color?: string
  label?: string
  sublabel?: string
}> = ({ value, max = 100, size = 80, strokeWidth = 8, color = '#60A5FA', label, sublabel }) => {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const progress = Math.min(value, max) / max
  const offset = circumference - progress * circumference

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size}>
        {/* 背景圆 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#f0f0f0"
          strokeWidth={strokeWidth}
        />
        {/* 进度圆 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-bold" style={{ color }}>{label || value.toFixed(0)}</span>
        {sublabel && <span className="text-[10px] text-gray-500">{sublabel}</span>}
      </div>
    </div>
  )
}

// 趋势指示器
const TrendIndicator: React.FC<{ value: number; threshold?: number }> = ({ value, threshold = 0 }) => {
  if (value > threshold + 5) {
    return <ArrowUp className="h-3 w-3 text-green-500" />
  } else if (value < threshold - 5) {
    return <ArrowDown className="h-3 w-3 text-red-500" />
  }
  return <Minus className="h-3 w-3 text-gray-400" />
}

// ============================================
// 性能分析 Card - 紧凑小卡片风格（基于任务详情）
// ============================================
const PerformanceMetricsCard: React.FC = () => {
  const data = MOCK_PERFORMANCE_DATA

  // 小卡片样式 - 紧凑版 + 渐变背景
  const metricCardStyle: React.CSSProperties = {
    padding: '8px',
    textAlign: 'center',
    background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)',
    borderRadius: '6px',
    border: '1px solid #e5e5e5',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  }

  return (
    <BaseCard title="性能指标总览" gradient collapsible defaultExpanded>
      {/* 所有指标统一展示 */}
      <div className="grid grid-cols-6 gap-1.5 mb-2.5">
        <div style={metricCardStyle} className="hover:shadow-md hover:border-blue-300">
          <div className="text-xs text-gray-500">TTFT</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {data.ttft.toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">ms</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-blue-300">
          <div className="text-xs text-gray-500">TPOT</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {data.tpot.toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">ms</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-green-300">
          <div className="text-xs text-gray-500">Total TPS</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {data.tps.toLocaleString()}
            <span className="text-xs font-normal text-gray-400 ml-1">tok/s</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-green-300">
          <div className="text-xs text-gray-500">TPS/Batch</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {(1000 / data.tpot).toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">tok/s</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-green-300">
          <div className="text-xs text-gray-500">TPS/Chip</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {data.tpsPerChip.toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">tok/s</span>
          </div>
        </div>
        <div style={metricCardStyle} className="hover:shadow-md hover:border-purple-300">
          <div className="text-xs text-gray-500">MFU</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {(data.mfu * 100).toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">%</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-cyan-300">
          <div className="text-xs text-gray-500">MBU</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {(data.mbu * 100).toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">%</span>
          </div>
        </div>
        <div style={metricCardStyle} className="hover:shadow-md hover:border-orange-300">
          <div className="text-xs text-gray-500">内存占用</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            {data.memoryUsedGB.toFixed(1)}
            <span className="text-xs font-normal text-gray-400 ml-1">/ {data.memoryCapacityGB}G</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-indigo-300">
          <div className="text-xs text-gray-500">硬件成本</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            ${(data.totalCost / 1000).toFixed(0)}
            <span className="text-xs font-normal text-gray-400 ml-1">K</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-blue-300">
          <div className="text-xs text-gray-500">服务器成本</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            ${(data.serverCost / 1000).toFixed(0)}
            <span className="text-xs font-normal text-gray-400 ml-1">K</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-violet-300">
          <div className="text-xs text-gray-500">互联成本</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            ${(data.interconnectCost / 1000).toFixed(0)}
            <span className="text-xs font-normal text-gray-400 ml-1">K</span>
          </div>
        </div>

        <div style={metricCardStyle} className="hover:shadow-md hover:border-pink-300">
          <div className="text-xs text-gray-500">单芯片成本</div>
          <div className="text-xl font-bold text-gray-800 mt-0.5">
            ${(data.costPerChip / 1000).toFixed(1)}<span className="text-xs font-normal text-gray-400 ml-1">K</span>
          </div>
        </div>
      </div>

      {/* 瓶颈提示 */}
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
        <Activity className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" />
        <div>
          <div className="text-sm font-medium text-yellow-800">瓶颈分析</div>
          <div className="text-xs text-yellow-700 mt-0.5">
            当前方案为<strong>带宽受限</strong>，通信占比 {(data.commRatio * 100).toFixed(0)}%。
            建议优化内存访问模式或增加芯片数量以提升并行度。
          </div>
        </div>
      </div>
    </BaseCard>
  )
}

// ============================================
// 风格 2: 对比卡片 - Prefill/Decode 分栏对比
// ============================================
const ComparisonStyleCard: React.FC = () => {
  const data = MOCK_PERFORMANCE_DATA

  const PhaseColumn: React.FC<{
    title: string
    time: number
    color: string
    metrics: { label: string; value: string; unit?: string }[]
  }> = ({ title, time, color, metrics }) => (
    <div className="flex-1 p-4 rounded-xl" style={{ backgroundColor: `${color}08` }}>
      <div className="flex items-center justify-between mb-3">
        <span className="font-semibold" style={{ color }}>{title}</span>
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ backgroundColor: `${color}15`, color }}>
          {time.toFixed(1)} ms
        </span>
      </div>
      <div className="space-y-2">
        {metrics.map((m, i) => (
          <div key={i} className="flex justify-between items-center text-sm">
            <span className="text-gray-500">{m.label}</span>
            <span className="font-medium text-gray-800">
              {m.value}
              {m.unit && <span className="text-xs text-gray-400 ml-0.5">{m.unit}</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <BaseCard title="阶段对比 (分栏风格)" gradient collapsible defaultExpanded>
      <div className="flex gap-4">
        <PhaseColumn
          title="Prefill 阶段"
          time={data.prefillTime}
          color="#ff4d4f"
          metrics={[
            { label: '计算时间', value: '65.2', unit: 'ms' },
            { label: '通信时间', value: '15.4', unit: 'ms' },
            { label: '访存时间', value: '5.0', unit: 'ms' },
            { label: 'MFU', value: (data.mfu * 100).toFixed(1), unit: '%' },
          ]}
        />
        <div className="flex flex-col items-center justify-center px-2">
          <div className="w-px h-full bg-gray-200" />
          <div className="absolute bg-white px-2 py-1 text-xs text-gray-400">VS</div>
        </div>
        <PhaseColumn
          title="Decode 阶段"
          time={data.decodeTime}
          color="#1890ff"
          metrics={[
            { label: '计算时间', value: '8.5', unit: 'ms' },
            { label: '通信时间', value: '2.8', unit: 'ms' },
            { label: '访存时间', value: '1.0', unit: 'ms' },
            { label: 'MBU', value: (data.mbu * 100).toFixed(1), unit: '%' },
          ]}
        />
      </div>

      {/* 底部汇总 */}
      <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-4 gap-3 text-center">
        <div>
          <div className="text-lg font-bold text-gray-800">{data.tps.toLocaleString()}</div>
          <div className="text-[10px] text-gray-400">总吞吐 (tok/s)</div>
        </div>
        <div>
          <div className="text-lg font-bold text-gray-800">{data.tpsPerChip.toFixed(1)}</div>
          <div className="text-[10px] text-gray-400">TPS/Chip</div>
        </div>
        <div>
          <div className="text-lg font-bold text-gray-800">{data.memoryUsedGB.toFixed(1)}</div>
          <div className="text-[10px] text-gray-400">显存 (GB)</div>
        </div>
        <div>
          <div className="text-lg font-bold" style={{ color: getScoreColor(data.overallScore) }}>
            {data.overallScore.toFixed(0)}
          </div>
          <div className="text-[10px] text-gray-400">综合评分</div>
        </div>
      </div>
    </BaseCard>
  )
}

// ============================================
// 风格 3: 混合风格 - 顶部概览 + 详情表格
// ============================================
const HybridStyleCard: React.FC = () => {
  const data = MOCK_PERFORMANCE_DATA

  const scores = [
    { name: '延迟', score: data.latencyScore, icon: Timer },
    { name: '吞吐', score: data.throughputScore, icon: TrendingUp },
    { name: '效率', score: data.efficiencyScore, icon: Cpu },
    { name: '均衡', score: data.balanceScore, icon: Activity },
  ]

  return (
    <BaseCard title="综合分析 (混合风格)" gradient collapsible defaultExpanded>
      {/* 顶部概览条 */}
      <div className="flex items-center gap-6 p-4 bg-gradient-to-r from-blue-50 via-white to-purple-50 rounded-xl mb-4">
        {/* 大评分 */}
        <div className="flex items-center gap-3">
          <CircularProgress
            value={data.overallScore}
            size={64}
            strokeWidth={6}
            color={getScoreColor(data.overallScore)}
            label={data.overallScore.toFixed(0)}
          />
          <div>
            <div className="text-xs text-gray-500">综合评分</div>
            <div className="font-semibold" style={{ color: getScoreColor(data.overallScore) }}>
              {getScoreLabel(data.overallScore)}
            </div>
          </div>
        </div>

        {/* 分隔线 */}
        <div className="w-px h-12 bg-gray-200" />

        {/* 分项评分 */}
        <div className="flex-1 grid grid-cols-4 gap-4">
          {scores.map(({ name, score, icon: Icon }) => (
            <div key={name} className="flex items-center gap-2">
              <Icon className="h-4 w-4 text-gray-400" />
              <div>
                <div className="text-xs text-gray-500">{name}</div>
                <div className="flex items-center gap-1">
                  <span className="font-semibold" style={{ color: getScoreColor(score) }}>
                    {score.toFixed(0)}
                  </span>
                  <TrendIndicator value={score} threshold={70} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 详情表格 */}
      <div className="border border-gray-100 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">指标</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">数值</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">状态</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            <tr>
              <td className="px-4 py-2.5 text-gray-700">TTFT (首token延迟)</td>
              <td className="px-4 py-2.5 text-right font-medium">{data.ttft.toFixed(1)} ms</td>
              <td className="px-4 py-2.5 text-right">
                <span className={`text-xs px-2 py-0.5 rounded-full ${data.ttft < 100 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {data.ttft < 100 ? '优秀' : '一般'}
                </span>
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2.5 text-gray-700">TPOT (每token延迟)</td>
              <td className="px-4 py-2.5 text-right font-medium">{data.tpot.toFixed(1)} ms</td>
              <td className="px-4 py-2.5 text-right">
                <span className={`text-xs px-2 py-0.5 rounded-full ${data.tpot < 20 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {data.tpot < 20 ? '优秀' : '一般'}
                </span>
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2.5 text-gray-700">TPS (总吞吐量)</td>
              <td className="px-4 py-2.5 text-right font-medium">{data.tps.toLocaleString()} tok/s</td>
              <td className="px-4 py-2.5 text-right">
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">优秀</span>
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2.5 text-gray-700">MFU (算力利用率)</td>
              <td className="px-4 py-2.5 text-right font-medium">{(data.mfu * 100).toFixed(1)}%</td>
              <td className="px-4 py-2.5 text-right">
                <span className={`text-xs px-2 py-0.5 rounded-full ${data.mfu > 0.4 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {data.mfu > 0.4 ? '良好' : '待优化'}
                </span>
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2.5 text-gray-700">MBU (带宽利用率)</td>
              <td className="px-4 py-2.5 text-right font-medium">{(data.mbu * 100).toFixed(1)}%</td>
              <td className="px-4 py-2.5 text-right">
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">优秀</span>
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2.5 text-gray-700">内存占用</td>
              <td className="px-4 py-2.5 text-right font-medium">{data.memoryUsedGB.toFixed(1)} / {data.memoryCapacityGB} GB</td>
              <td className="px-4 py-2.5 text-right">
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">充足</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 瓶颈提示 */}
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
        <Activity className="h-4 w-4 text-yellow-600 mt-0.5" />
        <div>
          <div className="text-sm font-medium text-yellow-800">瓶颈分析</div>
          <div className="text-xs text-yellow-700 mt-0.5">
            当前方案为<strong>带宽受限</strong>，建议优化内存访问模式或增加芯片数量以提升并行度。
          </div>
        </div>
      </div>
    </BaseCard>
  )
}

export default function CardPlayground() {
  const [expanded1, setExpanded1] = useState(true)
  const [expanded2, setExpanded2] = useState(false)
  const [selectedStyle, setSelectedStyle] = useState<'metrics' | 'comparison' | 'hybrid'>('metrics')

  return (
    <div className="w-full h-full bg-gray-50 overflow-auto">
      <div className="p-6 space-y-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">组件展示</h1>
          <p className="text-gray-600">BaseCard 功能展示 + 性能分析 Card 风格预览</p>
        </div>

        {/* ================================================ */}
        {/* 性能分析 Card 风格预览 */}
        {/* ================================================ */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b pb-3">
            性能分析 Card 风格预览
          </h2>
          <p className="text-sm text-gray-500 mb-6">
            以下是三种不同风格的性能分析 Card 设计，可根据实际需求选择最合适的方案。
          </p>

          {/* 风格切换按钮 */}
          <div className="flex gap-2 mb-6">
            {[
              { key: 'metrics', label: '紧凑指标卡片', desc: '小卡片并列 + 数字优先' },
              { key: 'comparison', label: '阶段对比卡片', desc: 'Prefill/Decode分栏' },
              { key: 'hybrid', label: '混合详情卡片', desc: '概览条 + 详情表格' },
            ].map(({ key, label, desc }) => (
              <button
                key={key}
                onClick={() => setSelectedStyle(key as typeof selectedStyle)}
                className={`px-4 py-3 rounded-lg border-2 transition-all ${
                  selectedStyle === key
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{label}</div>
                <div className="text-xs opacity-70">{desc}</div>
              </button>
            ))}
          </div>

          {/* 选中的风格预览 */}
          <div className="mb-8">
            {selectedStyle === 'metrics' && <PerformanceMetricsCard />}
            {selectedStyle === 'comparison' && <ComparisonStyleCard />}
            {selectedStyle === 'hybrid' && <HybridStyleCard />}
          </div>

          {/* 三种风格并排对比 */}
          <details className="mb-8">
            <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-700 mb-4">
              展开查看三种风格并排对比
            </summary>
            <div className="space-y-6">
              <PerformanceMetricsCard />
              <ComparisonStyleCard />
              <HybridStyleCard />
            </div>
          </details>
        </div>

        {/* ================================================ */}
        {/* BaseCard 功能展示 */}
        {/* ================================================ */}
        <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b pb-3">
          BaseCard 功能展示
        </h2>

        {/* 1. 基础用法 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            1. 基础用法
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard title="标准卡片" subtitle="这是副标题">
              <p className="text-sm">这是一个基础的卡片内容区域。</p>
            </BaseCard>

            <BaseCard
              title="带图标的卡片"
              icon={<Settings className="h-4 w-4 text-blue-600" />}
            >
              <p className="text-sm">左侧显示图标</p>
            </BaseCard>
          </div>
        </div>

        {/* 2. 渐变模式 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            2. 渐变模式 (gradient)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard title="渐变卡片" gradient>
              <p className="text-sm">蓝色渐变背景标题</p>
              <p className="text-xs text-gray-500">bg-gradient-to-r from-blue-50 to-white</p>
            </BaseCard>

            <BaseCard
              title="带副标题"
              subtitle="副标题信息"
              gradient
              icon={<Database className="h-4 w-4 text-blue-600" />}
            >
              <p className="text-sm">渐变模式 + 图标 + 副标题</p>
            </BaseCard>
          </div>
        </div>

        {/* 3. 可折叠模式 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            3. 可折叠模式 (collapsible)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard
              title="非受控折叠"
              collapsible
              defaultExpanded={true}
              gradient
            >
              <p className="text-sm">点击标题栏可以折叠/展开</p>
              <p className="text-xs text-gray-500">使用 defaultExpanded</p>
            </BaseCard>

            <BaseCard
              title="受控折叠"
              collapsible
              expanded={expanded1}
              onExpandChange={setExpanded1}
              gradient
            >
              <p className="text-sm">受控模式，状态由外部管理</p>
              <p className="text-xs text-gray-500">expanded + onExpandChange</p>
            </BaseCard>
          </div>
        </div>

        {/* 4. 折叠计数 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            4. 折叠计数显示 (collapsibleCount)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard
              title="手动连接"
              collapsible
              defaultExpanded
              collapsibleCount={5}
              gradient
            >
              <p className="text-sm">标题后显示计数 (5)</p>
              <p className="text-xs text-gray-500">适合列表类内容</p>
            </BaseCard>

            <BaseCard
              title="当前连接"
              collapsible
              expanded={expanded2}
              onExpandChange={setExpanded2}
              collapsibleCount={12}
              gradient
            >
              <p className="text-sm">受控模式 + 计数显示</p>
              <p className="text-xs text-gray-500">collapsibleCount={12}</p>
            </BaseCard>
          </div>
        </div>

        {/* 5. 编辑按钮 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            5. 编辑按钮 (onEdit)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard
              title="集群规模"
              collapsible
              defaultExpanded
              collapsibleCount={4}
              onEdit={() => alert('编辑集群规模')}
              gradient
            >
              <p className="text-sm">右上角显示编辑按钮</p>
              <p className="text-xs text-gray-500">点击编辑按钮触发回调</p>
            </BaseCard>

            <BaseCard
              title="硬件配置"
              collapsible
              defaultExpanded
              collapsibleCount={8}
              onEdit={() => alert('编辑硬件配置')}
              editLabel="修改"
              gradient
            >
              <p className="text-sm">自定义编辑按钮文字</p>
              <p className="text-xs text-gray-500">editLabel="修改"</p>
            </BaseCard>
          </div>
        </div>

        {/* 6. Titleless模式 */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-gray-700 border-b pb-2">
            6. Titleless模式 - 完全自定义
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <BaseCard
              titleless
              className="cursor-pointer hover:shadow-md"
            >
              <div className="px-4 py-8 text-center">
                <Settings className="h-12 w-12 text-blue-600 mx-auto mb-3" />
                <h3 className="font-semibold text-lg mb-2">快速操作</h3>
                <p className="text-sm text-gray-500">无标题栏，完全自定义布局</p>
              </div>
            </BaseCard>

            <BaseCard
              titleless
              glassmorphism
              className="cursor-pointer hover:shadow-md"
            >
              <div className="px-4 py-8 text-center">
                <Database className="h-12 w-12 text-purple-600 mx-auto mb-3" />
                <h3 className="font-semibold text-lg mb-2">毛玻璃效果</h3>
                <p className="text-sm text-gray-500">glassmorphism + titleless</p>
              </div>
            </BaseCard>
          </div>
        </div>

        {/* Props 总结 */}
        <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">BaseCard Props 总结</h2>
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-3 gap-4 font-semibold border-b pb-2">
              <div>Props</div>
              <div>类型</div>
              <div>说明</div>
            </div>
            {[
              ['title', 'ReactNode', '标题内容'],
              ['subtitle', 'string', '副标题'],
              ['icon', 'ReactNode', '标题图标'],
              ['gradient', 'boolean', '蓝色渐变标题'],
              ['collapsible', 'boolean', '可折叠'],
              ['defaultExpanded', 'boolean', '默认展开'],
              ['expanded', 'boolean', '受控展开状态'],
              ['onExpandChange', 'function', '展开状态回调'],
              ['contentClassName', 'string', '内容区自定义样式 [*]', true],
              ['titleless', 'boolean', '无标题模式 [*]', true],
              ['glassmorphism', 'boolean', '毛玻璃效果 [*]', true],
              ['collapsibleCount', 'number', '折叠区计数 [*]', true],
              ['onEdit', 'function', '编辑按钮回调 [*]', true],
              ['editLabel', 'string', '编辑按钮文字'],
            ].map(([prop, type, desc, highlight]) => (
              <div key={prop as string} className={`grid grid-cols-3 gap-4 py-2 border-b ${highlight ? 'bg-blue-50' : ''}`}>
                <div className="font-mono text-xs">{prop}</div>
                <div className="text-gray-600">{type}</div>
                <div className={`text-gray-600 ${highlight ? 'font-semibold' : ''}`}>{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ================================================ */}
        {/* 高级图表展示 */}
        {/* ================================================ */}
        <div className="mt-12">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b pb-3">
            高级图表展示
          </h2>
          <p className="text-sm text-gray-500 mb-6">
            3种高级可视化图表类型，用于大模型部署分析的数据展示
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* 时间分解图 (Treemap) */}
            <BaseCard title="时间分解图 - 时间占比" gradient collapsible defaultExpanded>
              <TimeBreakdownChart
                data={MOCK_PERFORMANCE_DATA.timeBreakdownData}
                height={300}
                title=""
              />
            </BaseCard>

            {/* 瀑布图 */}
            <BaseCard title="瀑布图 - 任务时间线" gradient collapsible defaultExpanded>
              <WaterfallChart
                data={MOCK_PERFORMANCE_DATA.waterfallTasks}
                height={300}
                title=""
              />
            </BaseCard>

            {/* 成本分解图 (Treemap) */}
            <BaseCard title="成本分解图 - 成本占比" gradient collapsible defaultExpanded>
              <CostBreakdownChart
                data={MOCK_PERFORMANCE_DATA.costBreakdownData}
                height={300}
                title=""
                unit="$"
              />
            </BaseCard>
          </div>

          {/* 图表说明 */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-800 mb-2">图表说明</h3>
            <ul className="text-xs text-blue-700 space-y-1">
              <li><strong>时间分解图 (Treemap)</strong>: 使用矩形面积显示各模块的时间占比，便于快速识别耗时模块</li>
              <li><strong>瀑布图</strong>: 展示并行任务的时间线，清晰显示任务的开始时间和持续时间</li>
              <li><strong>成本分解图 (Treemap)</strong>: 使用矩形面积展示成本的分层占比，支持多级嵌套数据</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
