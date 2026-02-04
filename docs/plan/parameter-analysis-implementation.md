# 参数分析图表移植实施计划

## 目标

将 CrossRing 项目的单参数曲线图和双参数热力图功能移植到 Tier6+Model,用于可视化分析参数扫描结果。

## 架构方案

### 技术选型
- **数据处理**: 前端聚合(从 `EvaluationResult` 中提取并计算统计值)
- **图表库**: ECharts 6.0 + echarts-for-react(与 CrossRing 兼容)
- **UI 集成**: Results 页面新增"参数分析" Tab
- **参数选择器**: TreeSelect 分层选择器(复用 CrossRing 的分类逻辑)

### 数据流
```
用户选择实验 (Results 页面)
  ↓
前端查询: GET /api/evaluation/experiments/{id}
  ↓
获取该实验的所有 EvaluationResult[]
  ↓
数据聚合处理:
  - 从 config_snapshot 提取参数值
  - 从 full_result 提取性能指标 (tps/tpot/ttft/mfu)
  - 按参数值分组统计(均值/最大/最小)
  ↓
渲染图表:
  - 单参数曲线图(均值 + 范围阴影)
  - 双参数热力图(基准值调整 + 下降比例)
```

---

## 实施步骤

### Phase 1: 数据层 (Day 1)

#### 1.1 创建数据聚合工具模块

**新建文件**: `frontend/src/pages/Results/utils/parameterAnalysis.ts`

**核心功能**:
- `extractParametersFromResults(results)` - 从结果列表中提取所有参数及其取值
- `aggregateSensitivityData(results, parameter, metric)` - 单参数敏感度聚合
  - 按参数值分组
  - 计算每组的均值、最大值、最小值、样本数
- `aggregateHeatmapData(results, paramX, paramY, metric)` - 双参数热力图聚合
  - 构建二维参数网格
  - 计算每个网格点的平均性能
- `extractParamValue(config, path)` - 从 config_snapshot 中提取嵌套参数值
  - 支持路径解析(如 `parallelism.tp`)

**数据格式**:
```typescript
// 单参数敏感度数据点
interface SensitivityDataPoint {
  value: number              // 参数值
  mean_performance: number   // 平均性能
  min_performance: number    // 最小性能
  max_performance: number    // 最大性能
  count: number              // 样本数
}

// 双参数热力图数据
interface HeatmapData {
  param_x: string
  param_y: string
  x_values: number[]         // X轴取值列表
  y_values: number[]         // Y轴取值列表
  data: Array<{              // 网格数据点
    x_value: number
    y_value: number
    mean_performance: number
  }>
}

// 图表配置(用于保存/加载)
interface ChartConfig {
  id: string
  name: string
  type: 'line' | 'heatmap'
  parameters: string[]       // 参数路径列表
  metric: string             // 性能指标
  createdAt: number
}
```

---

### Phase 2: 参数选择器 (Day 1-2)

#### 2.1 创建参数分类工具

**新建文件**: `frontend/src/pages/Results/utils/parameterClassifier.ts`

**参考**: CrossRing 的 `paramClassifier.ts`

**功能**: 将参数按类别分层(用于 TreeSelect)
- **第一层**: 配置类别
  - 模型配置(model)
  - 推理配置(inference)
  - 硬件参数(hardware)
  - 并行策略(parallelism)
  - 拓扑配置(topology)
- **第二层**: 具体参数
  - 带中文标签和单位

**实现逻辑**:
```typescript
export function classifyParameters(
  results: EvaluationResult[]
): ParameterTreeNode[] {
  const paramMap = extractParametersFromResults(results)

  return [
    {
      key: 'model',
      title: '模型配置',
      children: [
        { key: 'model.hidden_size', title: '隐藏层维度', unit: '' },
        { key: 'model.num_layers', title: '层数', unit: '' },
        // ...
      ]
    },
    {
      key: 'inference',
      title: '推理配置',
      children: [
        { key: 'inference.batch_size', title: '批次大小', unit: '' },
        { key: 'inference.input_seq_length', title: '输入序列长度', unit: 'tokens' },
        // ...
      ]
    },
    // ... parallelism, hardware, topology
  ]
}
```

#### 2.2 创建 TreeSelect 参数选择器组件

**新建文件**: `frontend/src/pages/Results/components/ParameterTreeSelect.tsx`

**基于**: shadcn/ui 的 Popover + Command 组件实现树形选择

**功能**:
- 多选模式(单参数曲线选1个,双参数热力图选2个)
- 搜索过滤
- 分层展示(可展开/折叠)
- 显示已选参数标签(最多3个,超出显示"+N")

---

### Phase 3: 图表组件 (Day 2-3)

#### 3.1 单参数曲线图

**新建文件**: `frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/SingleParamLineChart.tsx`

**参考**: CrossRing 的 `SingleParamLineChart` 组件(第134-276行)

**核心实现**:
- ECharts 配置: 3条系列叠加
  - 系列1: 最小值(不可见,用于堆叠基准)
  - 系列2: 范围(最大值-最小值,area style 半透明)
  - 系列3: 均值(实线,带标记点)
- 主题适配: 使用 `chartTheme.ts` 的 `CHART_COLORS.primary`
- Tooltip 增强: 显示均值、最大、最小、样本数

**配置示例**:
```typescript
series: [
  {
    name: '最小值',
    type: 'line',
    data: data.map(d => d.min_performance),
    lineStyle: { opacity: 0 },
    stack: 'range',
    symbol: 'none'
  },
  {
    name: '范围',
    type: 'line',
    data: data.map(d => d.max_performance - d.min_performance),
    areaStyle: { opacity: 0.15, color: CHART_COLORS.primary },
    lineStyle: { opacity: 0 },
    stack: 'range',
    symbol: 'none',
    smooth: true
  },
  {
    name: '均值',
    type: 'line',
    data: data.map(d => d.mean_performance),
    lineStyle: { color: CHART_COLORS.primary, width: 2 },
    itemStyle: { color: CHART_COLORS.primary },
    smooth: true,
    symbolSize: 6
  }
]
```

**交互功能**:
- 图表导出(PNG,pixelRatio=2)
- 响应式容器(ResizeObserver)

#### 3.2 双参数热力图

**新建文件**: `frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/DualParamHeatmap.tsx`

**参考**: CrossRing 的 `DualParamHeatmap` 组件(第278-518行)

**核心实现**:
- ECharts heatmap 类型
- visualMap 组件(支持拖动调整基准值)
- 动态单元格大小(固定60px正方形)
- 下降比例显示(rich text formatter)

**关键功能**:
1. **基准值调整系统**
```typescript
const [baseValue, setBaseValue] = useState<number | null>(null)
const effectiveBaseValue = baseValue ?? Math.max(...data.map(d => d.mean_performance))

// 监听 visualMap 拖动事件
const onEvents = {
  datarangeselected: (params: any) => {
    setBaseValue(params.selected[1])  // 上限值
  }
}
```

2. **下降比例计算**
```typescript
formatter: (params: any) => {
  const val = params.data[2]
  const dropPercent = ((effectiveBaseValue - val) / effectiveBaseValue) * 100
  const valueStr = val.toFixed(precision)
  const dropStr = dropPercent > 0 ? `-${dropPercent.toFixed(1)}%` : ''

  return dropStr
    ? `${valueStr}\n{drop|${dropStr}}`
    : valueStr
}
```

3. **自适应精度和字体**
```typescript
const range = maxVal - minVal
const precision = range > 100 ? 0 : range > 10 ? 1 : 2
const fontSize = Math.max(8, Math.min(12, cellSize / 5))
```

**visualMap 配置**:
```typescript
visualMap: {
  type: 'continuous',
  min: minVal,
  max: maxVal,
  calculable: true,     // 允许拖动
  realtime: false,      // 拖动结束后触发
  inRange: {
    color: ['#f5f5f5', '#91d5ff', '#1890ff', '#ff7875', '#ff4d4f']
    // 浅灰 → 浅蓝 → 深蓝 → 浅红 → 深红(适配 Tier6+Model 主题)
  }
}
```

---

### Phase 4: 主容器组件 (Day 3)

#### 4.1 参数分析面板

**新建文件**: `frontend/src/pages/Results/components/ParameterAnalysisPanel.tsx`

**职责**:
- 协调参数选择、图表渲染、配置管理
- 状态管理(图表类型、选中参数、性能指标)
- 数据聚合调用

**组件结构**:
```typescript
export const ParameterAnalysisPanel: React.FC<{
  experimentId: number
  results: EvaluationResult[]
}> = ({ experimentId, results }) => {
  // 状态
  const [chartType, setChartType] = useState<'line' | 'heatmap'>('line')
  const [selectedParams, setSelectedParams] = useState<string[]>([])
  const [selectedMetric, setSelectedMetric] = useState<'tps' | 'tpot' | 'ttft' | 'mfu'>('tps')

  // 提取可用参数(树形结构)
  const paramTree = useMemo(() =>
    classifyParameters(results), [results]
  )

  // 数据聚合
  const chartData = useMemo(() => {
    if (chartType === 'line' && selectedParams.length === 1) {
      return aggregateSensitivityData(results, selectedParams[0], selectedMetric)
    } else if (chartType === 'heatmap' && selectedParams.length === 2) {
      return aggregateHeatmapData(results, selectedParams[0], selectedParams[1], selectedMetric)
    }
    return null
  }, [results, selectedParams, selectedMetric, chartType])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3>参数分析</h3>
          <div className="flex gap-2">
            {/* 保存配置按钮 */}
            <Button onClick={handleSaveConfig}>保存配置</Button>
            {/* 导出图表按钮 */}
            <Button onClick={handleExport}>导出PNG</Button>
          </div>
        </div>

        <div className="flex items-center gap-4 mt-4">
          {/* 图表类型切换 */}
          <Select value={chartType} onValueChange={setChartType}>
            <SelectItem value="line">单参数曲线</SelectItem>
            <SelectItem value="heatmap">双参数热力图</SelectItem>
          </Select>

          {/* 性能指标选择 */}
          <Select value={selectedMetric} onValueChange={setSelectedMetric}>
            <SelectItem value="tps">吞吐量 (TPS)</SelectItem>
            <SelectItem value="tpot">每Token延迟 (TPOT)</SelectItem>
            <SelectItem value="ttft">首Token延迟 (TTFT)</SelectItem>
            <SelectItem value="mfu">模型效率 (MFU)</SelectItem>
          </Select>

          {/* 参数选择器(TreeSelect) */}
          <ParameterTreeSelect
            tree={paramTree}
            value={selectedParams}
            onChange={setSelectedParams}
            maxSelection={chartType === 'line' ? 1 : 2}
          />
        </div>
      </CardHeader>

      <CardContent>
        {!chartData && (
          <div className="text-center text-gray-500 py-8">
            请选择参数开始分析
          </div>
        )}

        {chartData && chartType === 'line' && (
          <SingleParamLineChart
            data={chartData}
            paramName={selectedParams[0]}
            metricName={METRIC_LABELS[selectedMetric]}
          />
        )}

        {chartData && chartType === 'heatmap' && (
          <DualParamHeatmap
            data={chartData}
            metricName={METRIC_LABELS[selectedMetric]}
          />
        )}
      </CardContent>
    </Card>
  )
}
```

#### 4.2 配置保存/加载功能

**存储方式**: localStorage(键名: `tier6_chart_configs_{experimentId}`)

**功能实现**:
```typescript
// 保存配置
const handleSaveConfig = () => {
  const config: ChartConfig = {
    id: Date.now().toString(),
    name: prompt('配置名称') || '未命名配置',
    type: chartType,
    parameters: selectedParams,
    metric: selectedMetric,
    createdAt: Date.now()
  }

  const existing = JSON.parse(
    localStorage.getItem(`tier6_chart_configs_${experimentId}`) || '[]'
  )
  existing.push(config)
  localStorage.setItem(
    `tier6_chart_configs_${experimentId}`,
    JSON.stringify(existing)
  )

  toast.success('配置已保存')
}

// 加载配置
const handleLoadConfig = (configId: string) => {
  const configs = JSON.parse(
    localStorage.getItem(`tier6_chart_configs_${experimentId}`) || '[]'
  )
  const config = configs.find((c: ChartConfig) => c.id === configId)

  if (config) {
    setChartType(config.type)
    setSelectedParams(config.parameters)
    setSelectedMetric(config.metric)
  }
}

// 显示已保存配置列表(下拉菜单)
<DropdownMenu>
  <DropdownMenuTrigger>加载配置</DropdownMenuTrigger>
  <DropdownMenuContent>
    {savedConfigs.map(config => (
      <DropdownMenuItem onClick={() => handleLoadConfig(config.id)}>
        {config.name} ({config.type})
      </DropdownMenuItem>
    ))}
  </DropdownMenuContent>
</DropdownMenu>
```

---

### Phase 5: Results 页面集成 (Day 4)

#### 5.1 修改 Results 页面

**修改文件**: `frontend/src/pages/Results/index.tsx`

**变更内容**:
1. 新增 Tab: "参数分析"
2. 实验选择逻辑(需选中一个实验才能查看参数分析)
3. 查询该实验的所有 EvaluationResult

**代码示例**:
```typescript
export const Results: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'list' | 'analysis'>('list')
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null)
  const [experimentResults, setExperimentResults] = useState<EvaluationResult[]>([])

  // 查询实验结果
  useEffect(() => {
    if (selectedExperimentId && activeTab === 'analysis') {
      fetchExperimentResults(selectedExperimentId).then(setExperimentResults)
    }
  }, [selectedExperimentId, activeTab])

  return (
    <div className="p-6">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="list">实验列表</TabsTrigger>
          <TabsTrigger value="analysis" disabled={!selectedExperimentId}>
            参数分析
          </TabsTrigger>
        </TabsList>

        <TabsContent value="list">
          {/* 现有实验列表 */}
          <ExperimentTable
            onSelectExperiment={setSelectedExperimentId}
          />
        </TabsContent>

        <TabsContent value="analysis">
          {selectedExperimentId && experimentResults.length > 0 ? (
            <ParameterAnalysisPanel
              experimentId={selectedExperimentId}
              results={experimentResults}
            />
          ) : (
            <div className="text-center text-gray-500 py-8">
              该实验暂无结果数据
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
```

#### 5.2 API 调用复用

**复用现有端点**: `GET /api/evaluation/experiments/{experiment_id}`
- 已经返回实验的所有任务和结果
- 从 `tasks` 数组中提取 `config_snapshot` 和 `result`

---

### Phase 6: 测试与优化 (Day 4-5)

#### 6.1 功能测试清单

- [ ] 单参数曲线图
  - [ ] 数据聚合正确(均值、最大、最小)
  - [ ] 图表渲染正常(阴影区域、曲线)
  - [ ] Tooltip 显示完整信息
  - [ ] 导出 PNG 功能

- [ ] 双参数热力图
  - [ ] 网格数据正确
  - [ ] 基准值调整交互
  - [ ] 下降比例计算准确
  - [ ] 自适应单元格大小

- [ ] 参数选择器
  - [ ] 树形结构展示正确
  - [ ] 搜索过滤功能
  - [ ] 多选/单选限制

- [ ] 配置保存/加载
  - [ ] 保存到 localStorage
  - [ ] 加载恢复参数和指标
  - [ ] 配置列表显示

- [ ] 边界情况
  - [ ] 无数据时显示提示
  - [ ] 单数据点时禁用分析
  - [ ] 参数值缺失时跳过
  - [ ] 大规模数据(>1000结果)性能测试

#### 6.2 性能优化

- 使用 `useMemo` 缓存聚合结果
- 参数树构建缓存
- 图表配置对象缓存
- 大数据场景(>500结果)显示加载提示

#### 6.3 样式适配

- 统一使用 `chartTheme.ts` 配色
- 卡片样式使用 `CHART_CARD_STYLE`
- 响应式布局(移动端优化)
- 暗色主题支持(如果项目有)

---

## 关键文件清单

### 新建文件(共7个)

1. **`frontend/src/pages/Results/utils/parameterAnalysis.ts`**
   - 数据聚合工具模块
   - 核心算法实现

2. **`frontend/src/pages/Results/utils/parameterClassifier.ts`**
   - 参数分类逻辑
   - 树形结构生成

3. **`frontend/src/pages/Results/components/ParameterTreeSelect.tsx`**
   - TreeSelect 参数选择器组件

4. **`frontend/src/pages/Results/components/ParameterAnalysisPanel.tsx`**
   - 参数分析主容器组件

5. **`frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/SingleParamLineChart.tsx`**
   - 单参数曲线图组件

6. **`frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/DualParamHeatmap.tsx`**
   - 双参数热力图组件

7. **`frontend/src/api/tasks.ts`** (可能需要新增函数)
   - `fetchExperimentResults()` API 调用

### 修改文件(共2个)

1. **`frontend/src/pages/Results/index.tsx`**
   - 新增"参数分析" Tab
   - 实验选择逻辑
   - 结果数据查询

2. **`frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/index.ts`**
   - 导出新增的图表组件

### 后端修改

**无需修改** - 现有 `GET /api/evaluation/experiments/{experiment_id}` 端点已满足需求

---

## 验证方式

### 端到端测试流程

1. **创建参数扫描实验**
   - 在 DeploymentAnalysis 页面配置参数扫描
   - 选择 2-3 个参数(如 `parallelism.tp`, `inference.batch_size`)
   - 提交批量任务(约10-20个组合)

2. **等待任务完成**
   - 查看 Results 页面任务状态
   - 确认所有任务状态为 `completed`

3. **查看参数分析**
   - 点击实验行,切换到"参数分析" Tab
   - 选择单参数曲线图,选择 `parallelism.tp`,指标选择 `tps`
   - 验证曲线图正常显示,阴影范围合理

4. **测试双参数热力图**
   - 切换到双参数热力图
   - 选择 `parallelism.tp` 和 `inference.batch_size`
   - 验证热力图正常显示,拖动色条调整基准值
   - 检查下降比例标注是否准确

5. **测试配置保存/加载**
   - 点击"保存配置",输入名称
   - 清空参数选择
   - 从"加载配置"菜单中选择刚保存的配置
   - 验证参数和指标恢复正确

6. **测试导出功能**
   - 点击"导出PNG"按钮
   - 验证导出的图片清晰(2x pixelRatio)

---

## 技术注意事项

### 1. ECharts 版本兼容
- CrossRing: 5.5.1 → Tier6+Model: 6.0
- 无重大 Breaking Changes,配置可直接复用
- 注意 TypeScript 类型定义可能有细微差异

### 2. 参数路径提取
- `config_snapshot` 为嵌套 JSON 对象
- 使用路径字符串(如 `parallelism.tp`)通过点分割递归提取
- 处理缺失值(某些任务可能没有该参数)

### 3. 性能优化策略
- 数据量 <500: 前端聚合无压力
- 数据量 500-1000: 使用 Web Worker 聚合(可选)
- 数据量 >1000: 考虑后端 API 优化(Phase 2)

### 4. 错误处理
- 参数值非数值 → 跳过该结果
- 参数缺失 → 跳过该结果
- 数据点 <2 → 显示"数据不足"提示
- 网络错误 → 显示错误提示,支持重试

### 5. 主题适配
- 复用 `chartTheme.ts` 的 `CHART_COLORS.primary` (#5E6AD2)
- 热力图色阶调整为与品牌色一致的蓝色系
- 卡片样式使用 `CHART_CARD_STYLE`

---

## 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| Phase 1 | 数据聚合工具 | 0.5天 |
| Phase 2 | 参数选择器 | 1天 |
| Phase 3 | 图表组件 | 1.5天 |
| Phase 4 | 主容器组件 | 0.5天 |
| Phase 5 | Results页面集成 | 0.5天 |
| Phase 6 | 测试与优化 | 1天 |
| **总计** | | **5天** |

---

## 成功标准

- ✅ 单参数曲线图正常展示均值和范围阴影
- ✅ 双参数热力图支持基准值调整和下降比例显示
- ✅ TreeSelect 参数选择器分层展示,支持搜索
- ✅ 配置保存/加载功能正常
- ✅ 图表导出 PNG 清晰
- ✅ 性能良好(<1000结果时聚合 <1秒)
- ✅ 无明显 Bug,边界情况处理完善
- ✅ 代码风格符合项目规范,TypeScript 类型完整
