# Tier6 配置系统设计

## 1. 配置层次结构

四类配置文件，各自独立管理，通过引用关联：

```
backend/tier6/configs/
├── chips/              # 芯片预设
│   ├── _template.yaml  # 模板 (带注释说明，不会被加载)
│   └── SG2262.yaml     # 芯片配置
├── models/             # 模型预设
│   └── deepseek_v3.yaml
├── benchmarks/         # 场景预设 (引用模型)
│   └── deepseek_v3-S32K-O1K-W16A16-B1.yaml
└── topologies/         # 拓扑预设 (引用芯片)
    └── P1-R1-B1-C8.yaml
```

引用关系：

```
benchmark  --model: deepseek_v3-->  models/deepseek_v3.yaml
topology   --chips.name: SG2262-->  chips/SG2262.yaml
```

## 2. 芯片配置格式

### 设计决策

- **两级存储模型**: gmem (片外 DRAM) + lmem (核内 SRAM)，不建模 l2m 和 smem
- **单 DMA 引擎**: 只有 GDMA (gmem <-> lmem)，不建模 SDMA 和 CDMA
- **片内互联 vs 片间互联**: 芯片配置只定义 NoC (片内)，c2c/b2b/r2r/p2p 在拓扑配置中定义

### 依据

CHIPMathica TPU Simulator 和 llm_simulator 两个现有模拟器均使用两级存储 + 单 GDMA 的简化模型，
l2m/smem 虽然物理上存在，但当前无模拟器对其建模，配置中不保留。

### 字段说明

详见 `chips/_template.yaml`，核心字段：

| 字段 | 用途 |
|------|------|
| `cores.count / lanes_per_core` | 核心结构 |
| `compute_units.cube.m/k/n` | 矩阵计算单元维度，决定 tiling 分块 |
| `compute_units.cube.mac_per_lane` | 按精度的 MAC 吞吐量 |
| `memory.gmem` | 片外存储 (type/capacity/bandwidth/utilization/latency) |
| `memory.lmem` | 核内 SRAM (capacity/bandwidth/latency/sram_utilization) |
| `dma_engines.gdma` | GDMA 引擎 (bandwidth/startup_latency/efficiency) |
| `align_bytes` | 内存对齐，影响 tiling padding |
| `compute_dma_overlap_rate` | 计算-DMA 重叠率，影响流水线隐藏 |
| `interconnect.noc` | 片内 NoC (topology/bandwidth/latency) |

### 峰值算力推导

```
TFLOPS = cores * lanes_per_core * mac_per_lane[dtype] * frequency_ghz * 2 / 1000
```

例: SG2262 FP8 = 4 * 64 * 1000 * 1.0 * 2 / 1000 = 512 TFLOPS

## 3. Benchmark 配置格式

### 设计决策

- Benchmark 通过引用关联模型预设，不内联模型参数
- Benchmark 只定义推理场景参数 (batch_size, seq_length 等)
- 避免参数重复，模型定义只在 `models/` 中维护一份

### 格式

```yaml
id: deepseek_v3-S32K-O1K-W16A16-B1
name: DeepSeek-V3 S32K O1K BF16 Batch1
model: deepseek_v3                      # 引用 models/deepseek_v3.yaml
inference:
  batch_size: 1
  input_seq_length: 32768
  output_seq_length: 1024
```

### 命名约定

`{model}-S{input_seq}-O{output_seq}-W{weight_dtype}A{act_dtype}-B{batch_size}`

## 4. 拓扑配置格式

### 设计决策

- 拓扑配置通过芯片名称引用芯片预设
- 片间互联 (c2c/b2b/r2r/p2p) 在拓扑的 `hardware_params.interconnect` 中定义
- 通信延迟配置 (comm_latency_config) 在拓扑配置中定义

### 格式

```yaml
name: P1-R1-B1-C8
pod_count: 1
racks_per_pod: 1
rack_config:
  boards:
    - chips:
        - name: SG2262          # 引用 chips/SG2262.yaml
          count: 8
hardware_params:
  interconnect:                 # 片间互联 (不在芯片配置中)
    c2c:
      bandwidth_gbps: 448
      latency_us: 0.2
    b2b: ...
    r2r: ...
    p2p: ...
comm_latency_config:
  rtt_tp_us: 0.35
  rtt_ep_us: 0.85
  ...
```

## 5. 数据流

```
[配置文件层 - YAML]
  引用关系，不重复数据
       |
       v
[后端: ConfigLoader]
  加载 YAML -> 解析引用 -> 合并为完整配置 -> 校验必填字段
       |
       v
[API 层]
  返回 resolved 完整配置给前端 (不含引用)
       |
       v
[前端: 展示 + 编辑]
  用户可修改任何参数，可设置参数扫描
       |
       v
[前端: 提交]
  发送完整配置 (用户确认后的最终版，无引用)
       |
       v
[后端: 仿真引擎]
  直接消费完整配置
       |
       v
[结果: 快照存储]
  保存提交时的完整配置 (确保可复现)
```

## 6. 前端保存/另存为/重载逻辑

### 四类配置各自独立 CRUD

每类配置都有独立的保存/另存为/重载操作，通过引用串联：

| 配置类型 | 加载 | 保存 | 另存为 | 重载 |
|----------|------|------|--------|------|
| **芯片预设** | 从 YAML 加载 | 覆盖 YAML | 新建 YAML | 丢弃编辑，重新加载 |
| **模型预设** | 从 YAML 加载 | 覆盖 YAML | 新建 YAML | 丢弃编辑，重新加载 |
| **Benchmark** | 从 YAML 加载 + 解析模型引用 | 覆盖 YAML (保持引用) | 新建 YAML | 丢弃编辑，重新加载 |
| **拓扑配置** | 从 YAML 加载 + 解析芯片引用 | 覆盖 YAML (保持引用) | 新建 YAML | 丢弃编辑，重新加载 |

### 后端 API

每类配置对应一组 CRUD 端点：

```
GET    /api/{type}           # 列表
GET    /api/{type}/{name}    # 加载 (返回 resolved 完整配置)
POST   /api/{type}           # 创建 (另存为)
PUT    /api/{type}/{name}    # 更新 (保存)
DELETE /api/{type}/{name}    # 删除
```

其中 `{type}` = chips / models / benchmarks / topologies

### 交互场景示例

**修改模型参数 (如 dtype)**:

```
1. 模型预设面板: 选择 deepseek_v3 -> 编辑 dtype 为 fp8
2. 模型预设面板: "另存为" -> 创建 deepseek_v3_fp8
3. Benchmark 面板: 切换模型引用为 deepseek_v3_fp8
4. Benchmark 面板: "另存为" -> 创建新 benchmark
```

**修改芯片参数 (如带宽)**:

```
1. 芯片预设面板: 选择 SG2262 -> 编辑 gmem.bandwidth_gbps 为 400
2. 芯片预设面板: "另存为" -> 创建 SG2262_400bw
3. 拓扑面板: 切换芯片引用为 SG2262_400bw
4. 拓扑面板: "保存"
```

### 参数扫描

参数扫描由前端完成，不影响预设文件：

```
1. 用户选好 benchmark + topology
2. 扫描面板: 选择参数和范围 (如 batch_size: [1, 2, 4, 8])
3. 前端生成 N 个完整配置 (resolve 引用 + 替换扫描值)
4. 提交 N 个独立仿真任务
5. 结果中保存每个任务的完整 config_snapshot
```

### 前端 UI 结构

```
┌─ 芯片预设 ──────────────────────────────────┐
│ [SG2262 v]  [保存] [另存为] [重载]            │
│ frequency / memory / compute_units / ...      │
└───────────────────────────────────────────────┘

┌─ 模型预设 ──────────────────────────────────┐
│ [deepseek_v3 v]  [保存] [另存为] [重载]       │
│ hidden_size / num_layers / moe / mla / ...    │
└───────────────────────────────────────────────┘

┌─ Benchmark ─────────────────────────────────┐
│ [deepseek_v3-S32K v]  [保存] [另存为] [重载]  │
│ model: deepseek_v3 (引用)                     │
│ batch_size / input_seq_length / ...           │
└───────────────────────────────────────────────┘

┌─ 拓扑配置 ──────────────────────────────────┐
│ [P1-R1-B1-C8 v]  [保存] [另存为] [重载]       │
│ chip: SG2262 (引用)                           │
│ pod_count / interconnect / comm_latency / ... │
└───────────────────────────────────────────────┘
```

## 7. 配置加载规则

**禁止使用默认值**。加载配置时如果必需字段找不到，必须抛出错误，
指明缺失的字段名和配置文件路径。详见 CLAUDE.md "配置参数加载规则"。

唯一例外：明确标注为"可选"的字段，且在 `_template.yaml` 中有文档说明。

## 8. 字段命名统一

所有配置文件、API 返回、前端显示、仿真引擎消费使用同一套字段名。
不做映射/转换，不维护两套命名。

当前存在的两套命名 (CHIPMathica 风格 vs llm_simulator 风格) 需要统一为一套，
前端和后端同步修改。具体统一方案待定。
