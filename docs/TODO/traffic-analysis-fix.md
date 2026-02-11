# 拓扑流量分析修复方案

## 问题概述

当前拓扑流量分析数据不正确。后端有 `TrafficAnalyzer` 但从未被调用（死代码），实际传给前端的数据来自 `compat.py` 中的启发式估算函数，不是真实仿真数据。

## 现状诊断

### 当前数据流（断裂的）

```
L3 ExecPlan
  CommInstance: chip_ids[], path_key     <-- 有芯片 ID 和链路类型
       |
       v
L4 CommEvaluator
  StepMetrics: t_comm, bytes_read/write  <-- 有通信时间和字节数
  meta: {path_key, participants, comm_type, tp, ep}
       |                                 <-- chip_ids 在这里丢失了
       v
L0 compat.py: generate_link_traffic_stats()
  启发式估算，不使用 TrafficAnalyzer    <-- 用猜测代替真实数据
       |
       v
Frontend: TopologyTrafficChart
  LinkTrafficStats[]                     <-- 展示不准确的流量数据
```

### 问题 1: TrafficAnalyzer 是死代码

`L5_reporting/traffic_analysis.py` 中的 `TrafficAnalyzer.analyze()` 期望:
```python
exec_plan.prefill_steps  # ExecPlan 上不存在
exec_plan.decode_steps   # ExecPlan 上不存在
```

实际 `ExecPlan`（`L3_mapping/plan/exec_plan.py`）的结构是:
```python
@dataclass
class ExecPlan:
    tile_config: ...
    kernel_config: ...
    timeline: list
    instances: list[OpInstance | CommInstance]  # 不是 prefill_steps/decode_steps
    binding: ...
    precedence: list
    ...
```

`TrafficAnalyzer` 设计时假设的 API 与实际完全不匹配。

### 问题 2: compat.py 的启发式估算

`generate_link_traffic_stats()` 的工作方式:
1. 从 `op_id` 字符串猜测通信类型（如包含 "allreduce" 就认为是 AllReduce）
2. 创建合成的 `chip_0, chip_1, ...` ID，可能与真实拓扑不匹配
3. 用硬编码的 Ring 模式分配流量
4. 找不到通信数据时，用总字节数的百分比估算（15% TP, 10% EP, 5% PP）
5. 使用硬编码的默认带宽值（448 Gbps c2c, 450 Gbps b2b）

这意味着前端展示的流量数据是**粗略近似值**，不反映实际仿真计算。

### 问题 3: L4 评估丢失了芯片级信息

L3 的 `CommInstance` 包含 `chip_ids`（参与通信的芯片列表），但传入 L4 `CommEvaluator` 后，只保留了 `participants`（数量）和 `path_key`（链路类型），具体的芯片 ID 丢失了。

## 前端期望的数据格式

`TopologyTrafficChart.tsx` 期望接收 `LinkTrafficStats[]`:

```typescript
interface LinkTrafficStats {
  source: string;                              // 源芯片 ID
  target: string;                              // 目标芯片 ID
  trafficMb: number;                           // 累计流量 (MB)
  bandwidthGbps: number;                       // 链路带宽 (Gbps)
  latencyUs: number;                           // 链路延迟 (us)
  utilizationPercent: number;                  // 带宽利用率 (0-100)
  linkType: string;                            // 链路类型: c2c/b2b/r2r/p2p
  contributingTasks: string[];                 // 贡献的任务 ID
  taskTypeBreakdown: Record<string, number>;   // 按通信类型分解流量 (MB)
}
```

可视化特性:
- 热力图着色: 0-30% 绿色, 30-60% 黄色, 60-80% 橙色, 80-100% 红色
- 线宽映射: 流量越大线越粗 (2-6px)
- 瓶颈标记: >80% 利用率的链路标红

## TrafficAnalyzer 已定义的数据结构

`traffic_analysis.py` 中已定义但未使用的结构:

```python
@dataclass
class LinkTraffic:
    src: str                              # 源设备 ID
    dst: str                              # 目标设备 ID
    link_type: LinkType                   # c2c|b2b|r2r|p2p
    total_bytes: int                      # 总流量
    total_time_us: float                  # 总通信时间
    bandwidth_gbps: float                 # 链路带宽
    utilization: float                    # 利用率 (0-1)
    comm_breakdown: dict[str, int]        # 按通信类型分解

@dataclass
class DeviceTraffic:
    device_id: str
    total_send_bytes: int
    total_recv_bytes: int
    send_breakdown: dict[str, int]
    recv_breakdown: dict[str, int]

@dataclass
class TrafficReport:
    total_bytes: int
    total_time_us: float
    links: list[LinkTraffic]
    devices: list[DeviceTraffic]
    comm_breakdown: dict[str, int]        # 按通信类型汇总
    phase_breakdown: dict[str, int]       # 按阶段 (prefill/decode) 汇总
```

## 实现计划

### Step 1: 在 L4 评估时保留芯片级通信信息

**修改文件**: `L4_evaluation/evaluators/comm.py`

当前 `CommEvaluator.evaluate()` 接收 `attrs` dict，其中有 `path_key` 和 `participants`。需要增加 `chip_ids` 和 `comm_type` 的传递。

```python
# 当前 StepMetrics.meta 内容:
meta = {
    "evaluator": "comm",
    "op_type": op_type,
    "path_key": path_key,
    "participants": participants,
    ...
}

# 需要增加:
meta = {
    ...
    "chip_ids": attrs.get("chip_ids"),       # [0, 1, 2, 3, ...] 参与的芯片索引
    "comm_type": attrs.get("comm_type"),     # "allreduce" / "alltoall" / "p2p" / ...
    "comm_reason": attrs.get("reason"),      # "tp_sync" / "ep_dispatch" / "pp_transfer" / ...
    "comm_bytes": comm_bytes,                # 实际通信字节数
}
```

### Step 2: 确保 L3 传递完整的通信属性

**修改文件**: `L3_mapping/` 相关调度模块

检查 L3 生成 `CommInstance` 和传入 L4 评估的过程，确保以下属性被正确传递:

| 属性 | 来源 | 用途 |
|------|------|------|
| `chip_ids` | CommInstance.chip_ids | 参与通信的芯片列表 |
| `path_key` | CommInstance.path_key | 链路类型 (c2c/b2b/r2r/p2p) |
| `comm_type` | 从 op_type 推断 | allreduce/allgather/reducescatter/all2all/p2p |
| `comm_reason` | 从并行策略推断 | tp_sync/ep_dispatch/pp_transfer/dp_allreduce/sp_* |
| `comm_bytes` | 从工作负载计算 | 实际传输字节数 |

### Step 3: 重写 TrafficAnalyzer 适配实际数据结构

**修改文件**: `L5_reporting/traffic_analysis.py`

将 `analyze()` 方法改为接收 `StepMetrics[]` 而不是 `ExecPlan`:

```python
class TrafficAnalyzer:
    def analyze(
        self,
        step_metrics: list[StepMetrics],
        topology_config: dict,           # 拓扑配置，用于获取链路带宽
        chip_id_mapping: dict[int, str], # 芯片索引 -> 物理设备 ID 映射
    ) -> TrafficReport:
        """从评估结果的 step_metrics 中提取流量信息"""

        for step in step_metrics:
            if step.t_comm <= 0:
                continue

            chip_ids = step.meta.get("chip_ids")
            comm_type = step.meta.get("comm_reason", "unknown")
            comm_bytes = step.meta.get("comm_bytes", step.bytes_read)
            path_key = step.meta.get("path_key", "c2c")

            if chip_ids is None or len(chip_ids) < 2:
                continue

            # 将芯片索引转为物理设备 ID
            device_ids = [chip_id_mapping[cid] for cid in chip_ids]

            # 根据通信算法分配每条链路的流量
            self._distribute_traffic(
                device_ids=device_ids,
                comm_bytes=comm_bytes,
                comm_time_us=step.t_comm * 1000,  # ms -> us
                comm_type=comm_type,
                path_key=path_key,
            )

        return self._build_report()
```

#### 流量分配策略 `_distribute_traffic()`

根据不同的通信模式，将总流量分配到具体的设备对:

```python
def _distribute_traffic(self, device_ids, comm_bytes, comm_time_us, comm_type, path_key):
    n = len(device_ids)

    if "p2p" in comm_type or n == 2:
        # P2P: 直接 src -> dst
        self._add_link_traffic(device_ids[0], device_ids[1], comm_bytes, ...)

    elif "allreduce" in comm_type or "allgather" in comm_type or "reducescatter" in comm_type:
        # Ring 算法: 每个设备发送到下一个设备
        per_device_bytes = comm_bytes * (n - 1) // n  # Ring AllReduce 系数
        per_device_time = comm_time_us / n
        for i in range(n):
            src = device_ids[i]
            dst = device_ids[(i + 1) % n]
            self._add_link_traffic(src, dst, per_device_bytes, per_device_time, ...)

    elif "all2all" in comm_type:
        # All-to-All: 每对设备之间都有流量
        per_pair_bytes = comm_bytes // (n * (n - 1))
        per_pair_time = comm_time_us / (n * (n - 1))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self._add_link_traffic(device_ids[i], device_ids[j], per_pair_bytes, ...)
```

### Step 4: 构建芯片 ID 映射

**修改/新增**: 在 L0 或 L3 层构建映射

仿真中使用的芯片索引 (0, 1, 2, ...) 需要映射到拓扑中的物理设备 ID，格式为 `pod_rack_board_chip`:

```python
def build_chip_id_mapping(topology_config: dict) -> dict[int, str]:
    """构建芯片索引 -> 物理设备 ID 映射"""
    mapping = {}
    chip_index = 0
    for pod_idx in range(topology_config["pod_count"]):
        for rack_idx in range(topology_config["racks_per_pod"]):
            for board in topology_config["rack_config"]["boards"]:
                for board_idx in range(board["count"]):
                    for chip in board["chips"]:
                        for chip_idx in range(chip["count"]):
                            device_id = f"{pod_idx}_{rack_idx}_{board_idx}_{chip_idx}"
                            mapping[chip_index] = device_id
                            chip_index += 1
    return mapping
```

这个映射需要与前端 `TopologyGraph` 组件使用的 ID 格式保持一致。

### Step 5: 集成到仿真流水线

**修改文件**: `L0_entry/engine.py`

在仿真完成后调用 TrafficAnalyzer:

```python
# L4 评估完成后:
engine_result = evaluation_engine.run(...)

# 构建芯片 ID 映射
chip_id_mapping = build_chip_id_mapping(topology_config)

# 流量分析
traffic_analyzer = TrafficAnalyzer()
traffic_report = traffic_analyzer.analyze(
    step_metrics=engine_result.step_metrics,
    topology_config=topology_config,
    chip_id_mapping=chip_id_mapping,
)

# 转换为前端格式
link_traffic_stats = convert_traffic_report_to_stats(traffic_report, topology_config)
```

### Step 6: 格式转换 - TrafficReport -> LinkTrafficStats

**修改文件**: `L0_entry/compat.py`

替换现有的 `generate_link_traffic_stats()` 启发式函数:

```python
def convert_traffic_report_to_stats(
    report: TrafficReport,
    topology_config: dict,
) -> list[dict]:
    """将 TrafficReport 转换为前端 LinkTrafficStats 格式"""
    stats = []
    for link in report.links:
        # 从拓扑配置获取链路带宽和延迟
        interconnect = topology_config["hardware_params"]["interconnect"]
        link_config = interconnect.get(link.link_type.value, {})

        stats.append({
            "source": link.src,
            "target": link.dst,
            "trafficMb": link.total_bytes / (1024 * 1024),
            "bandwidthGbps": link_config.get("bandwidth_gbps", link.bandwidth_gbps),
            "latencyUs": link_config.get("latency_us", 0),
            "utilizationPercent": link.utilization * 100,
            "linkType": link.link_type.value,
            "contributingTasks": list(link.comm_breakdown.keys()),
            "taskTypeBreakdown": {
                k: v / (1024 * 1024)  # bytes -> MB
                for k, v in link.comm_breakdown.items()
            },
        })
    return stats
```

### Step 7: 替换 compat.py 中的旧实现

将 `generate_link_traffic_stats()` 的调用替换为新的 `convert_traffic_report_to_stats()`，删除旧的启发式代码。

## 数据流修复后

```
L3 ExecPlan
  CommInstance: chip_ids[], path_key, comm_bytes, comm_type
       |
       v
L4 CommEvaluator
  StepMetrics.meta: {chip_ids, comm_type, comm_reason, comm_bytes, path_key}
       |
       v
L5 TrafficAnalyzer.analyze(step_metrics, topology, chip_mapping)
  TrafficReport: links[], devices[], comm_breakdown, phase_breakdown
       |
       v
L0 convert_traffic_report_to_stats()
  LinkTrafficStats[]: {source, target, trafficMb, utilizationPercent, linkType, ...}
       |
       v
Frontend: TopologyTrafficChart
  正确的拓扑流量热力图可视化
```

## 验证方法

### 测试场景 1: TP=8 AllReduce (单板内)

8 芯片在同一块板上做 TP AllReduce:
- 预期: 7 条 c2c 链路，Ring 模式，每条流量 = total_bytes * 7/8
- 利用率 = 流量 / (带宽 * 通信时间)

### 测试场景 2: PP=4 P2P (跨板)

4 个 PP Stage 之间的点对点通信:
- 预期: 3 条 b2b 链路，单向传输
- 流量 = 激活值大小 (batch * hidden_size * dtype_bytes)

### 测试场景 3: EP=4 All-to-All (MoE 路由)

4 组 EP 之间的 All-to-All:
- 预期: 12 条链路 (4*3)，每对之间有流量
- 流量 = token_count * expert_dim * dtype_bytes / ep

### 测试场景 4: 混合并行 TP=4, PP=2, EP=2

验证不同并行策略的流量正确叠加在同一拓扑链路上。

## 涉及文件

| 文件 | 修改类型 |
|------|----------|
| `L4_evaluation/evaluators/comm.py` | **修改** - StepMetrics.meta 增加 chip_ids 等字段 |
| `L3_mapping/` 调度相关 | **修改** - 确保通信属性完整传递到 L4 |
| `L5_reporting/traffic_analysis.py` | **重写** - 适配实际数据结构 |
| `L0_entry/engine.py` | **修改** - 集成 TrafficAnalyzer |
| `L0_entry/compat.py` | **修改** - 替换启发式 generate_link_traffic_stats() |
| `L3_mapping/plan/exec_plan.py` | **可能修改** - CommInstance 增加 comm_bytes/comm_type |
