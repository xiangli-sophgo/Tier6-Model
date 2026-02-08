# Cycle 级精确建模设计方案

## 一、设计目标

### 1.1 核心目标

1. **Cycle 级精确性**: 建模 VOQ、Crossbar、iSLIP 调度器等微架构组件的完整状态机
2. **动态延迟计算**: 延迟随交换机内部状态 (队列深度、拥塞程度) 实时变化
3. **PFC/ECN 反压建模**: 模拟拥塞信号在多跳网络中的传播效应
4. **与 Tier6 拓扑集成**: 无缝融入现有 Pod-Rack-Board-Chip 层级拓扑

### 1.2 非目标

- 完整 TCP/IP 协议栈 (仅建模 RDMA/RoCE 相关行为)
- RTL 级精确 (不追求门级时序)
- 硬件功耗/热建模 (Phase 1 不涉及)

### 1.3 设计策略

采用**混合精度方案**: 交换机内部使用 cycle 级状态机，NIC/链路使用参数化模型。

理由: 交换机是延迟不确定性的主要来源 (零负载 ~250ns，incast ~50us)，而 NIC 和链路延迟相对固定。

---

## 二、微架构设计

### 2.1 整体架构

```
+----------------------------------------------------------------+
|                      Network Switch                             |
|                                                                 |
|  Input Port 0          +------------------+      Output Port 0  |
|  +----------+          |                  |      +----------+   |
|  | Parser   |          |    Crossbar      |      |  Output  |   |
|  | + Route  |--+------>|    Switching     |----->|  Queue   |   |
|  | Lookup   |  |       |    Fabric        |      |  + Sched |   |
|  +----------+  |       |    (N x N)       |      +----------+   |
|       |        |       |                  |           |         |
|       v        |       +------------------+           |         |
|  +----------+  |              ^                       |         |
|  | VOQ      |  |              |                       |         |
|  | Buffer   |--+-->  Scheduler/Arbiter  <-------------+         |
|  | (N queues)|       (iSLIP, 2-4 iter)                          |
|  +----------+                                                   |
|                                                                 |
|  Shared Buffer Pool (32-128 MB)                                 |
|  + Dynamic Threshold Controller                                 |
|  + ECN Marking Logic                                            |
|  + PFC Generation Logic                                         |
+----------------------------------------------------------------+
```

### 2.2 流水线设计

采用 **12 阶段流水线**，对应真实数据中心交换机 ASIC 的处理阶段:

```
Stage  1: PHY_RX          -- 物理层接收 (SerDes 解串)
Stage  2: PARSER          -- 帧解析 (Eth/IP/UDP/BTH)
Stage  3: INGRESS_MATCH   -- 入口匹配 (MAC/IP 查表, ACL)
Stage  4: TRAFFIC_MANAGER -- 流量管理 (QoS 分类, ECN 检查)
Stage  5: VOQ_ENQUEUE     -- VOQ 入队 (缓冲分配, 动态阈值检查)
Stage  6: ARBITER_REQ     -- 调度请求 (iSLIP Step 1: Request)
Stage  7: ARBITER_GRANT   -- 调度授权 (iSLIP Step 2: Grant)
Stage  8: ARBITER_ACCEPT  -- 调度接受 (iSLIP Step 3: Accept)
Stage  9: XBAR_TRAVERSE   -- Crossbar 穿越
Stage 10: EGRESS_PROCESS  -- 出口处理 (ECN 标记, PFC 检查)
Stage 11: OUTPUT_SCHEDULE  -- 输出调度 (DWRR/WFQ)
Stage 12: PHY_TX          -- 物理层发送 (串行化)
```

**周期数分配** (500 MHz 时钟, 2ns/cycle):

| 阶段 | 周期数 | 延迟 (ns) | 说明 |
|------|--------|-----------|------|
| PHY_RX | 2 | 4 | SerDes 解串 |
| PARSER | 1 | 2 | 硬件流水线解析 |
| INGRESS_MATCH | 2 | 4 | TCAM/Hash 查表 |
| TRAFFIC_MANAGER | 1 | 2 | QoS 分类 |
| VOQ_ENQUEUE | 1 | 2 | 写入缓冲 |
| ARBITER (3 steps) | 4-8 | 8-16 | iSLIP 2-4 次迭代 |
| XBAR_TRAVERSE | 1 | 2 | Crossbar 传输 |
| EGRESS_PROCESS | 1 | 2 | ECN/PFC 处理 |
| OUTPUT_SCHEDULE | 1 | 2 | DWRR 调度 |
| PHY_TX | 变长 | 30+ | 取决于包大小 |
| **总计 (零负载)** | **~15** | **~30 + 串行化** | 不含排队延迟 |

### 2.3 关键组件详细设计

#### VOQ 缓冲区

```python
class VOQBuffer:
    """虚拟输出队列缓冲区"""

    # 状态
    queues: dict[(int, int, int), deque]  # (input_port, output_port, priority) -> packet_queue
    current_depth: dict[(int, int, int), int]  # 每个 VOQ 的当前深度

    # 共享缓冲池
    shared_pool_total: int          # 总缓冲大小 (bytes)
    shared_pool_used: int           # 已使用 (bytes)
    reserved_per_voq: int           # 每个 VOQ 的保底缓冲
    alpha: float                    # 动态阈值系数

    def can_enqueue(self, voq_key, packet_size) -> bool:
        """动态阈值准入控制"""
        static_quota = self.reserved_per_voq
        shared_available = self.shared_pool_total - self.shared_pool_used
        dynamic_quota = self.alpha * shared_available
        max_allowed = static_quota + dynamic_quota
        return self.current_depth[voq_key] + packet_size <= max_allowed

    def enqueue(self, voq_key, packet):
        """入队 (必须先通过 can_enqueue 检查)"""
        self.queues[voq_key].append(packet)
        self.current_depth[voq_key] += packet.size
        self.shared_pool_used += packet.size

    def dequeue(self, voq_key) -> Packet:
        """出队"""
        packet = self.queues[voq_key].popleft()
        self.current_depth[voq_key] -= packet.size
        self.shared_pool_used -= packet.size
        return packet
```

#### iSLIP 调度器

```python
class iSLIPScheduler:
    """iSLIP 迭代轮询匹配调度器"""

    def __init__(self, num_ports, max_iterations=2):
        self.num_ports = num_ports
        self.max_iterations = max_iterations
        self.grant_ptrs = [0] * num_ports   # 每个输出端口的 Grant 指针
        self.accept_ptrs = [0] * num_ports  # 每个输入端口的 Accept 指针

    def schedule(self, voq_status) -> list[tuple[int, int]]:
        """
        执行 iSLIP 调度
        voq_status: dict[(in, out)] -> bool  (VOQ 是否非空)
        返回: 匹配对列表 [(input, output), ...]
        """
        matched_inputs = set()
        matched_outputs = set()
        matches = []

        for iteration in range(self.max_iterations):
            # Step 1: Request
            requests = {}  # output -> [input_list]
            for (inp, out), non_empty in voq_status.items():
                if non_empty and inp not in matched_inputs and out not in matched_outputs:
                    requests.setdefault(out, []).append(inp)

            # Step 2: Grant (round-robin from grant_ptr)
            grants = {}  # input -> [output_list]
            for out, inp_list in requests.items():
                if out in matched_outputs:
                    continue
                # 从 grant_ptr[out] 开始轮询
                selected = self._round_robin_select(
                    inp_list, self.grant_ptrs[out], self.num_ports
                )
                grants.setdefault(selected, []).append(out)

            # Step 3: Accept (round-robin from accept_ptr)
            for inp, out_list in grants.items():
                if inp in matched_inputs:
                    continue
                selected_out = self._round_robin_select(
                    out_list, self.accept_ptrs[inp], self.num_ports
                )
                matches.append((inp, selected_out))
                matched_inputs.add(inp)
                matched_outputs.add(selected_out)

                # 关键: 仅在被接受时更新指针
                self.grant_ptrs[selected_out] = (inp + 1) % self.num_ports
                self.accept_ptrs[inp] = (selected_out + 1) % self.num_ports

        return matches

    def _round_robin_select(self, candidates, ptr, n):
        """从 ptr 位置开始轮询选择"""
        for offset in range(n):
            candidate = (ptr + offset) % n
            if candidate in candidates:
                return candidate
        return candidates[0]
```

#### ECN/PFC 控制逻辑

```python
class CongestionController:
    """拥塞控制逻辑"""

    def __init__(self, config):
        self.ecn_kmin = config.ecn_kmin          # ECN 标记开始阈值
        self.ecn_kmax = config.ecn_kmax          # ECN 全部标记阈值
        self.ecn_pmax = config.ecn_pmax          # 最大标记概率
        self.pfc_threshold = config.pfc_threshold # PFC 触发阈值
        self.pfc_active = {}                      # (port, priority) -> bool

    def check_ecn(self, queue_depth, queue_capacity) -> bool:
        """检查是否需要标记 ECN"""
        usage = queue_depth / queue_capacity
        if usage < self.ecn_kmin:
            return False
        elif usage >= self.ecn_kmax:
            return True
        else:
            # 线性概率标记
            prob = self.ecn_pmax * (usage - self.ecn_kmin) / (self.ecn_kmax - self.ecn_kmin)
            return random.random() < prob

    def check_pfc(self, queue_depth, queue_capacity, port, priority) -> bool:
        """检查是否需要发送 PFC PAUSE"""
        usage = queue_depth / queue_capacity
        if usage >= self.pfc_threshold and not self.pfc_active.get((port, priority)):
            self.pfc_active[(port, priority)] = True
            return True  # 发送 PAUSE
        elif usage < self.pfc_threshold * 0.8 and self.pfc_active.get((port, priority)):
            self.pfc_active[(port, priority)] = False
            return False  # 发送 RESUME (通过 PAUSE time=0)
        return False
```

---

## 三、配置格式

### 3.1 交换机预设配置

新增目录: `backend/configs/switch_presets/*.yaml`

```yaml
# switch_presets/tor_400g_32port.yaml
name: "ToR 400G 32-Port Switch"
type: "top_of_rack"

# 基本参数
num_ports: 32
port_bandwidth_gbps: 400
backplane_bandwidth_tbps: 12.8

# 微架构参数
microarchitecture:
  clock_frequency_mhz: 500
  pipeline_stages: 12

  # 输入缓冲 (VOQ)
  input_buffer:
    architecture: "voq"
    total_buffer_mb: 32
    sharing_mode: "dynamic_threshold"
    reserved_per_voq_kb: 128
    alpha: 1.0

  # 交换矩阵
  crossbar:
    type: "unbuffered"
    internal_speedup: 1.0

  # 调度器
  scheduler:
    algorithm: "islip"
    max_iterations: 2

  # 输出队列
  output_queue:
    enabled: true
    depth_packets: 64
    scheduling_policy: "dwrr"

# 拥塞控制
congestion_control:
  ecn:
    kmin_ratio: 0.50      # 50% 缓冲使用率开始标记
    kmax_ratio: 0.80      # 80% 全部标记
    pmax: 0.2             # 最大标记概率
  pfc:
    threshold_ratio: 0.85  # 85% 触发 PFC
    watchdog_ms: 10        # PFC 死锁检测超时

# 流量控制
flow_control:
  type: "credit_based"
  credit_delay_cycles: 4
```

### 3.2 拓扑配置扩展

在现有 `topologies/*.yaml` 中新增交换机节点:

```yaml
name: "P1-R4-B32-C256-with-switches"
pod_count: 1
racks_per_pod: 4

# 交换机定义
switches:
  - name: "tor_rack0"
    type: "top_of_rack"
    location: {pod: 0, rack: 0}
    preset_id: "tor_400g_32port"

  - name: "agg_pod0"
    type: "aggregation"
    location: {pod: 0}
    preset_id: "agg_800g_64port"

# 连接定义
connections:
  # Board 间通过 ToR 交换机
  - from: {pod: 0, rack: 0, board: 0}
    to: {pod: 0, rack: 0, board: 1}
    type: "via_switch"
    switch_name: "tor_rack0"
    bandwidth_gbps: 400

  # Rack 间通过汇聚交换机
  - from: {pod: 0, rack: 0}
    to: {pod: 0, rack: 1}
    type: "via_switch"
    switch_name: "agg_pod0"
    bandwidth_gbps: 800
```

---

## 四、仿真引擎集成

### 4.1 交换机状态机

每个仿真周期，交换机状态机执行以下操作:

```
每个 cycle:
  1. PHY_RX: 接收到达的包，写入解析队列
  2. PARSER + MATCH: 解析包头，查表确定出端口
  3. TRAFFIC_MGR: QoS 分类，确定优先级
  4. VOQ_ENQUEUE: 动态阈值检查 -> 入队或丢弃
  5. iSLIP: 对所有非空 VOQ 执行调度匹配
  6. XBAR: 匹配成功的包穿越 Crossbar
  7. EGRESS: ECN 标记检查，PFC 检查
  8. OUTPUT: 输出队列调度 (DWRR)，串行化发送
  9. CREDIT: 处理 Credit 返回，更新 Credit 计数器
  10. STATS: 更新统计计数器
```

### 4.2 事件驱动集成

在 `simulator.py` 中新增交换机事件类型:

```python
class EventType(str, Enum):
    # 现有事件
    COMPUTE_START = "compute_start"
    COMPUTE_END = "compute_end"
    COMM_START = "comm_start"
    COMM_END = "comm_end"

    # 交换机事件
    SWITCH_PACKET_ARRIVE = "switch_packet_arrive"
    SWITCH_PACKET_ENQUEUE = "switch_packet_enqueue"
    SWITCH_SCHEDULE = "switch_schedule"
    SWITCH_XBAR_TRANSFER = "switch_xbar_transfer"
    SWITCH_PACKET_DEPART = "switch_packet_depart"
    SWITCH_PFC_PAUSE = "switch_pfc_pause"
    SWITCH_PFC_RESUME = "switch_pfc_resume"
    SWITCH_ECN_MARK = "switch_ecn_mark"
```

### 4.3 延迟计算接口

```python
def calc_switch_latency(
    data_size_gb: float,
    switch: Switch,
    src_port: int,
    dst_port: int,
) -> SwitchLatencyResult:
    """
    通过交换机状态机计算动态延迟

    与旧版公式计算的区别:
    - 旧版: 使用 M/M/1 排队论近似，延迟是静态函数
    - 新版: 基于 VOQ 实际队列深度和 iSLIP 调度结果，延迟随状态变化
    """
    # 将大消息分割为包
    packets = segment_to_packets(data_size_gb, mtu=9000)

    total_latency_ns = 0
    for packet in packets:
        # 1. 入口处理 (固定延迟)
        ingress_ns = switch.pipeline_latency_ns("ingress")

        # 2. 排队延迟 (动态 -- 取决于当前 VOQ 深度)
        voq_key = (src_port, dst_port, packet.priority)
        queue_ns = switch.voq.current_depth[voq_key] * switch.per_packet_service_time_ns

        # 3. 调度延迟 (动态 -- 取决于端口竞争)
        schedule_ns = switch.scheduler.estimate_wait_cycles(src_port, dst_port) * switch.clock_period_ns

        # 4. Crossbar + 出口 (固定延迟)
        xbar_egress_ns = switch.pipeline_latency_ns("xbar_egress")

        # 5. 串行化
        serialization_ns = packet.size_bits / (switch.port_bandwidth_gbps * 1e9) * 1e9

        packet_latency = ingress_ns + queue_ns + schedule_ns + xbar_egress_ns + serialization_ns
        total_latency_ns = max(total_latency_ns, packet_latency)

        # 更新交换机状态
        switch.process_packet(packet, src_port, dst_port)

    return SwitchLatencyResult(
        total_latency_ns=total_latency_ns,
        queue_latency_ns=queue_ns,
        # ... 详细分解
    )
```

---

## 五、性能指标输出

### 5.1 交换机级指标

```python
{
    "switch_name": "tor_rack0",
    "throughput": {
        "total_gbps": 11520.5,
        "utilization_ratio": 0.90,
        "backpressure_ratio": 0.05
    },
    "latency": {
        "average_ns": 285.3,
        "p50_ns": 120.5,
        "p95_ns": 680.2,
        "p99_ns": 1250.8
    },
    "buffer": {
        "peak_usage_mb": 12.5,
        "average_usage_mb": 4.8,
        "overflow_count": 125,
        "drop_rate": 0.001
    },
    "congestion": {
        "ecn_marked_packets": 45230,
        "pfc_pause_count": 12,
        "pfc_total_pause_us": 85.3
    },
    "scheduler": {
        "average_iterations": 1.85,
        "match_efficiency": 0.982
    }
}
```

### 5.2 路径延迟分解

```python
{
    "path": "Chip(0,0,0,0) -> Chip(0,0,5,2)",
    "hops": [
        {
            "type": "chip_to_switch",
            "from": "Chip(0,0,0,0)",
            "to": "Switch(tor_rack0)",
            "latency_breakdown": {
                "serialization_ns": 30.0,
                "switch_ingress_ns": 10.0,
                "switch_queue_ns": 250.0,
                "switch_schedule_ns": 16.0,
                "switch_xbar_ns": 2.0,
                "switch_egress_ns": 4.0,
                "total_ns": 312.0
            }
        },
        {
            "type": "switch_to_chip",
            "from": "Switch(tor_rack0)",
            "to": "Chip(0,0,5,2)",
            "latency_breakdown": {
                "serialization_ns": 30.0,
                "propagation_ns": 8.0,
                "total_ns": 38.0
            }
        }
    ],
    "total_latency_ns": 350.0
}
```

---

## 六、实施路线图

### Phase 1: 数据结构与配置 (1 周)

- 定义 `SwitchConfig`, `Switch`, `VOQBuffer`, `iSLIPScheduler` 类
- 实现交换机预设配置加载 (`switch_presets/*.yaml`)
- 扩展拓扑解析器支持交换机节点
- 单元测试: 配置加载、VOQ 基本操作

### Phase 2: Cycle 级状态机 (2 周)

- 实现 12 阶段流水线状态机
- 实现 iSLIP 调度器 (支持 1-4 次迭代)
- 实现共享缓冲池 + 动态阈值算法
- 集成到事件驱动仿真器 (`simulator.py`)
- 单元测试: 零负载延迟、VOQ HOL 消除、iSLIP 公平性

### Phase 3: 拥塞控制 (2 周)

- 实现 ECN 标记逻辑 (概率标记)
- 实现 PFC PAUSE/RESUME 生成与处理
- 实现 Credit-based 流控
- 实现 PFC 死锁检测 (Watchdog)
- 单元测试: ECN 阈值行为、PFC 反压传播

### Phase 4: 集成测试与可视化 (2 周)

- Rack 内 Board 间通信 (经由 ToR) 集成测试
- Rack 间通信 (经由汇聚交换机) 集成测试
- AllReduce Incast 场景验证
- Gantt 图扩展 (显示交换机排队/调度活动)
- 前端交换机性能面板

### Phase 5: 验证与优化 (1 周)

- 对比排队论模型 (M/M/1) 验证精度提升
- 对比 SimAI/NS-3 的参考结果
- 仿真速度优化 (热路径优化、批量调度)
- 文档完善
