# 04. Tier6+Model 建模分析

本文分析现有 Tier6+Model 工具对 SG2262 C2C 方案的建模能力，识别差距并提出建模策略。

## 4.1 现有工具的 C2C 建模能力

### 当前拓扑模型

现有工具使用 **5 层静态层次结构**：

```
Pod -> Rack -> Board -> Chip -> Die
```

每层通过固定参数描述互联：

| 层级 | 参数 | 含义 |
|------|------|------|
| c2c | bandwidth_gbps, latency_us | 同板芯片间（Chip-to-Chip） |
| b2b | bandwidth_gbps, latency_us | 跨板（Board-to-Board） |
| r2r | bandwidth_gbps, latency_us | 跨机架（Rack-to-Rack） |
| p2p | bandwidth_gbps, latency_us | 跨 Pod（Pod-to-Pod） |

### 当前路由模型

现有路由基于**位置层次判断**，无显式路由表：

```python
if same_board:     -> c2c (NVLink-like)
elif same_rack:    -> b2b (PCIe-like)
elif same_pod:     -> r2r (InfiniBand-like)
else:              -> p2p (Ethernet-like)
```

### 当前通信建模

- **AllReduce**: 支持 Ring/Double Binary Tree/Halving-Doubling/Reduce-Broadcast
- **AllToAll**: 支持 Pairwise/Ring/Bruck
- **P2P**: 直接 Send/Recv
- **带宽模型**: 组内取最小带宽，延迟取最大
- **无拥塞建模**: 并发通信不影响单链路带宽

## 4.2 SG2262 C2C 方案 vs 现有模型的差距分析

### 差距 1: 两层互联结构 (L1/L2)

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| 互联层次 | L1 (cluster 内) + L2 (cluster 间) | 固定 4 层 (c2c/b2b/r2r/p2p) |
| L1 拓扑 | all2all/torus/ring/mesh/clos | 无显式拓扑类型 |
| L2 拓扑 | 仅 clos（交换机） | 无显式拓扑类型 |
| 路由 | CLE 两层路由 + itlv + 查找表 | 层次位置判断 |

**[建模]** 映射策略：
- SG2262 的 **L1 (cluster 内)** 可映射到现有模型的 **Board 层级**（c2c 链路）
- SG2262 的 **L2 (cluster 间)** 可映射到现有模型的 **Rack/Pod 层级**（r2r/p2p 链路）
- Cluster 概念可通过 Board 的 chips count 体现（最多 32 芯片）

但存在精度损失：
- 现有模型 Board 内所有芯片直连，而 SG2262 L1 可能是 ring/torus 等非全连接拓扑
- L1 非全连接拓扑中，不同芯片对之间的有效带宽和延迟不同

### 差距 2: 物理链路拓扑

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| 物理端口 | 8 组 x4 Link | 无端口概念 |
| 端口分配 | L1/L2 端口按拓扑分配 | 所有芯片等价连接 |
| 带宽聚合 | 多端口 itlv | 单一带宽参数 |
| 路径唯一性 | 严格单路径 | 无路径概念 |

**[建模]** 映射策略：
- 将 8 组 x4 Link 的总带宽（448 GB/s @ 112G）设为 c2c 带宽参数
- 对于非全连接拓扑（如 ring），需按实际连接关系计算有效带宽
  - Ring 拓扑 8 芯片：每芯片分配 2 个 port 给邻居，有效点对点带宽 = 2 x 56 = 112 GB/s
  - 但 AllReduce Ring 算法利用率接近 100%

### 差距 3: 交换机延迟

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| 交换机 | L1/L2 可有交换机 | 仅 switch_delay_us 参数 |
| 交换机拓扑 | 可多层 clos | 无交换机拓扑 |
| 芯片转发 | SG2262 可作为转发节点 | 无转发概念 |

**[建模]** 映射策略：
- L1 交换机延迟可纳入 c2c latency_us
- L2 交换机延迟可纳入 r2r/p2p latency_us
- 芯片转发额外延迟可通过增加 switch_delay_us 近似

### 差距 4: CDMA 多线程与搬运开销

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| DMA 引擎 | 4 CDMA/Die, 8 Thread/CDMA | 无 DMA 建模 |
| 搬运带宽 | 64 GB/s/CDMA | 直接使用链路带宽 |
| fence 开销 | 软件插入 fence 同步 | 无 |
| Send/Receive | 复杂握手流程 | 简单 P2P |

**[建模]** 映射策略：
- CDMA 带宽上限可作为链路有效带宽的约束
  - 若 CDMA 总带宽 < C2C 物理带宽，应取 CDMA 带宽
  - SG2262: CDMA 256 GB/s/Die < C2C 448 GB/s/Chip（双 Die），需分析实际瓶颈
- fence/同步开销可纳入通信启动延迟（start latency）

### 差距 5: 保序窗口与 Memory Consistency

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| 保序机制 | CHS(硬件保序) / CFS(fence) | 无保序建模 |
| 保序开销 | 窗口等待 + fence 同步 | 无 |
| Memory Protect | 16 组地址保护 | 无 |

**[建模]** 映射策略：
- 保序开销对大块数据传输影响较小（仅在同步点生效）
- 可通过调整通信启动延迟或通信效率因子间接建模
- 精确建模需要引入 fence 同步点的时间开销

### 差距 6: 在网计算 (All Reduce)

| 维度 | SG2262 C2C | 现有模型 |
|------|-----------|---------|
| 硬件 Reduce | 支持 add/max/min @ fp32/fp16/bp16 | 纯通信建模 |
| 在网计算 | 交换机/芯片转发时可做 Reduce | 仅端侧计算 |

**[建模]** 映射策略：
- 在网计算可通过提升 AllReduce 的 network_efficiency 参数近似
- 或通过自定义 AllReduce 算法参数降低通信量（数据量减半）

## 4.3 具体建模方案

### 方案一: 层次映射（最小改动）

将 SG2262 C2C 两层互联映射到现有 5 层模型，不修改代码：

```yaml
# SG2262 all2all+clos 拓扑示例: 4 cluster x 8 chip = 32 chips
name: "SG2262-L1All2All-L2Clos-32C"

pod_count: 1
racks_per_pod: 1

rack_config:
  boards:
    - chips:
        - name: "SG2262"
          preset_id: "sg2262"
          count: 8          # L1 cluster = 8 chips per board
      count: 4              # 4 clusters = 4 boards

hardware_params:
  chips:
    SG2262:
      name: "SG2262"
      compute_tflops_fp8: 768
      compute_tflops_bf16: 384
      memory_capacity_gb: 64
      memory_bandwidth_gbps: 12000
      # ...

  interconnect:
    c2c:
      bandwidth_gbps: 448    # L1: 8x4 @ 112G 全带宽
      latency_us: 0.2        # L1: 芯片直连延迟
    b2b:
      bandwidth_gbps: 400    # L2: 交换机互联带宽
      latency_us: 3.0        # L2: 含交换机延迟
    r2r:
      bandwidth_gbps: 400
      latency_us: 5.0
    p2p:
      bandwidth_gbps: 200
      latency_us: 10.0

  comm_latency_config:
    allreduce_algorithm: "ring"
    alltoall_algorithm: "pairwise"
    enable_compute_comm_overlap: true
    network_efficiency: 0.85
```

**映射关系**:

| SG2262 概念 | 映射到 | 说明 |
|-------------|--------|------|
| L1 Cluster (<=32 chips) | Board | 同 Board 芯片使用 c2c 链路 |
| L2 交换机 | Rack (b2b) | 跨 Board 使用 b2b 链路 |
| 8 组 x4 Link @ 112G | c2c.bandwidth_gbps = 448 | 物理带宽总量 |
| 交换机延迟 | b2b.latency_us | 含交换机转发开销 |

**优点**: 零代码改动，快速可用
**缺点**: 无法区分 L1 不同拓扑（all2all vs ring vs torus）的性能差异

### 方案二: 拓扑感知建模（中等改动）

引入 L1 拓扑类型参数，影响有效带宽计算：

```yaml
# 新增参数（扩展 comm_latency_config）
comm_latency_config:
  # 现有参数...
  l1_topology: "all2all"      # all2all / ring / torus / mesh / cube / clos
  l1_cluster_size: 8           # L1 cluster 内芯片数
  l1_ports_per_chip: 8         # 每芯片 C2C 端口数
  l1_serdes_rate_gbps: 112     # SerDes 速率
  l1_lanes_per_port: 4         # 每端口 lane 数
```

根据 L1 拓扑类型计算有效带宽：

| L1 拓扑 | 有效点对点带宽 | AllReduce 带宽利用率 |
|---------|---------------|-------------------|
| all2all (8 chips) | 每对 1 port = 56 GB/s | 接近 100% |
| ring (8 chips) | 每邻居 1 port = 56 GB/s | Ring: ~87.5% |
| torus (8 chips) | 每邻居 2 port = 112 GB/s | 接近 Ring |
| cube (8 chips) | 每邻居 1 port = 56 GB/s | 取决于路由 |
| clos (交换机) | 取决于交换机带宽 | 取决于算法 |

**优点**: 更准确反映不同 L1 拓扑的性能特征
**缺点**: 需修改 topology.py 的带宽计算逻辑

### 方案三: 全链路建模（大改动）

引入显式连接图和端口级建模：

```yaml
# 显式定义每个芯片的端口分配
port_allocation:
  - chip_id: 0
    ports:
      - port_id: 0
        type: "l1"
        connected_to: {chip_id: 1, port_id: 0}
        bandwidth_gbps: 56
        latency_us: 0.2
      - port_id: 1
        type: "l1"
        connected_to: {chip_id: 2, port_id: 1}
        # ...
      - port_id: 6
        type: "l2"
        connected_to: "switch_0"
        bandwidth_gbps: 56
        latency_us: 1.0
      - port_id: 7
        type: "l2"
        connected_to: "switch_0"
        bandwidth_gbps: 56
        latency_us: 1.0
```

**优点**: 完全精确建模 SG2262 C2C 行为
**缺点**: 改动量大，需重写 TopologyParser 和路由逻辑

## 4.4 建模精度 vs 复杂度评估

| 建模方面 | 方案一(映射) | 方案二(拓扑感知) | 方案三(全链路) | SG2262 重要性 |
|----------|------------|----------------|--------------|-------------|
| L1/L2 带宽 | 近似 | 精确 | 精确 | 高 |
| L1 拓扑影响 | 忽略 | 参数化 | 精确 | 中 |
| 交换机延迟 | 近似 | 近似 | 精确 | 中 |
| 端口分配 | 忽略 | 参数化 | 精确 | 低 |
| CDMA 瓶颈 | 忽略 | 可加入 | 精确 | 中 |
| 保序开销 | 忽略 | 参数化 | 精确 | 低 |
| 在网 Reduce | 效率因子 | 效率因子 | 精确 | 低 |
| 芯片转发 | 忽略 | 忽略 | 精确 | 低 |
| 代码改动量 | 无 | 中等 | 大 |  |

## 4.5 推荐建模路径

### 阶段一: 快速验证（方案一）

使用现有工具的层次映射，零代码改动：

1. 将 L1 cluster 映射为 Board（c2c 链路）
2. 将 L2 交换机映射为 Rack（b2b 链路）
3. 配置合适的带宽和延迟参数
4. 通过 `network_efficiency` 调节拓扑效率

适用场景：快速评估 SG2262 部署方案的大致性能

### 阶段二: 拓扑细化（方案二）

在通信延迟计算中引入 L1 拓扑感知：

1. 在 topology YAML 中增加 `l1_topology` 等参数
2. 在 `latency.py` 中根据 L1 拓扑类型调整有效带宽
3. 在 `topology.py` 中支持不同的端口分配策略

适用场景：需要对比不同 L1 拓扑（ring vs all2all vs torus）对推理性能的影响

### 阶段三: 精确建模（方案三）

引入完整的端口级连接图和 CLE 路由建模：

1. 扩展 TopologyParser 支持显式连接定义
2. 实现 CLE 路由算法（itlv/查找表）
3. 加入 CDMA 多线程调度建模
4. 加入保序窗口延迟建模

适用场景：需要精确预测特定拓扑下的通信性能瓶颈

## 4.6 关键建模参数对照表

供配置拓扑 YAML 时参考的 SG2262 参数：

| 参数 | SG2262 值 | 对应 YAML 字段 | 说明 |
|------|----------|---------------|------|
| C2C 总带宽 (112G) | 448 GB/s | `interconnect.c2c.bandwidth_gbps` | 8 x4 @ 112G |
| C2C 总带宽 (56G) | 224 GB/s | `interconnect.c2c.bandwidth_gbps` | 8 x4 @ 56G |
| C2C 延迟 | 0.2 us | `interconnect.c2c.latency_us` | 直连延迟 |
| 交换机延迟 | ~1.0 us | `comm_params.switch_delay_us` | L1/L2 交换机 |
| L1 cluster size | <=32 | Board chips count | all2all 最大规模 |
| 芯片数上限 | 1024 | 总 chip count | MAC ID 10 bit |
| CDMA 带宽/Die | 256 GB/s | (暂无直接对应) | 4 x 64 GB/s |
| CDMA Thread/Die | 32 | (暂无直接对应) | 4 x 8 |
| 保序窗口 | 8/12/32 | (暂无直接对应) | 三种模式 |
| 在网计算 | add/max/min | `network_efficiency` 调高 | 近似效果 |
