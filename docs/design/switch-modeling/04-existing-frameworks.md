# 现有仿真框架深度分析

本文档分析了 4 个与交换机建模相关的仿真框架，评估其技术方案、优劣和可借鉴之处。

---

## 一、SimAI (阿里巴巴, NSDI 2025)

### 1.1 概述

SimAI 是阿里巴巴发表在 NSDI 2025 的全栈 AI 训练/推理仿真器。

- **GitHub**: https://github.com/aliyun/SimAI
- **License**: Apache 2.0
- **验证精度**: 98.1% 平均对齐度 (A100/H100 集群)

### 1.2 架构

SimAI 由 4+1 个组件组成:

```
[AICB: 工作负载生成] --> workload.txt
          |
[astra-sim-alibabacloud: 核心引擎] <--> [SimCCL: NCCL 分解]
          |
  sim_send() / sim_recv()
          |
[ns-3-alibabacloud: 包级网络仿真]
          |
[vidur-alibabacloud: 推理调度] (v1.5+)
```

**AICB (AI Communication Benchmark)**: 工作负载生成层
- 支持 LLaMA, GPT, Mistral MoE, DeepSeek, Qwen 等模型
- 通过 AIOB 子模块 profile 真实 GPU kernel 执行时间
- 输出格式: 每层的计算时间 (ns)、通信类型、数据量

**SimCCL**: 修改版 NCCL，将集合通信分解为点对点原语
- 支持 Ring, Tree (Double Binary Tree), NVLS 算法
- 支持 PXN (PCI x NVLink) 跨节点通信

**astra-sim-alibabacloud**: 基于 Georgia Tech ASTRA-sim 扩展的核心引擎

**ns-3-alibabacloud**: 基于 ns3-rdma/HPCC 的包级网络仿真

### 1.3 交换机模型 (关键)

SimAI 的交换机模型源自 Broadcom ASIC 风格，工作在**缓冲/队列级别**，不是微架构级别。

**已建模内容**:
- **SwitchMmu**: 共享缓冲池 (默认 32 MB)，入口/出口缓冲记账
- **ECN 标记**: 概率标记 (KMIN/KMAX/PMAX 参数)，带宽相关阈值
- **PFC 生成**: 静态/动态阈值触发 PAUSE 帧
- **ECMP 转发**: 5-tuple Hash 负载均衡
- **监控**: 队列深度、带宽、速率、CNP 监控

**未建模内容**:
- 无 VOQ (Virtual Output Queues)
- 无 Crossbar 结构和仲裁器 (iSLIP 等)
- 无流水线阶段 (Parser, Match-Action 等)
- 交换机内部视为瞬时转发
- 无 SerDes/FEC 延迟

### 1.4 关键配置参数

```
BUFFER_SIZE            32              # 每交换机缓冲 (MB)
ENABLE_QCN             1               # 启用 ECN
USE_DYNAMIC_PFC_THRESHOLD  1           # 动态 PFC 阈值
PACKET_PAYLOAD_SIZE    9000            # Jumbo Frame
CC_MODE                1               # 1=DCQCN, 3=HPCC, 7=TIMELY

# ECN 阈值 (带宽相关)
KMAX_MAP  400Gbps:3200  (bytes)
KMIN_MAP  400Gbps:800   (bytes)
PMAX_MAP  400Gbps:0.2

# DCQCN 参数
RATE_AI    50 Mb/s        # 加性增
RATE_HAI   100 Mb/s       # 快速增
MIN_RATE   100 Mb/s       # 最低速率
RP_TIMER   900             # 速率增加定时器
```

### 1.5 性能与局限

| 指标 | 数值 |
|------|------|
| 验证精度 | 98.1% (A100/H100) |
| NS-3 模式速度 | ~2 小时 / 128 GPU / 1 iteration |
| 多线程加速 | 23x (lock-free) |
| 分析模式速度 | 秒级 (100-1000x 快于 NS-3) |

**局限**:
- 交换机模型缺乏微架构精度 (无 VOQ/Crossbar/调度器)
- NS-3 模式对千卡规模不实际
- 分析模式完全不建模网络拥塞
- 强依赖 NVIDIA NCCL，通用性受限

### 1.6 可借鉴之处

- **SwitchMmu 缓冲管理**: 动态阈值算法的参数配置方案
- **ECN 阈值映射**: 带宽相关的 KMIN/KMAX 配置
- **DCQCN 参数**: 真实集群验证过的拥塞控制参数
- **多线程优化**: lock-free 全局上下文共享

---

## 二、ASTRA-sim 2.0 (Georgia Tech/Meta/Intel, ISPASS 2023)

### 2.1 概述

ASTRA-sim 是分布式 AI 训练端到端仿真器，核心特点是三层架构和可插拔网络后端。

- **GitHub**: https://github.com/astra-sim/astra-sim
- **验证精度**: 5% 平均误差 (分析后端, V100)
- **贡献者**: Georgia Tech, Meta, Intel, AMD, NVIDIA, HPE

### 2.2 三层架构

```
+-------------------------------------------+
| Workload Layer (Chakra ET 格式)            |
| - DAG 表示: 计算/通信/内存节点             |
| - 支持 TP, PP, DP, EP 并行策略            |
+-------------------------------------------+
| System Layer (AstraCCL)                    |
| - 集合通信算法: Ring, DoubleBinaryTree,   |
|   HalvingDoubling, AllToAll               |
| - 分解为 sim_send() / sim_recv()          |
+-------------------------------------------+
| Network Layer (可插拔后端)                  |
| - Analytical (快速, ~5% 误差)             |
| - NS-3 (包级)                             |
| - Garnet (cycle-accurate NoC)             |
| - htsim (Ultra Ethernet Consortium)       |
+-------------------------------------------+
```

### 2.3 网络后端对比

| 后端 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| **Analytical** | ~5% 误差 | 756x 快于 Garnet | 大规模设计空间探索 |
| **NS-3** | 高 (包级) | 中等 | 数据中心协议研究 |
| **Garnet** | 极高 (cycle-accurate) | 慢 | NoC 微架构研究 |
| **htsim** | 高 (包级) | 中等 | UEC 协议研究 |

### 2.4 Analytical 后端

使用多维拓扑构建块:

```yaml
topologies-per-dim: ["Ring", "Switch"]     # 维度 0: Ring, 维度 1: Switch
units-count: [8, 4]                        # 每维度 NPU 数
link-latency: [500, 5000]                  # 延迟 (ns)
link-bandwidth: [50, 25]                   # 带宽 (GB/s)
```

交换机在 Analytical 后端中只是带宽/延迟抽象，无微架构建模。

### 2.5 Garnet 后端 (Cycle-Accurate)

Garnet 2.0 来自 gem5，提供 5 阶段路由器流水线:

| 阶段 | 功能 |
|------|------|
| BW (Buffer Write) | 接收 flit 写入 VC |
| RC (Route Compute) | 计算输出端口 |
| SA (Switch Allocation) | 2-phase 分离式分配 |
| VS (VC Selection) | 选择输出 VC |
| ST (Switch Traversal) | 穿越 Crossbar |

**但 Garnet 是 NoC 仿真器，不适用于数据中心交换机**:

| 方面 | Garnet (NoC) | 数据中心交换机 |
|------|-------------|--------------|
| 规模 | 片上 sub-mm | 机架/数据中心级 |
| 缓冲 | 1-5 flit/VC (~0.1KB) | MB 级共享缓冲 |
| 端口数 | 4-8 (2D mesh) | 32-128 |
| 流控 | Credit/VC | PFC, ECN, DCQCN |
| 包大小 | 16 bytes flit | 1500-9000 bytes MTU |
| 多播 | 不支持 | IGMP snooping |

### 2.6 可借鉴之处

- **AstraNetworkAPI 接口设计**: 干净的 sim_send()/sim_recv() + 回调模式
- **多维拓扑概念**: 维度化的拓扑构建 (类似 Tier6 的 c2c/b2b/r2r/p2p)
- **集合通信分解**: Ring, DoubleBinaryTree, HalvingDoubling 算法实现
- **可插拔后端架构**: 允许不同精度的网络模型互换

---

## 三、BookSim 2.0 (Stanford, ISPASS 2013)

### 3.1 概述

BookSim 是 Stanford 开发的 cycle-accurate NoC 仿真器，是学术界最广泛使用的互联网络仿真器。

- **GitHub**: https://github.com/booksim/booksim2
- **代码量**: ~15K LOC C++
- **历史**: ~15 年，多次验证对比 RTL

### 3.2 路由器流水线

BookSim 的 IQ Router 实现 5 阶段流水线:

```
Stage 1: BW (Buffer Write)      -- flit 写入 VC 缓冲
Stage 2: RC (Route Compute)     -- 计算输出端口
Stage 3: VA (VC Allocation)     -- 分配虚拟通道 (iSLIP)
Stage 4: SA (Switch Allocation) -- 分配交换资源 (iSLIP)
Stage 5: ST (Switch Traversal)  -- 穿越 Crossbar
       + LT (Link Traversal)   -- 穿越物理链路 (下一 cycle)
```

### 3.3 iSLIP 实现细节

BookSim 中 iSLIP 是 VA 和 SA 的默认分配器 (`src/allocators/islip.cpp`):

**指针更新规则 (关键)**:
- Grant 指针**仅在授权被接受时**更新
- 未被接受的授权不更新指针
- 这创造了**去同步化效应** -- 各输出仲裁器指针逐渐分散，大幅减少碰撞

**吞吐量保证**:
- 均匀流量: 100% 吞吐 (可证明收敛到最大匹配)
- 非均匀流量: 高吞吐 (竞争性接近 maximum-weight matching)
- 公平性: Round-robin 防止饥饿

### 3.4 完整配置参数

**路由器参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_vcs` | 16 | 每物理通道的虚拟通道数 |
| `vc_buf_size` | 8 | 每 VC 缓冲深度 (flit) |
| `internal_speedup` | 1.0 | 路由器内部加速比 |
| `routing_delay` | 1 | RC 延迟 (cycles) |
| `vc_alloc_delay` | 1 | VA 延迟 |
| `sw_alloc_delay` | 1 | SA 延迟 |
| `speculative` | 0 | 投机性 SA |
| `hold_switch_for_packet` | 0 | 多 flit 包持有 switch |

**支持的拓扑**: torus, mesh, cmesh, fly, fattree, flatfly, dragonfly, anynet, qtree, tree4

**支持的分配器**: islip, pim, max_size, loa, wavefront, separable_input_first, separable_output_first

**支持的路由算法**: 20+ (dor_mesh, xy_yx_mesh, adaptive, valiant, chaos, nca_fattree, ...)

### 3.5 对数据中心交换机建模的局限

| 方面 | BookSim (NoC) | 数据中心交换机需求 |
|------|---------------|------------------|
| 粒度 | Flit 级 (4-128 bits) | Packet 级 (64B-9KB MTU) |
| 缓冲 | 8-64 flit/VC (~0.1-1 KB) | 12 MB 片上 SRAM 或 100+ MB |
| 端口数 | 4-8 (2D mesh radix) | 32-128 |
| 流控 | Wormhole / VCT | Store-and-Forward / Cut-Through |
| 协议 | 无 | Ethernet/IP/RDMA |
| QoS | VC 优先级 | DSCP, PFC, WRED |
| 拥塞控制 | VC 反压 | ECN, PFC, DCQCN |

### 3.6 可借鉴之处

- **模块化架构**: Factory 模式的 Router/Allocator/Topology 设计
- **iSLIP 算法实现**: 指针去同步化的精确实现
- **Credit-based 流控**: flit 级 Credit 机制
- **可配置流水线**: routing_delay, vc_alloc_delay, sw_alloc_delay 独立可配
- **大量分配器**: 8 种分配器可对比研究

---

## 四、CNSim (清华大学, ATC 2024)

### 4.1 概述

CNSim 是清华大学发表在 USENIX ATC 2024 的 cycle-accurate 包并行仿真器，专为芯粒 (chiplet) 网络设计。

- **GitHub**: https://github.com/Yinxiao-Feng/chiplet-network-sim
- **代码量**: ~5K LOC C/C++
- **加速比**: 11-14x (相比传统仿真器)

### 4.2 核心创新: 包中心仿真

**传统方法 (BookSim 风格)**: 每个 cycle 遍历所有路由器和 VC，代价 O(routers x VCs)

**CNSim 方法**: 状态存储在 Packet 中，仿真遍历**活跃包**而非路由器

```cpp
class Packet {
    NodeID source_, destination_;
    int length_;                        // Flit 数
    vector<VCInfo> flit_trace_;         // 每 flit 位置追踪
    VCInfo next_vc_;                    // 下一分配的 VC
    bool switch_allocated_;             // SA 结果
    int process_timer_;                 // 注入预处理延迟
    int SA_timer_;                      // Switch allocation 延迟
    int link_timer_;                    // 链路穿越延迟
    int wait_timer_;                    // 排队延迟
    int trans_timer_;                   // 总传输时间
};
```

**优势**: 空闲路由器零仿真开销，低中注入率下大幅减少计算量。

### 4.3 Atomic-Based 多线程

使用原子变量替代锁进行线程协调:

```cpp
// 原子计数器分配工作
std::atomic<int> pkt_i;

// 每个工作线程:
while (true) {
    int my_batch = pkt_i.fetch_add(issue_width);
    if (my_batch >= packet_count) break;
    for (int i = my_batch; i < min(my_batch + issue_width, packet_count); i++) {
        system->update(*all_packets[i]);
    }
}

// 缓冲竞争通过 atomic 解决:
class Buffer {
    std::atomic_bool in_link_used_;
    std::atomic_bool sw_link_used_;
    std::atomic_int* vc_buffer_;       // 每 VC 空闲空间 (atomic)
    std::atomic<Packet*>* vc_head_packet;
};
```

### 4.4 可配置流水线

支持 1/2/3 阶段路由器:

| 模式 | 阶段 | 延迟 |
|------|------|------|
| OneStage | RC+VA+SA 合并 | 1 cycle |
| TwoStage | RC 分离, VA+SA 合并 | 2 cycles |
| ThreeStage | RC, VA, SA 各独立 | 3 cycles |

### 4.5 异构链路类型

| 通道类型 | 宽度 | 延迟 | 用途 |
|---------|------|------|------|
| `on_chip_channel` | 1 flit/cycle | 1 cycle | 芯粒内 NoC |
| `off_chip_parallel` | 1 flit/cycle | 2 cycles | D2D 并行接口 |
| `off_chip_serial` | 2 flits/cycle | 4 cycles | D2D 串行 (UCIe) |
| `long_distance` | 1 flit/cycle | 10 cycles | 跨封装/远程 |

### 4.6 性能数据

- 仿真规模: 5000 亿 cycles, 200 亿包
- 仿真时间: ~20 小时
- 传统仿真器估计: 200+ 小时
- 加速比: 11-14x

### 4.7 可借鉴之处

- **包中心仿真范式**: 将状态存储在 Packet 中而非路由器 -- 可直接用于数据中心交换机
- **Atomic 多线程**: 无锁并发调度，first-come-first-served 竞争解决
- **可配置流水线深度**: 不同交换机可配置不同流水线复杂度
- **异构链路**: 不同带宽/延迟的链路混合建模

---

## 五、框架对比总结

### 5.1 综合对比

| 特性 | SimAI | ASTRA-sim | BookSim | CNSim |
|------|-------|-----------|---------|-------|
| **目标场景** | AI 训练 | AI 训练 | NoC | Chiplet 网络 |
| **交换机精度** | 缓冲/队列级 | 无/NoC | Cycle-accurate NoC | Cycle-accurate |
| **数据中心适用性** | 高 (但交换机粗) | 中 (Garnet 不适用) | 低 (NoC 级) | 中 (需适配) |
| **仿真速度** | 中 (NS-3) / 快 (分析) | 快 (分析) / 慢 (Garnet) | 慢 | 快 (11-14x) |
| **多线程** | 23x (lock-free) | 无 | 无 | 11-14x (atomic) |
| **PFC/ECN** | 完整 | 无 | 无 | 无 |
| **代码量** | ~10K+ LOC | 大型 | ~15K LOC | ~5K LOC |
| **License** | Apache 2.0 | MIT | BSD | 未明确 |

### 5.2 对 Tier6 交换机建模的建议

基于以上分析，推荐的技术组合:

1. **仿真范式**: 采用 CNSim 的**包中心仿真**范式
   - 理由: Tier6 的 `latency.py` 已经是消息/包级别工作，自然匹配
   - 空闲交换机零开销，适合大规模拓扑

2. **调度器**: 采用 BookSim 的**模块化分配器框架**
   - 理由: iSLIP 的指针去同步化实现是学术标准
   - Factory 模式允许未来扩展其他分配器

3. **缓冲管理**: 采用 SimAI 的 **SwitchMmu 参数方案**
   - 理由: 经过真实集群验证 (98.1% 精度)
   - KMIN/KMAX/PMAX 带宽相关配置已有参考值

4. **拥塞控制**: 采用 SimAI 的 **ECN/PFC/DCQCN 实现**
   - 理由: 已验证的参数集和实现逻辑
   - 支持完整的 AI 网络无损语义

5. **多线程**: 如需加速，采用 CNSim 的 **atomic-based 线程模型**
   - 理由: 无锁设计适合包中心仿真
   - 在大规模场景下缓冲竞争稀少

### 5.3 参考论文

| 框架 | 论文 | 会议 |
|------|------|------|
| SimAI | "SimAI: Unifying Architecture Design and Performance Tunning for Large-Scale LLM Training with a Generalized Simulation Framework" | NSDI 2025 |
| ASTRA-sim 2.0 | "Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale" | ISPASS 2023 |
| BookSim 2.0 | "A Detailed and Flexible Cycle-Accurate Network-on-Chip Simulator" | ISPASS 2013 |
| CNSim | "Evaluating Chiplet-based Large-Scale Interconnection Networks via Cycle-Accurate Packet-Parallel Simulation" | ATC 2024 |
| iSLIP | "The iSLIP Scheduling Algorithm for Input-Queued Switches" | IEEE/ACM ToN 1999 |
| HPCC | "High Precision Congestion Control" | SIGCOMM 2019 |
| VOQ | "Dynamically-allocated Multi-queue Buffers for VLSI Communication Switches" | IEEE TC 1992 |

### 5.4 在线资源

- SimAI: https://github.com/aliyun/SimAI
- ASTRA-sim: https://github.com/astra-sim/astra-sim
- BookSim: https://github.com/booksim/booksim2
- CNSim: https://github.com/Yinxiao-Feng/chiplet-network-sim
- ns3-rdma (SimAI 交换机模型来源): https://github.com/bobzhuyb/ns3-rdma
- gem5 Garnet 2.0: https://www.gem5.org/documentation/general_docs/ruby/garnet-2/
