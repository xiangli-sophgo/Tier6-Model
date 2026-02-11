# L2 Arch -- 硬件架构层

## 功能概述

L2 定义 5 级硬件层级结构，从 Pod (集群) 到 Core (计算核心):

```
Pod (集群)
 +-- Rack (机柜)
     +-- Board (节点/服务器)
         +-- Chip (加速器)
             +-- Core (计算核心)
                 +-- Compute Units (Cube, Vector)
```

不在范围: 不做性能评估、并行切分。

## 模块清单

| 模块 | 职责 |
|------|------|
| `chip.py` | ChipSpecImpl (峰值算力, 内存, 微架构) |
| `core.py` | CoreSpecImpl (计算核心规格) |
| `compute.py` | ComputeSpec (Cube/Vector 单元) |
| `memory.py` | MemoryHierarchyImpl (GMEM, LMEM, SRAM) |
| `interconnect.py` | InterconnectSpecImpl (NoC, c2c) |
| `dma.py` | DMAEngineImpl |
| `board.py` | BoardSpecImpl (N chips per board) |
| `rack.py` | RackSpecImpl (M boards per rack) |
| `pod.py` | PodSpecImpl (K racks per pod) |
| `topology.py` | TopologySpec (4 级通信参数) + TopologySpecImpl |
| `protocols.py` | 硬件抽象协议 |

## ChipSpecImpl

### 核心能力

```python
class ChipSpecImpl:
    name: str                   # "SG2262"
    core_count: int             # 4
    frequency_ghz: float        # 1.0
    interconnect: InterconnectSpecImpl  # NoC 规格

    # 关键方法
    def get_peak_flops(dtype: str, unit: str) -> float:
        """峰值算力 (FLOPS)
        dtype: "BF16", "FP8", "INT8", ...
        unit: "cube", "vector"
        """

    def get_gmem_bandwidth() -> float:
        """全局内存带宽 (GB/s)"""

    def get_total_sram() -> int:
        """总 SRAM 容量 (bytes)"""

    @classmethod
    def from_config(name, config: dict) -> ChipSpecImpl:
        """从 YAML 配置构建"""
```

### 芯片配置格式 (YAML)

```yaml
name: SG2262
architecture: TPU_V7
process: 7nm
frequency_ghz: 1
align_bytes: 32
compute_dma_overlap_rate: 0.8    # 计算/DMA 重叠率
compute_efficiency: 0.9          # 计算效率

cores:
  count: 4
  lanes_per_core: 64

compute_units:
  cube:                          # GEMM 单元
    m: 16; k: 32; n: 8
    mac_per_lane:
      BF16: 500; FP8: 1000; INT8: 1000
  vector:                        # 向量单元
    eu_per_lane:
      BF16: 32; FP16: 32; FP32: 16

memory:
  gmem:                          # 全局内存 (HBM/LPDDR)
    type: LPDDR5
    capacity_gb: 64
    bandwidth_gbps: 273
    bandwidth_utilization: 0.85
  lmem:                          # 本地内存 (SRAM)
    capacity_mb: 64
    bandwidth_gbps: 2000
    sram_utilization: 0.45

dma_engines:
  gdma:
    bandwidth_gbps: 68
    efficiency: 0.9

interconnect:
  NoC:
    topology: Mesh
    bandwidth_gbps: 1000
```

### 峰值算力计算

```
peak_flops = cores * lanes_per_core * mac_per_lane[dtype] * frequency_ghz * 1e9 * 2

示例 (SG2262, BF16, Cube):
  = 4 * 64 * 500 * 1.0 * 1e9 * 2
  = 256 TFLOPS
```

## TopologySpec

### 4 级通信参数

```python
@dataclass
class TopologySpec:
    # Chip-to-Chip (同 board 内)
    c2c_bandwidth_gbps: float    # 448 GB/s (NVLink/自研互联)
    c2c_latency_us: float        # 0.2 us

    # Board-to-Board (同 rack 内)
    b2b_bandwidth_gbps: float    # 450 GB/s
    b2b_latency_us: float        # 0.35 us

    # Rack-to-Rack (同 pod 内)
    r2r_bandwidth_gbps: float    # 200 GB/s (InfiniBand)
    r2r_latency_us: float        # 2.0 us

    # Pod-to-Pod (跨 pod)
    p2p_bandwidth_gbps: float    # 100 GB/s (Ethernet)
    p2p_latency_us: float        # 5.0 us

    # 附加延迟参数
    memory_read_latency_us: float   # DDR 读延迟
    memory_write_latency_us: float  # DDR 写延迟
    noc_latency_us: float           # NoC 延迟
    die_to_die_latency_us: float    # Die-to-Die 延迟
    switch_latency_us: float        # 交换机延迟
    cable_latency_us: float         # 线缆延迟
```

### 拓扑配置格式 (YAML)

```yaml
name: P1-R1-B1-C8
pods:
- count: 1
  racks:
  - count: 1
    boards:
    - name: Board
      count: 1
      chips:
      - name: SG2262
        count: 8

chips:
  SG2262:
    # ... 完整芯片配置 (见上)

interconnect:
  links:
    c2c: { bandwidth_gbps: 448, latency_us: 0.2 }
    b2b: { bandwidth_gbps: 450, latency_us: 0.35 }
    r2r: { bandwidth_gbps: 200, latency_us: 2 }
    p2p: { bandwidth_gbps: 100, latency_us: 5 }
  comm_params:
    bandwidth_utilization: 0.95
    sync_latency_us: 0
    switch_latency_us: 1
    cable_latency_us: 0.025
    memory_read_latency_us: 0.15
    memory_write_latency_us: 0.01
    noc_latency_us: 0.05
    die_to_die_latency_us: 0.04
```

## TopologySpecImpl

### 路径解析

```python
@dataclass
class TopologySpecImpl:
    pods: dict[str, list[str]]        # pod_id -> rack_ids
    racks: dict[str, list[str]]       # rack_id -> board_ids
    boards: dict[str, list[str]]      # board_id -> chip_ids
    chips: list[str]                  # 所有 chip_id
    link_profiles: dict[str, LinkProfileImpl]  # 链路参数

    def resolve_path(src_chip, dst_chip) -> tuple[str, int]:
        """解析两个 chip 间的路径键与跳数
        - 同 board: ("c2c", 1)
        - 同 rack 不同 board: ("b2b", 2)
        - 同 pod 不同 rack: ("r2r", 3)
        - 跨 pod: ("p2p", 4)
        """
```

## Board / Rack / Pod

### BoardSpecImpl

```python
@dataclass
class BoardSpecImpl:
    board_id: str
    chips: list[ChipSpecImpl]     # 板上所有芯片
    chip_interconnect: LinkProfileImpl  # 芯片间互联 (c2c)

    def get_total_compute() -> float:
        """板卡总算力"""
    def get_allreduce_time(data_bytes, algorithm) -> float:
        """板内 AllReduce 时间"""
```

### RackSpecImpl

```python
@dataclass
class RackSpecImpl:
    rack_id: str
    boards: list[BoardSpecImpl]
    b2b_bandwidth_gbps: float     # 板间带宽
    b2b_latency_us: float         # 板间延迟
```

### PodSpecImpl

```python
@dataclass
class PodSpecImpl:
    pod_id: str
    racks: list[RackSpecImpl]
    r2r_bandwidth_gbps: float     # 机柜间带宽
    r2r_latency_us: float         # 机柜间延迟
```

## 硬件参数合并

L4 评估层需要一个扁平的 `dict[str, float]` 作为硬件参数。
`merge_specs()` 将三个 Spec 合并:

```python
def merge_specs(
    hardware: HardwareSpec,      # 芯片级 (compute_tflops, memory_bw)
    topology: TopologySpec,      # 通信参数 (c2c/b2b/r2r/p2p bw+lat)
    comm_protocol: CommProtocolSpec,  # 协议参数 (bw_utilization, sync_lat)
) -> dict[str, float]:
    return {**hw.to_dict(), **topo.to_dict(), **comm.to_dict()}
```

输出 dict 包含约 25 个 key，被 L4 的 ChipCostModel / CoreCostModel / CommEvaluator 直接使用。
