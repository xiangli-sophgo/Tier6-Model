# GEMM 精确评估器对齐计划

## 一、背景与目标

### 当前问题

现有 `latency.py` 使用简化的 Roofline 模型：

```python
# 当前实现 (latency.py:332-360)
def calc_attention_qkv_latency(...):
    flops = 2 * B * S * H * qkv_size / TP
    tflops = flops / 1e12
    compute_time = tflops / hardware.chip.compute_tflops_fp16 * 1000

    data_gb = (weight_bytes + io_bytes) / (1024 ** 3)
    memory_time = calc_hbm_read_latency(data_gb, hardware)

    return max(memory_time, compute_time)  # 简单 Roofline
```

**问题**：
1. 假设 100% 硬件利用率（实际只有 60-85%）
2. 没有考虑 Tile 分块策略
3. 没有考虑 SRAM 限制
4. 没有考虑内存对齐开销
5. 没有考虑计算-搬运重叠

**精度影响**：误差 20-40%，特别是小 batch decode 场景

### 对齐目标

将 DS_TPU_1209 的精确 GEMM 评估器移植到 Tier6+Model：
- **精度目标**：误差 < 10%
- **兼容性目标**：向后兼容，无微架构参数时降级到简化模型
- **性能目标**：评估时间 < 10ms（使用缓存）

---

## 二、文件结构设计

**核心原则**: 将原有的大文件 `latency.py` (28000+ tokens) 拆分为职责清晰的小模块。

```
backend/llm_simulator/
├── evaluators/                    # 新增: 精确评估器
│   ├── __init__.py               # 导出接口
│   ├── arch_config.py            # 硬件微架构配置
│   ├── gemm_eval.py              # GEMM 精确评估器
│   ├── presets.py                # 预定义硬件配置
│   └── utils.py                  # 工具函数
│
├── latency/                       # 重构: 原 latency.py 拆分
│   ├── __init__.py               # 统一导出 (保持 API 兼容)
│   ├── core.py                   # 核心: 评估器初始化、GEMM 通用接口
│   ├── attention.py              # Attention 相关延迟
│   ├── ffn.py                    # FFN 相关延迟
│   ├── mla.py                    # MLA (DeepSeek) 专用延迟
│   ├── moe.py                    # MoE 专用延迟
│   ├── memory.py                 # 内存访问延迟 (KV Cache, PCIe, HBM)
│   └── communication.py          # 通信延迟 (AllReduce, P2P, AllToAll)
│
├── types.py                       # 修改: 扩展 ChipHardwareConfig
├── simulator.py                   # 修改: 调用新的 latency 模块
└── ...
```

### 模块职责划分

| 模块 | 职责 | 大小估计 |
|------|------|---------|
| `latency/core.py` | 评估器初始化、`calc_gemm_latency()` | ~150 行 |
| `latency/attention.py` | QKV、Score、Softmax、Output | ~200 行 |
| `latency/ffn.py` | Gate、Up、Down | ~100 行 |
| `latency/mla.py` | MLA Q投影、KV压缩、Score、Output | ~200 行 |
| `latency/moe.py` | Gate、Expert FFN、负载均衡 | ~150 行 |
| `latency/memory.py` | KV Cache、PCIe、HBM、Embedding | ~200 行 |
| `latency/communication.py` | AllReduce、P2P、AllToAll、SP | ~200 行 |
| `latency/__init__.py` | 统一导出所有函数 | ~50 行 |

**总计**: ~1250 行，平均每文件 ~160 行 (原文件 ~1300 行)

---

## 三、详细实现步骤

### Step 1: 创建工具函数 (`evaluators/utils.py`)

```python
"""
GEMM 评估器工具函数
"""

def ceil_div(x: int, y: int) -> int:
    """向上取整除法"""
    return (x + y - 1) // y


def align_up(x: int, alignment: int) -> int:
    """向上对齐到 alignment 的倍数"""
    return ((x + alignment - 1) // alignment) * alignment


def flops_gemm(m: int, n: int, k: int) -> int:
    """计算 GEMM 的 FLOPs (2×M×N×K)"""
    return 2 * m * n * k
```

### Step 2: 创建硬件微架构配置 (`evaluators/arch_config.py`)

```python
"""
加速器微架构配置

定义精确建模所需的硬件参数，包括：
- 计算单元参数 (Cube/Tensor Core 维度)
- 内存层次参数 (SRAM 大小、带宽)
- 对齐约束
- 计算-搬运重叠率
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AcceleratorMicroArch:
    """加速器微架构配置"""

    # ========== 计算单元配置 ==========
    num_cores: int = 64
    """核心数量 (TPU cores / CUDA SMs)"""

    cube_m: int = 16
    """矩阵单元 M 维度"""

    cube_k: int = 32
    """矩阵单元 K 维度 (累加维度)"""

    cube_n: int = 8
    """矩阵单元 N 维度"""

    freq_ghz: float = 1.0
    """核心频率 (GHz)，可选，默认从 FLOPS 反推"""

    # ========== 内存配置 ==========
    sram_size_bytes: int = 2 * 1024 * 1024
    """每核 SRAM 大小 (字节)，默认 2MB"""

    sram_utilization: float = 0.45
    """SRAM 可用比例 (预留给系统/编译器)"""

    dram_bandwidth_bytes: float = 3200e9
    """DRAM 总带宽 (字节/秒)"""

    # ========== 对齐约束 ==========
    lane_num: int = 16
    """SIMD lane 数量 (行对齐基数)"""

    align_bytes: int = 32
    """内存对齐字节数 (列对齐基数)"""

    # ========== 执行模型 ==========
    compute_dma_overlap_rate: float = 0.8
    """计算与 DMA 搬运的重叠率 (0-1)"""

    # ========== 派生属性 ==========
    @property
    def macs_per_cycle(self) -> int:
        """每周期 MAC 操作数"""
        return self.cube_m * self.cube_k * self.cube_n

    @property
    def flops_per_second(self) -> float:
        """峰值 FLOPS"""
        return 2.0 * self.num_cores * self.macs_per_cycle * self.freq_ghz * 1e9

    @property
    def dma_bandwidth_per_core(self) -> float:
        """每核 DMA 带宽 (字节/秒)"""
        return self.dram_bandwidth_bytes / self.num_cores

    @property
    def effective_sram_bytes(self) -> int:
        """有效可用 SRAM 大小"""
        return int(self.sram_size_bytes * self.sram_utilization)

    def compute_freq_from_flops(self, total_flops: float) -> float:
        """从总 FLOPS 反推频率"""
        return total_flops / (2.0 * self.num_cores * self.macs_per_cycle * 1e9)
```

### Step 3: 创建预定义配置 (`evaluators/presets.py`)

```python
"""
预定义硬件微架构配置
"""

from .arch_config import AcceleratorMicroArch


# ==================== 算能 SG2260E ====================
SG2260E_ARCH = AcceleratorMicroArch(
    num_cores=64,
    cube_m=16,
    cube_k=32,
    cube_n=8,
    sram_size_bytes=2 * 1024 * 1024,  # 2MB
    sram_utilization=0.45,
    dram_bandwidth_bytes=273e9 * 0.893,  # 273 GB/s × 89.3%
    lane_num=16,
    align_bytes=32,
    compute_dma_overlap_rate=0.8,
)
# 频率从 FLOPS 反推
SG2260E_ARCH.freq_ghz = SG2260E_ARCH.compute_freq_from_flops(64e12)  # 64 TFLOPS


# ==================== NVIDIA H100 SXM ====================
H100_SXM_ARCH = AcceleratorMicroArch(
    num_cores=132,  # SM 数量
    cube_m=16,
    cube_k=16,
    cube_n=16,  # Tensor Core: 16×16×16
    sram_size_bytes=256 * 1024,  # 每 SM 共享内存 256KB
    sram_utilization=0.5,
    dram_bandwidth_bytes=3350e9 * 0.85,  # HBM3 3.35 TB/s × 85%
    lane_num=32,
    align_bytes=128,  # 128B 对齐
    compute_dma_overlap_rate=0.9,
)
H100_SXM_ARCH.freq_ghz = H100_SXM_ARCH.compute_freq_from_flops(989e12)


# ==================== NVIDIA A100 ====================
A100_ARCH = AcceleratorMicroArch(
    num_cores=108,
    cube_m=16,
    cube_k=16,
    cube_n=8,
    sram_size_bytes=192 * 1024,
    sram_utilization=0.5,
    dram_bandwidth_bytes=2039e9 * 0.85,
    lane_num=32,
    align_bytes=128,
    compute_dma_overlap_rate=0.85,
)
A100_ARCH.freq_ghz = A100_ARCH.compute_freq_from_flops(312e12)


# 配置查找表
ARCH_PRESETS = {
    'sg2260e': SG2260E_ARCH,
    'h100': H100_SXM_ARCH,
    'a100': A100_ARCH,
}


def get_arch_preset(name: str) -> AcceleratorMicroArch:
    """获取预定义配置"""
    name_lower = name.lower()
    if name_lower not in ARCH_PRESETS:
        raise ValueError(f"未知硬件类型: {name}，可选: {list(ARCH_PRESETS.keys())}")
    return ARCH_PRESETS[name_lower]
```

### Step 4: 实现 GEMM 精确评估器 (`evaluators/gemm_eval.py`)

```python
"""
GEMM 精确评估器

移植自 DS_TPU_1209/gemm.py，核心功能：
1. 多核分块策略搜索
2. Tile 大小搜索（受 SRAM 约束）
3. 循环顺序优化
4. 架构利用率计算
5. 计算-搬运重叠模型
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from functools import lru_cache

from .arch_config import AcceleratorMicroArch
from .utils import ceil_div, align_up

# 数据类型字节数
DTYPE_BYTES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'fp8': 1,
    'int8': 1,
}


@dataclass
class GEMMResult:
    """GEMM 评估结果"""
    latency_us: float
    """总延迟 (微秒)"""

    compute_time_us: float
    """计算时间 (微秒)"""

    memory_time_us: float
    """访存时间 (微秒)"""

    flops: int
    """浮点运算数"""

    dram_traffic_bytes: int
    """DRAM 流量 (字节)"""

    arch_utilization: float
    """架构利用率 (0-1)，考虑对齐损失"""

    effective_utilization: float
    """有效利用率 (0-1)，考虑访存瓶颈"""

    best_tile: Tuple[int, int, int]
    """最佳 Tile 大小 (m_t, n_t, k_t)"""

    best_loop_order: str
    """最佳循环顺序 ('mnk', 'nkm', 'mkn')"""

    best_partition: Tuple[int, int, int, int]
    """最佳多核分块 (P_G, P_M, P_N, P_K)"""


class GEMMEvaluator:
    """GEMM 精确评估器"""

    def __init__(self, arch: AcceleratorMicroArch):
        """
        初始化评估器

        Args:
            arch: 硬件微架构配置
        """
        self.arch = arch
        self._valid_partitions = self._compute_valid_partitions()

    def _compute_valid_partitions(self) -> List[Tuple[int, int, int, int]]:
        """
        枚举所有合法的多核分块方案

        约束: P_G × P_M × P_N × P_K = num_cores
        """
        partitions = []
        cores = self.arch.num_cores

        for p_g in range(1, cores + 1):
            if cores % p_g != 0:
                continue
            rem_m = cores // p_g

            for p_m in range(1, rem_m + 1):
                if rem_m % p_m != 0:
                    continue
                rem_n = rem_m // p_m

                for p_n in range(1, rem_n + 1):
                    if rem_n % p_n != 0:
                        continue
                    p_k = rem_n // p_n
                    partitions.append((p_g, p_m, p_n, p_k))

        return partitions

    def _find_legal_tiles(
        self,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> List[Tuple[int, int, int]]:
        """
        搜索所有能放进 SRAM 的 Tile 大小

        SRAM 布局:
        - A tile: [m_t, k_t] × input_dtype_bytes
        - B tile: [k_t, n_t] × input_dtype_bytes
        - C tile: [m_t, n_t] × output_dtype_bytes
        """
        if m_blk * n_blk * k_blk == 0:
            return [(0, 0, 0)]

        tiles = []
        cube_m = self.arch.cube_m
        cube_n = self.arch.cube_n
        cube_k = self.arch.cube_k
        sram_limit = self.arch.effective_sram_bytes
        lane_num = self.arch.lane_num
        align_bytes = self.arch.align_bytes

        # 对齐函数
        def align_row(r: int) -> int:
            return align_up(r, lane_num)

        def align_col(c: int, elem_bytes: int) -> int:
            return align_up(c * elem_bytes, align_bytes)

        # 从大到小搜索 Tile (越大越好，数据复用越多)
        for m_t in range(align_up(m_blk, cube_m), 0, -cube_m):
            align_row_m = align_row(m_t)

            for n_t in range(align_up(n_blk, cube_n), 0, -cube_n):
                align_col_n = align_col(n_t, output_dtype_bytes)
                align_row_n = align_row(n_t)

                # C tile 必须放得下
                c_tile_bytes = align_row_n * align_col_n
                avail = sram_limit - c_tile_bytes

                if avail <= 0:
                    continue

                # 计算最大 k_t
                # A tile: [m_t, k_t], B tile: [k_t, n_t]
                bytes_per_k = (align_row_m + align_row_n) * input_dtype_bytes
                max_k = int(avail / bytes_per_k) if bytes_per_k > 0 else k_blk

                if max_k <= 0:
                    continue

                # 对齐到 cube_k
                k_t = min(k_blk, max_k)
                align_k = align_up(k_t, cube_k)

                if align_k < cube_k:
                    k_t = max_k
                else:
                    k_t = (align_k - cube_k) if align_k > max_k else align_k
                    if k_t <= 0:
                        continue

                # Pareto 最优检查 (去除被支配的 tile)
                is_dominated = any(
                    m0 >= m_t and n0 >= n_t and k0 >= k_t
                    for m0, n0, k0 in tiles
                )
                if not is_dominated:
                    tiles.append((m_t, n_t, k_t))

        # 如果没找到合法 tile，使用最小的 cube 大小
        if not tiles:
            tiles.append((cube_m, cube_n, cube_k))

        return tiles

    def _calc_dram_traffic(
        self,
        loop_order: str,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        m_t: int,
        n_t: int,
        k_t: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> int:
        """
        计算 DRAM 流量 (字节)

        不同循环顺序的流量差异:
        - mnk: K 在最内层，A/B 各重复加载 tile_num_n/tile_num_m 次
        - nkm: M 在最内层，B 只加载一次，但 C 需要多次累加
        - mkn: N 在最内层，A 只加载一次，但 C 需要多次累加
        """
        if m_blk * n_blk * k_blk == 0:
            return 0

        if m_t <= 0 or n_t <= 0 or k_t <= 0:
            return 0

        tile_num_m = ceil_div(m_blk, m_t)
        tile_num_n = ceil_div(n_blk, n_t)
        tile_num_k = ceil_div(k_blk, k_t)

        a_size = m_blk * k_blk * input_dtype_bytes
        b_size = n_blk * k_blk * input_dtype_bytes
        c_size = m_blk * n_blk * output_dtype_bytes

        if loop_order == 'mnk':
            # A 重复 tile_num_n 次, B 重复 tile_num_m 次
            return a_size * tile_num_n + b_size * tile_num_m + c_size

        elif loop_order == 'nkm':
            # B 只加载一次, A 重复 tile_num_n 次
            # C 需要 tile_num_k - 1 次累加 (读+写 FP32)
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return b_size + a_size * tile_num_n + partial_sum_traffic + c_size

        else:  # mkn
            # A 只加载一次, B 重复 tile_num_m 次
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return a_size + b_size * tile_num_m + partial_sum_traffic + c_size

    def _calc_arch_utilization(
        self,
        g_blk: int,
        m_blk: int,
        n_blk: int,
        k_blk: int,
    ) -> Tuple[float, float]:
        """
        计算架构利用率和计算时间

        架构利用率 = 实际 MACs / 对齐后理论 MACs

        Returns:
            (arch_utilization, compute_time_us)
        """
        if m_blk * n_blk * k_blk == 0:
            return 0.0, 0.0

        # 实际 MAC 数
        real_macs = m_blk * n_blk * k_blk

        # 对齐后的理论 MAC 数
        theo_macs = (
            align_up(m_blk, self.arch.cube_m) *
            align_up(k_blk, self.arch.cube_k) *
            align_up(n_blk, self.arch.cube_n)
        )

        arch_util = real_macs / theo_macs if theo_macs > 0 else 0.0

        # 计算时间 (微秒)
        # t = theo_macs × g_blk / macs_per_cycle / freq_ghz / 1e3
        t_us = theo_macs * g_blk / self.arch.macs_per_cycle / self.arch.freq_ghz / 1e3

        return arch_util, t_us

    def _evaluate_partition(
        self,
        p_g: int,
        p_m: int,
        p_n: int,
        p_k: int,
        G: int,
        M: int,
        N: int,
        K: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> Tuple[float, GEMMResult]:
        """
        评估单个分块方案

        Returns:
            (total_time_us, GEMMResult)
        """
        # 每核分配的维度
        g_nom = ceil_div(G, p_g)
        m_nom = ceil_div(M, p_m)
        n_nom = ceil_div(N, p_n)
        k_nom = ceil_div(K, p_k)

        # 搜索最佳 tile 和循环顺序
        tiles = self._find_legal_tiles(m_nom, n_nom, k_nom, input_dtype_bytes, output_dtype_bytes)

        min_traffic = float('inf')
        best_tile = tiles[0] if tiles else (0, 0, 0)
        best_order = 'mnk'

        for m_t, n_t, k_t in tiles:
            if m_t <= 0:
                continue
            for order in ('mnk', 'nkm', 'mkn'):
                traffic = self._calc_dram_traffic(
                    order, m_nom, n_nom, k_nom, m_t, n_t, k_t,
                    input_dtype_bytes, output_dtype_bytes
                )
                if traffic < min_traffic:
                    min_traffic = traffic
                    best_tile = (m_t, n_t, k_t)
                    best_order = order

        m_t, n_t, k_t = best_tile

        # 遍历所有核心，计算最长执行时间
        total_flops = 0
        total_traffic = 0
        max_time = 0.0
        best_t_comp = 0.0
        best_t_dma = 0.0
        active_cores = 0

        for i_g in range(p_g):
            g_start = i_g * g_nom
            g_blk = max(min(G - g_start, g_nom), 0)

            for i_m in range(p_m):
                m_start = i_m * m_nom
                m_blk = max(min(M - m_start, m_nom), 0)

                for i_n in range(p_n):
                    n_start = i_n * n_nom
                    n_blk = max(min(N - n_start, n_nom), 0)

                    for i_k in range(p_k):
                        k_start = i_k * k_nom
                        k_blk = max(min(K - k_start, k_nom), 0)

                        # 计算当前核心的 FLOPs
                        core_flops = 2 * g_blk * m_blk * n_blk * k_blk

                        # 计算架构利用率和计算时间
                        arch_util, t_comp = self._calc_arch_utilization(g_blk, m_blk, n_blk, k_blk)

                        # 计算 DRAM 流量和访存时间
                        traffic = g_blk * self._calc_dram_traffic(
                            best_order, m_blk, n_blk, k_blk, m_t, n_t, k_t,
                            input_dtype_bytes, output_dtype_bytes
                        )
                        t_dma = 1e6 * traffic / self.arch.dma_bandwidth_per_core

                        # 计算-搬运重叠
                        overlap = self.arch.compute_dma_overlap_rate
                        t_total = (min(t_comp, t_dma) * (1 - overlap) +
                                   max(t_comp, t_dma))

                        # 更新统计
                        if t_total > max_time:
                            max_time = t_total
                            best_t_comp = t_comp
                            best_t_dma = t_dma

                        if core_flops > 0:
                            active_cores += 1

                        total_flops += core_flops
                        total_traffic += traffic

        # 计算总体利用率
        if max_time > 0:
            # 理论峰值: num_cores × macs_per_cycle × freq × 2 (FLOPs)
            peak_flops_per_us = (
                self.arch.num_cores * self.arch.macs_per_cycle * self.arch.freq_ghz * 2 / 1e3
            )
            overall_util = total_flops / (max_time * peak_flops_per_us)
            arch_util_avg = best_t_comp / max_time * (total_flops / (2 * G * M * N * K)) if max_time > 0 else 0
        else:
            overall_util = 0.0
            arch_util_avg = 0.0

        result = GEMMResult(
            latency_us=max_time,
            compute_time_us=best_t_comp,
            memory_time_us=best_t_dma,
            flops=total_flops,
            dram_traffic_bytes=int(total_traffic),
            arch_utilization=arch_util_avg,
            effective_utilization=overall_util,
            best_tile=best_tile,
            best_loop_order=best_order,
            best_partition=(p_g, p_m, p_n, p_k),
        )

        return max_time, result

    @lru_cache(maxsize=4096)
    def evaluate(
        self,
        G: int,
        M: int,
        K: int,
        N: int,
        input_dtype: str = "fp8",
        output_dtype: str = "bf16",
    ) -> GEMMResult:
        """
        评估 GEMM: C[G, M, N] = A[G, M, K] × B[G, K, N]

        Args:
            G: Batch/Group 维度 (可以是 1)
            M: 输出行数
            K: 累加维度
            N: 输出列数
            input_dtype: 输入数据类型 ('fp8', 'bf16', 'fp16')
            output_dtype: 输出数据类型 ('bf16', 'fp32')

        Returns:
            GEMMResult: 包含延迟、利用率、最佳配置等

        Example:
            >>> evaluator = GEMMEvaluator(SG2260E_ARCH)
            >>> result = evaluator.evaluate(1, 48, 7168, 2048)  # FFN
            >>> print(f"延迟: {result.latency_us:.2f} μs")
        """
        input_bytes = DTYPE_BYTES.get(input_dtype, 1)
        output_bytes = DTYPE_BYTES.get(output_dtype, 2)

        best_time = float('inf')
        best_result = None

        # 遍历所有分块方案，选择最优
        for partition in self._valid_partitions:
            p_g, p_m, p_n, p_k = partition

            time_us, result = self._evaluate_partition(
                p_g, p_m, p_n, p_k,
                G, M, N, K,
                input_bytes, output_bytes,
            )

            if time_us < best_time:
                best_time = time_us
                best_result = result

        return best_result


# ==================== 便捷接口 ====================

_evaluator_cache: dict = {}


def get_gemm_evaluator(arch: AcceleratorMicroArch) -> GEMMEvaluator:
    """获取或创建 GEMM 评估器 (缓存单例)"""
    key = id(arch)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = GEMMEvaluator(arch)
    return _evaluator_cache[key]


def eval_gemm(
    arch: AcceleratorMicroArch,
    G: int,
    M: int,
    K: int,
    N: int,
    input_dtype: str = "fp8",
    output_dtype: str = "bf16",
) -> GEMMResult:
    """
    快速评估 GEMM

    Args:
        arch: 硬件微架构配置
        G, M, K, N: GEMM 维度
        input_dtype: 输入类型
        output_dtype: 输出类型

    Returns:
        GEMMResult
    """
    evaluator = get_gemm_evaluator(arch)
    return evaluator.evaluate(G, M, K, N, input_dtype, output_dtype)
```

### Step 5: 创建模块入口 (`evaluators/__init__.py`)

```python
"""
精确评估器模块

提供 GEMM、FlashAttention 等算子的精确性能评估。
"""

from .arch_config import AcceleratorMicroArch
from .presets import (
    SG2260E_ARCH,
    H100_SXM_ARCH,
    A100_ARCH,
    ARCH_PRESETS,
    get_arch_preset,
)
from .gemm_eval import (
    GEMMResult,
    GEMMEvaluator,
    get_gemm_evaluator,
    eval_gemm,
)
from .utils import ceil_div, align_up

__all__ = [
    # 配置
    'AcceleratorMicroArch',
    'SG2260E_ARCH',
    'H100_SXM_ARCH',
    'A100_ARCH',
    'ARCH_PRESETS',
    'get_arch_preset',
    # GEMM
    'GEMMResult',
    'GEMMEvaluator',
    'get_gemm_evaluator',
    'eval_gemm',
    # 工具
    'ceil_div',
    'align_up',
]
```

### Step 6: 扩展 `types.py` 中的硬件配置

```python
# 在 ChipHardwareConfig 中添加可选的微架构参数

@dataclass
class ChipHardwareConfig:
    """芯片硬件配置"""
    # ... 现有字段保持不变 ...

    # ========== 新增: 微架构参数 (可选) ==========
    # 如果提供这些参数，将使用精确 GEMM 评估器
    # 如果不提供，降级到简化 Roofline 模型

    cube_m: Optional[int] = None
    """矩阵单元 M 维度"""

    cube_k: Optional[int] = None
    """矩阵单元 K 维度"""

    cube_n: Optional[int] = None
    """矩阵单元 N 维度"""

    sram_size_kb: Optional[float] = None
    """每核 SRAM 大小 (KB)"""

    sram_utilization: Optional[float] = None
    """SRAM 可用比例"""

    lane_num: Optional[int] = None
    """SIMD lane 数"""

    align_bytes: Optional[int] = None
    """内存对齐字节数"""

    compute_dma_overlap_rate: Optional[float] = None
    """计算-搬运重叠率"""

    def has_micro_arch(self) -> bool:
        """检查是否有微架构参数"""
        return self.cube_m is not None and self.cube_k is not None
```

### Step 7: 拆分重写 `latency/` 模块

**设计原则**:

1. 将原 `latency.py` 拆分为多个小模块
2. 所有 GEMM 相关延迟计算全部使用精确评估器
3. 通过 `__init__.py` 统一导出

---

#### 7.1 `latency/core.py` - 核心模块

```python
"""
延迟计算核心模块

职责:
- 评估器初始化与管理
- GEMM 通用接口
"""

from typing import Optional
from ..types import HardwareConfig, get_bytes_per_element
from ..evaluators import (
    AcceleratorMicroArch,
    GEMMEvaluator,
    GEMMResult,
    get_arch_preset,
)

# 全局状态
_current_arch: AcceleratorMicroArch = None
_gemm_evaluator: GEMMEvaluator = None


def init_evaluators(hardware: HardwareConfig) -> None:
    """初始化评估器 (模拟开始时调用)"""
    global _current_arch, _gemm_evaluator
    _current_arch = _create_arch_from_hardware(hardware)
    _gemm_evaluator = GEMMEvaluator(_current_arch)


def _create_arch_from_hardware(hardware: HardwareConfig) -> AcceleratorMicroArch:
    """从硬件配置创建微架构"""
    chip = hardware.chip
    chip_type = getattr(chip, 'chip_type', '').lower()

    # 查找预设
    if 'sg2260' in chip_type:
        arch = get_arch_preset('sg2260e')
    elif 'h100' in chip_type:
        arch = get_arch_preset('h100')
    elif 'a100' in chip_type:
        arch = get_arch_preset('a100')
    else:
        arch = get_arch_preset('sg2260e')

    # 更新带宽
    arch.dram_bandwidth_bytes = chip.memory_bandwidth_gbps * 1e9 * chip.memory_bandwidth_utilization
    return arch


def get_arch() -> AcceleratorMicroArch:
    """获取当前微架构"""
    if _current_arch is None:
        raise RuntimeError("评估器未初始化")
    return _current_arch


def calc_gemm_latency(M: int, K: int, N: int, G: int = 1) -> float:
    """计算 GEMM 延迟 (ms)"""
    result = _gemm_evaluator.evaluate(G, M, K, N)
    return result.latency_us / 1000


def calc_gemm_with_details(M: int, K: int, N: int, G: int = 1) -> GEMMResult:
    """计算 GEMM 延迟并返回详细信息"""
    return _gemm_evaluator.evaluate(G, M, K, N)
```

---

#### 7.2 `latency/attention.py` - Attention 模块

```python
"""
Attention 相关延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig
from .core import calc_gemm_latency, get_arch


def calc_attention_qkv_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """QKV 投影延迟"""
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    head_dim = H // model.num_attention_heads
    tp = parallelism.tp

    q_latency = calc_gemm_latency(M=B*S, K=H, N=H//tp)
    kv_dim = model.num_kv_heads * head_dim // tp
    k_latency = calc_gemm_latency(M=B*S, K=H, N=kv_dim)
    v_latency = calc_gemm_latency(M=B*S, K=H, N=kv_dim)

    return q_latency + k_latency + v_latency


def calc_attention_score_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """Attention Score (Q @ K^T) 延迟"""
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    heads = model.num_attention_heads
    head_dim = H // heads
    tp = parallelism.tp

    G = B * heads // tp
    return calc_gemm_latency(M=S, K=head_dim, N=context_length, G=G)


def calc_attention_output_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """Attention Output 延迟 (Softmax@V + OutProj)"""
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    heads = model.num_attention_heads
    head_dim = H // heads
    tp = parallelism.tp

    G = B * heads // tp
    sv_latency = calc_gemm_latency(M=S, K=context_length, N=head_dim, G=G)
    out_latency = calc_gemm_latency(M=B*S, K=H//tp, N=H)

    return sv_latency + out_latency


def calc_attention_softmax_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
    context_length: int,
) -> float:
    """Softmax 延迟 (内存受限)"""
    B, S = inference.batch_size, num_tokens
    heads = model.num_attention_heads // parallelism.tp

    data_bytes = B * heads * S * context_length * 2  # BF16
    data_gb = data_bytes / 1e9
    effective_bw = get_arch().dram_bandwidth_bytes / 1e9

    return 2 * data_gb / effective_bw * 1000
```

---

#### 7.3 `latency/ffn.py` - FFN 模块

```python
"""
FFN (Feed-Forward Network) 相关延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig
from .core import calc_gemm_latency


def calc_ffn_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """FFN Gate 投影延迟"""
    B, S, H, I = inference.batch_size, num_tokens, model.hidden_size, model.intermediate_size
    return calc_gemm_latency(M=B*S, K=H, N=I//parallelism.tp)


def calc_ffn_up_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """FFN Up 投影延迟"""
    B, S, H, I = inference.batch_size, num_tokens, model.hidden_size, model.intermediate_size
    return calc_gemm_latency(M=B*S, K=H, N=I//parallelism.tp)


def calc_ffn_down_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """FFN Down 投影延迟"""
    B, S, H, I = inference.batch_size, num_tokens, model.hidden_size, model.intermediate_size
    return calc_gemm_latency(M=B*S, K=I//parallelism.tp, N=H)


def calc_layernorm_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """LayerNorm/RMSNorm 延迟 (内存受限)"""
    from ..types import get_bytes_per_element
    from .core import get_arch

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    bytes_per_elem = get_bytes_per_element(model.dtype)

    data_bytes = 2 * B * S * H * bytes_per_elem
    data_gb = data_bytes / 1e9
    effective_bw = get_arch().dram_bandwidth_bytes / 1e9

    return data_gb / effective_bw * 1000
```

---

#### 7.4 `latency/moe.py` - MoE 模块

```python
"""
MoE (Mixture of Experts) 相关延迟计算
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig
from .core import calc_gemm_latency, get_arch


# 专家负载不均衡查表 (来自 DS_TPU_1209)
EXPERT_LOAD_IMBALANCE = {
    # (batch, num_chips) -> max_experts
    (4, 1): 30.5, (4, 2): 17.3, (4, 4): 10.4, (4, 8): 6.6,
    (8, 1): 57.4, (8, 2): 31.4, (8, 4): 17.8, (8, 8): 10.7,
    (16, 1): 106.3, (16, 2): 57.4, (16, 4): 31.4, (16, 8): 17.8,
    (32, 1): 188.3, (32, 2): 106.3, (32, 4): 57.4, (32, 8): 31.4,
    (64, 1): 320.0, (64, 2): 188.3, (64, 4): 106.3, (64, 8): 57.4,
}


def _get_expert_load_imbalance(total_tokens: int, num_experts: int, ep: int) -> float:
    """获取专家负载不均衡因子"""
    tokens_per_expert_avg = total_tokens / num_experts

    if tokens_per_expert_avg < 1:
        return 2.0
    elif tokens_per_expert_avg < 4:
        return 1.5
    elif tokens_per_expert_avg < 16:
        return 1.3
    else:
        return 1.1


def calc_moe_gate_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """MoE Gate 路由延迟"""
    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    num_experts = model.moe_config.num_experts if model.moe_config else 8
    return calc_gemm_latency(M=B*S, K=H, N=num_experts)


def calc_moe_expert_ffn_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """MoE Expert FFN 延迟 (考虑负载不均衡)"""
    if not model.moe_config:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    moe = model.moe_config
    ep = parallelism.ep

    expert_I = moe.expert_intermediate_size or model.intermediate_size
    top_k = moe.num_experts_per_tok

    # 负载不均衡因子
    load_imbalance = _get_expert_load_imbalance(B * S, moe.num_experts, ep)

    tokens_per_expert = max(1, int(B * S * top_k / moe.num_experts * load_imbalance / ep))
    experts_per_chip = moe.num_experts // ep

    # 单专家 FFN
    gate = calc_gemm_latency(M=tokens_per_expert, K=H, N=expert_I)
    up = calc_gemm_latency(M=tokens_per_expert, K=H, N=expert_I)
    down = calc_gemm_latency(M=tokens_per_expert, K=expert_I, N=H)

    single_expert = gate + up + down
    return single_expert * experts_per_chip / get_arch().num_cores
```

---

#### 7.5 `latency/mla.py` - MLA 模块 (DeepSeek V3)

```python
"""
MLA (Multi-head Latent Attention) 相关延迟计算
DeepSeek V3/R1 专用
"""

from ..types import LLMModelConfig, InferenceConfig, ParallelismStrategy, HardwareConfig
from .core import calc_gemm_latency
from .attention import calc_attention_qkv_latency


def calc_mla_q_projection_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """MLA Q 投影延迟 (LoRA-style)"""
    if not model.mla_config:
        return calc_attention_qkv_latency(model, inference, parallelism, hardware, num_tokens)

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads
    head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim

    # Down: [B×S, H] → [B×S, q_lora_rank]
    down = calc_gemm_latency(M=B*S, K=H, N=mla.q_lora_rank)
    # Up: [B×S, q_lora_rank] → [B×S, heads×head_dim/TP]
    up = calc_gemm_latency(M=B*S, K=mla.q_lora_rank, N=heads*head_dim//tp)

    return down + up


def calc_mla_kv_compression_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    num_tokens: int,
) -> float:
    """MLA KV 压缩延迟"""
    if not model.mla_config:
        return 0.0

    B, S, H = inference.batch_size, num_tokens, model.hidden_size
    mla = model.mla_config

    return calc_gemm_latency(M=B*S, K=H, N=mla.kv_lora_rank)


def calc_mla_decode_attention_latency(
    model: LLMModelConfig,
    inference: InferenceConfig,
    parallelism: ParallelismStrategy,
    hardware: HardwareConfig,
    context_length: int,
) -> float:
    """MLA Decode 阶段 Attention 延迟 (MQA 优化)"""
    if not model.mla_config:
        return 0.0

    B, H = inference.batch_size, model.hidden_size
    mla = model.mla_config
    tp = parallelism.tp
    heads = model.num_attention_heads

    # W_KC @ c_t^KV: [heads, B, qk_nope, kv_lora_rank]
    w_kc = calc_gemm_latency(
        M=B, K=mla.qk_nope_head_dim, N=mla.kv_lora_rank, G=heads//tp
    )
    # W_VC @ c_t^KV: [heads, B, v_head_dim, kv_lora_rank]
    w_vc = calc_gemm_latency(
        M=B, K=mla.v_head_dim, N=mla.kv_lora_rank, G=heads//tp
    )

    # MQA: [B×heads/TP, 1, context]
    mqa = calc_gemm_latency(M=1, K=context_length, N=mla.v_head_dim, G=B*heads//tp)

    return w_kc + w_vc + mqa
```

---

#### 7.6 `latency/communication.py` - 通信模块

```python
"""
集合通信延迟计算
"""


def calc_tp_allreduce_latency(
    data_size_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    tp: int,
) -> float:
    """TP AllReduce 延迟 (Ring)"""
    if tp <= 1:
        return 0.0

    transfer_factor = 2 * (tp - 1) / tp
    transfer_ms = (data_size_gb * transfer_factor / bandwidth_gbps) * 1000
    startup_ms = latency_us * (tp - 1) / 1000

    return transfer_ms + startup_ms


def calc_pp_p2p_latency(
    data_size_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
) -> float:
    """PP P2P 通信延迟"""
    transfer_ms = (data_size_gb / bandwidth_gbps) * 1000
    startup_ms = latency_us / 1000
    return transfer_ms + startup_ms


def calc_ep_alltoall_latency(
    data_size_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    ep: int,
) -> float:
    """EP All-to-All 延迟"""
    if ep <= 1:
        return 0.0

    # All-to-All: 每个节点发送 (ep-1)/ep 的数据
    transfer_factor = (ep - 1) / ep
    transfer_ms = (data_size_gb * transfer_factor / bandwidth_gbps) * 1000
    startup_ms = latency_us * (ep - 1) / 1000

    return transfer_ms + startup_ms


def calc_sp_allgather_latency(
    data_size_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    sp: int,
) -> float:
    """SP AllGather 延迟"""
    if sp <= 1:
        return 0.0

    transfer_factor = (sp - 1) / sp
    transfer_ms = (data_size_gb * transfer_factor / bandwidth_gbps) * 1000
    startup_ms = latency_us * (sp - 1) / 1000

    return transfer_ms + startup_ms


def calc_sp_reduce_scatter_latency(
    data_size_gb: float,
    bandwidth_gbps: float,
    latency_us: float,
    sp: int,
) -> float:
    """SP ReduceScatter 延迟"""
    return calc_sp_allgather_latency(data_size_gb, bandwidth_gbps, latency_us, sp)
```

---

#### 7.7 `latency/__init__.py` - 统一导出

```python
"""
延迟计算模块

统一导出所有延迟计算函数，保持 API 兼容
"""

# 核心
from .core import (
    init_evaluators,
    get_arch,
    calc_gemm_latency,
    calc_gemm_with_details,
)

# Attention
from .attention import (
    calc_attention_qkv_latency,
    calc_attention_score_latency,
    calc_attention_output_latency,
    calc_attention_softmax_latency,
)

# FFN
from .ffn import (
    calc_ffn_gate_latency,
    calc_ffn_up_latency,
    calc_ffn_down_latency,
    calc_layernorm_latency,
)

# MoE
from .moe import (
    calc_moe_gate_latency,
    calc_moe_expert_ffn_latency,
)

# MLA
from .mla import (
    calc_mla_q_projection_latency,
    calc_mla_kv_compression_latency,
    calc_mla_decode_attention_latency,
)

# Communication
from .communication import (
    calc_tp_allreduce_latency,
    calc_pp_p2p_latency,
    calc_ep_alltoall_latency,
    calc_sp_allgather_latency,
    calc_sp_reduce_scatter_latency,
)

__all__ = [
    # Core
    'init_evaluators', 'get_arch', 'calc_gemm_latency', 'calc_gemm_with_details',
    # Attention
    'calc_attention_qkv_latency', 'calc_attention_score_latency',
    'calc_attention_output_latency', 'calc_attention_softmax_latency',
    # FFN
    'calc_ffn_gate_latency', 'calc_ffn_up_latency', 'calc_ffn_down_latency',
    'calc_layernorm_latency',
    # MoE
    'calc_moe_gate_latency', 'calc_moe_expert_ffn_latency',
    # MLA
    'calc_mla_q_projection_latency', 'calc_mla_kv_compression_latency',
    'calc_mla_decode_attention_latency',
    # Communication
    'calc_tp_allreduce_latency', 'calc_pp_p2p_latency', 'calc_ep_alltoall_latency',
    'calc_sp_allgather_latency', 'calc_sp_reduce_scatter_latency',
]
```

---

## 四、测试计划

### 单元测试 (`tests/test_gemm_eval.py`)

```python
"""
GEMM 评估器单元测试
"""
import pytest
from backend.llm_simulator.evaluators import (
    SG2260E_ARCH,
    GEMMEvaluator,
    eval_gemm,
)


class TestGEMMEvaluator:
    """GEMM 评估器测试"""

    @pytest.fixture
    def evaluator(self):
        return GEMMEvaluator(SG2260E_ARCH)

    def test_basic_gemm(self, evaluator):
        """基本 GEMM 测试"""
        result = evaluator.evaluate(G=1, M=1024, K=1024, N=1024)

        assert result.latency_us > 0
        assert result.flops == 2 * 1024 * 1024 * 1024
        assert 0 < result.arch_utilization <= 1.0
        assert result.best_tile[0] > 0

    def test_small_batch_decode(self, evaluator):
        """小 batch decode 场景 (精度关键)"""
        # DeepSeek V3 FFN Down: [48, 2048] × [2048, 7168]
        result = evaluator.evaluate(G=1, M=48, K=2048, N=7168)

        # 预期利用率较低 (M=48 对齐损失)
        assert result.arch_utilization < 0.8
        assert result.latency_us > 50  # 应该 > 50μs

    def test_large_batch_prefill(self, evaluator):
        """大 batch prefill 场景"""
        # [4096, 7168] × [7168, 7168]
        result = evaluator.evaluate(G=1, M=4096, K=7168, N=7168)

        # 预期利用率较高
        assert result.arch_utilization > 0.9

    def test_caching(self, evaluator):
        """测试缓存生效"""
        # 第一次调用
        result1 = evaluator.evaluate(G=1, M=512, K=512, N=512)

        # 第二次调用 (应该命中缓存)
        result2 = evaluator.evaluate(G=1, M=512, K=512, N=512)

        assert result1.latency_us == result2.latency_us


class TestGEMMAccuracy:
    """GEMM 精度验证 (对比 DS_TPU_1209 参考值)"""

    # 参考值来自 DS_TPU_1209 实测
    REFERENCE_DATA = [
        # (M, K, N, expected_latency_us, tolerance)
        (48, 7168, 2048, 82, 0.15),     # FFN Down
        (48, 7168, 576, 25, 0.15),      # MLA q_a_proj
        (4096, 7168, 7168, 1200, 0.10), # Prefill FFN
    ]

    def test_against_reference(self):
        """对比参考实现"""
        for M, K, N, expected, tolerance in self.REFERENCE_DATA:
            result = eval_gemm(SG2260E_ARCH, G=1, M=M, K=K, N=N)

            error = abs(result.latency_us - expected) / expected
            assert error < tolerance, \
                f"GEMM[{M},{K},{N}]: got {result.latency_us:.1f}μs, " \
                f"expected {expected}μs, error {error:.1%}"
```

### 集成测试

```python
"""
集成测试: 验证完整模拟流程
"""
def test_simulation_with_precise_gemm():
    """测试带精确 GEMM 的完整模拟"""
    # 配置带微架构参数的硬件
    hardware_dict = {
        "chip": {
            "chip_type": "SG2260E",
            "compute_tflops_fp16": 64,
            "memory_gb": 64,
            "memory_bandwidth_gbps": 273,
            "num_cores": 64,
            # 微架构参数
            "cube_m": 16,
            "cube_k": 32,
            "cube_n": 8,
            "sram_size_kb": 2048,
            "sram_utilization": 0.45,
        },
        # ...
    }

    # 运行模拟
    result = run_simulation(
        topology_dict=...,
        model_dict=deepseek_v3_config,
        hardware_dict=hardware_dict,
        ...
    )

    # 验证结果
    assert result['stats']['ttft'] > 0
    assert result['stats']['dynamicMfu'] > 0
```

---

## 五、前端修改 (可选)

### 添加微架构配置 UI

```typescript
// frontend/src/types/hardware.ts
interface ChipMicroArchConfig {
  cubeM?: number;
  cubeK?: number;
  cubeN?: number;
  sramSizeKB?: number;
  sramUtilization?: number;
  laneNum?: number;
  alignBytes?: number;
  computeDmaOverlapRate?: number;
}

// frontend/src/components/ConfigPanel/HardwareConfig.tsx
// 添加"高级微架构设置"折叠面板
```

---

## 六、实施顺序

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| **Phase 1** | 创建 `evaluators/` 目录结构 | 0.5h |
| **Phase 2** | 实现 `utils.py`, `arch_config.py` | 1h |
| **Phase 3** | 实现 `presets.py` | 0.5h |
| **Phase 4** | 实现 `gemm_eval.py` 核心逻辑 | 3h |
| **Phase 5** | 扩展 `types.py` | 0.5h |
| **Phase 6** | 修改 `latency.py` 集成 | 2h |
| **Phase 7** | 单元测试 | 1h |
| **Phase 8** | 集成测试 & 调试 | 2h |
| **Phase 9** | 前端修改 (可选) | 1h |

**总计**: 约 11-12 小时

---

## 七、验证标准

### 精度验证

| 场景 | 目标精度 |
|------|---------|
| Prefill (大 batch) | < 10% 误差 |
| Decode (小 batch) | < 15% 误差 |
| MoE Expert | < 15% 误差 |

### 性能验证

- 单次 GEMM 评估 < 1ms
- 完整模拟 < 5s (使用缓存)

### 兼容性验证

- 无微架构参数时，行为与原有实现一致
- API 无破坏性变更
