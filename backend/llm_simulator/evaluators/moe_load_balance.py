"""
MoE 负载均衡评估器

用途：
    计算 MoE 推理时最忙芯片需要加载的专家数，解决专家路由随机性导致的负载不均问题

核心问题：
    - MoE 的 Router 网络会为每个 token 随机选择 Top-K 个专家
    - 专家分布到多个芯片（EP 并行），每个芯片负责一部分专家
    - 由于路由的随机性，某些芯片会被调用更多次，需要加载更多不同的专家
    - 最慢的芯片决定总延迟（木桶效应）

物理意义：
    MAX_EXPERT_TABLE[batch_size][chips] = 最忙芯片需要加载的不同专家个数

    例如：batch=4, chips=32 → 3.18
    含义：32 个芯片中，最忙的那个芯片需要加载约 3.18 个不同专家的参数并计算

使用方法：
    1. 调用 get_max_expert_load(batch_size, chips) 获取专家加载数
    2. 用于计算：
       - GEMM 的 G 维度（专家并行维度）
       - 专家参数搬运时间 = max_experts × expert_param_size / dram_bandwidth
       - MoE 层总延迟 = max(计算时间, 搬运时间) + 重叠惩罚

数据来源：
    - 蒙特卡洛模拟（10000 次迭代）
    - 模拟 DeepSeek V3 配置：256 专家，Top-8 路由
    - 已验证：模拟值与表值误差 < 1%

参考：
    DS_TPU_1209/model.py:get_max_expert()
"""

import math
import random
from typing import Optional, Dict, Tuple


# ============================================================================
# 硬编码查找表（从 DS_TPU 提取，已通过蒙特卡洛验证）
# ============================================================================

MAX_EXPERT_TABLE: Dict[int, Dict[int, float]] = {
    # batch_size: {chips: max_experts}
    #
    # 规律：
    # 1. batch 越小，负载越不均（max_experts 越大）
    # 2. chips 越多，负载越分散（max_experts 越小）
    # 3. chips=256 时固定为 1.0（每芯片只能加载自己的 1 个专家）
    # 4. 大 batch + 小 chips 时接近理论极限（256/chips）

    4: {
        1: 30.5121,    # 单芯片：32 次调用覆盖 30.5 个专家（生日悖论）
        2: 17.3445,
        4: 10.3651,
        8: 6.575,
        16: 4.44845,
        32: 3.18425,   # 32 芯片：最忙的加载 3.18 个专家
        64: 2.326,
        128: 1.8603,
        256: 1.0       # 256 芯片：每芯片只能加载 1 个专家
    },
    8: {
        1: 57.4309,
        2: 31.40505,
        4: 17.81475,
        8: 10.6518,
        16: 6.69615,
        32: 4.4437,
        64: 3.0746,
        128: 1.9997,
        256: 1.0
    },
    12: {
        1: 81.11235,
        2: 43.49865,
        4: 24.12945,
        8: 13.95695,
        16: 8.4397,
        32: 5.35915,
        64: 3.47415,
        128: 2.0,
        256: 1.0
    },
    16: {
        1: 101.9848,
        2: 54.1023,
        4: 29.53585,
        8: 16.7107,
        16: 9.86765,
        32: 6.05895,
        64: 3.8144,
        128: 2.0,
        256: 1.0
    },
    24: {
        1: 136.4634,
        2: 71.4372,
        4: 38.2351,
        8: 21.0515,
        16: 11.9747,
        32: 7.01615,
        64: 3.99745,
        128: 2.0,
        256: 1.0
    },
    32: {
        1: 163.27965,
        2: 84.75245,
        4: 44.79135,
        8: 24.19435,
        16: 13.42475,
        32: 7.5834,
        64: 4.0,
        128: 2.0,
        256: 1.0
    },
    40: {
        1: 184.0948,
        2: 94.89465,
        4: 49.68275,
        8: 26.51455,
        16: 14.41565,
        32: 7.91665,
        64: 4.0,
        128: 2.0,
        256: 1.0
    },
    48: {
        1: 200.20185,
        2: 102.74895,
        4: 53.3845,
        8: 28.1981,
        16: 15.1102,
        32: 7.9938,
        64: 4.0,
        128: 2.0,
        256: 1.0
    },
    64: {
        1: 222.417,
        2: 113.352,
        4: 58.32715,
        8: 30.30105,
        16: 15.8369,
        32: 8.0,       # 理论极限：256 专家 / 32 芯片 = 8
        64: 4.0,
        128: 2.0,
        256: 1.0
    },
    128: {
        1: 251.59985,
        2: 126.611,
        4: 63.7934,
        8: 31.9994,
        16: 16.0,      # 开始接近理论极限
        32: 8.0,
        64: 4.0,
        128: 2.0,
        256: 1.0
    },
    256: {
        1: 255.9269,   # 接近 256（几乎激活所有专家）
        2: 127.9988,   # 接近 128
        4: 64.0,       # 达到理论极限
        8: 32.0,
        16: 16.0,
        32: 8.0,
        64: 4.0,
        128: 2.0,
        256: 1.0
    }
}


# ============================================================================
# 蒙特卡洛模拟（兜底方案）
# ============================================================================

def monte_carlo_max_experts(
    batch_size: int,
    chips: int,
    num_experts: int = 256,
    topk: int = 8,
    iterations: int = 1000,
    seed: Optional[int] = None
) -> float:
    """
    蒙特卡洛模拟：计算最忙芯片需要加载的专家数（期望值）

    用途：
        当查找表中没有对应的 batch/chips 配置时，临时模拟计算

    算法：
        1. 随机生成 batch_size 个 token 的专家选择（每个选 topk 个）
        2. 统计每个芯片被激活的不同专家集合
        3. 记录最忙芯片的专家个数
        4. 重复 iterations 次，返回期望值

    Args:
        batch_size: token 数量
        chips: EP 芯片数（专家并行度）
        num_experts: 专家总数（DeepSeek V3 固定 256）
        topk: 每 token 选择的专家数（DeepSeek V3 固定 8）
        iterations: 模拟次数（越多越精确，建议 1000-10000）
        seed: 随机种子（用于可重复性）

    Returns:
        最忙芯片需要加载的不同专家个数（期望值）

    示例：
        >>> monte_carlo_max_experts(batch_size=4, chips=32)
        3.18  # 最忙芯片需要加载约 3.18 个专家

    性能：
        - iterations=1000: 约 10-50ms
        - iterations=10000: 约 100-500ms
    """
    if seed is not None:
        random.seed(seed)

    max_experts_list = []
    experts_per_chip = num_experts // chips  # 每芯片负责的专家数

    for _ in range(iterations):
        # 每个芯片被激活的专家集合
        chip_experts = [set() for _ in range(chips)]

        # 模拟 batch_size 个 token 的路由
        for _ in range(batch_size):
            # 随机选择 topk 个专家（模拟 Router 网络）
            selected_experts = random.sample(range(num_experts), topk)

            # 将专家分配到对应芯片
            for expert_id in selected_experts:
                chip_id = expert_id // experts_per_chip
                chip_experts[chip_id].add(expert_id)

        # 统计最忙的芯片激活的专家数
        max_experts = max(len(experts) for experts in chip_experts)
        max_experts_list.append(max_experts)

    # 返回期望值
    return sum(max_experts_list) / len(max_experts_list)


# ============================================================================
# 线性插值（提升覆盖率）
# ============================================================================

def _interpolate_batch(
    batch_size: int,
    chips: int
) -> Optional[float]:
    """
    batch_size 维度的线性插值

    策略：
        在表中找到 batch_size 的前后两个采样点，进行线性插值

    Args:
        batch_size: 目标 batch 大小
        chips: 芯片数（必须在表中）

    Returns:
        插值结果，如果无法插值则返回 None

    示例：
        batch=10, chips=32 在表中
        找到 batch=8 和 batch=12 的值
        插值：value = v8 + (v12 - v8) × (10 - 8) / (12 - 8)
    """
    # 找到 batch_size 的前后采样点
    batch_points = sorted(MAX_EXPERT_TABLE.keys())

    if batch_size <= batch_points[0]:
        # 小于最小值，用最小值
        if chips in MAX_EXPERT_TABLE[batch_points[0]]:
            return MAX_EXPERT_TABLE[batch_points[0]][chips]
        return None

    if batch_size >= batch_points[-1]:
        # 大于最大值，用最大值
        if chips in MAX_EXPERT_TABLE[batch_points[-1]]:
            return MAX_EXPERT_TABLE[batch_points[-1]][chips]
        return None

    # 找到前后两个点
    for i in range(len(batch_points) - 1):
        b1, b2 = batch_points[i], batch_points[i + 1]
        if b1 <= batch_size <= b2:
            # 检查 chips 是否在两个点都存在
            if chips in MAX_EXPERT_TABLE[b1] and chips in MAX_EXPERT_TABLE[b2]:
                v1 = MAX_EXPERT_TABLE[b1][chips]
                v2 = MAX_EXPERT_TABLE[b2][chips]
                # 线性插值
                ratio = (batch_size - b1) / (b2 - b1)
                return v1 + (v2 - v1) * ratio
            return None

    return None


def _interpolate_chips(
    batch_size: int,
    chips: int
) -> Optional[float]:
    """
    chips 维度的线性插值（对数空间）

    注意：
        chips 通常是 2 的幂次（1, 2, 4, 8, ...），使用对数插值更准确

    Args:
        batch_size: batch 大小（必须在表中）
        chips: 目标芯片数

    Returns:
        插值结果，如果无法插值则返回 None
    """
    if batch_size not in MAX_EXPERT_TABLE:
        return None

    chip_points = sorted(MAX_EXPERT_TABLE[batch_size].keys())

    if chips <= chip_points[0]:
        return MAX_EXPERT_TABLE[batch_size][chip_points[0]]

    if chips >= chip_points[-1]:
        return MAX_EXPERT_TABLE[batch_size][chip_points[-1]]

    # 找到前后两个点（对数空间）
    for i in range(len(chip_points) - 1):
        c1, c2 = chip_points[i], chip_points[i + 1]
        if c1 <= chips <= c2:
            v1 = MAX_EXPERT_TABLE[batch_size][c1]
            v2 = MAX_EXPERT_TABLE[batch_size][c2]
            # 对数插值（chips 是 2 的幂次）
            log_ratio = (math.log2(chips) - math.log2(c1)) / (math.log2(c2) - math.log2(c1))
            return v1 + (v2 - v1) * log_ratio

    return None


def _try_interpolate(
    batch_size: int,
    chips: int
) -> Optional[float]:
    """
    尝试插值（优先 batch 维度，其次 chips 维度）

    策略：
        1. 如果 chips 在表中，对 batch_size 插值
        2. 如果 batch_size 在表中，对 chips 插值
        3. 都不在表中，返回 None

    Args:
        batch_size: batch 大小
        chips: 芯片数

    Returns:
        插值结果，失败返回 None
    """
    # 优先尝试 batch 维度插值
    result = _interpolate_batch(batch_size, chips)
    if result is not None:
        return result

    # 其次尝试 chips 维度插值
    result = _interpolate_chips(batch_size, chips)
    if result is not None:
        return result

    return None


# ============================================================================
# 主查询接口
# ============================================================================

def get_max_expert_load(
    batch_size: int,
    chips: int,
    allow_simulation: bool = True,
    simulation_iterations: int = 1000
) -> float:
    """
    获取最忙芯片需要加载的专家数

    查询策略（三级回退）：
        1. 优先查找表（精确匹配）
        2. 尝试插值（扩展覆盖范围）
        3. 蒙特卡洛模拟（兜底方案）

    Args:
        batch_size: 当前处理的 token 数量
        chips: EP 芯片数（专家并行度）
        allow_simulation: 是否允许蒙特卡洛模拟（默认 True）
        simulation_iterations: 模拟迭代次数（默认 1000）

    Returns:
        最忙芯片需要加载的专家个数（浮点数）

    使用示例：
        >>> # Decode 阶段：batch=4, EP=32
        >>> max_experts = get_max_expert_load(batch_size=4, chips=32)
        >>> print(max_experts)  # 3.18
        >>>
        >>> # 用于 GEMM 评估
        >>> gemm_result = gemm_evaluator.evaluate(
        ...     G=math.ceil(max_experts),  # 向上取整到 4
        ...     M=tokens_per_expert,
        ...     K=hidden_dim,
        ...     N=expert_intermediate_size / moe_tp
        ... )
        >>>
        >>> # 用于计算权重搬运
        >>> expert_param_size = 3 * hidden_dim * expert_intermediate_size * dtype_bytes
        >>> weight_load_time_us = max_experts * expert_param_size / dram_bandwidth_gbps * 1e6

    注意事项：
        1. batch_size 超过 256 会被截断到 256
        2. 返回值是浮点数，用于 GEMM 时需要向上取整 math.ceil()
        3. 计算权重搬运时使用原始浮点数（更精确）
        4. 仅适用于 DeepSeek V3 配置（256 专家，Top-8）

    性能：
        - 查表命中：O(1)，< 1μs
        - 插值：O(log n)，< 10μs
        - 模拟：O(iterations)，10-500ms
    """
    # 截断到表的最大值
    batch_size = min(batch_size, 256)

    # 策略 1: 精确查表
    if batch_size in MAX_EXPERT_TABLE:
        if chips in MAX_EXPERT_TABLE[batch_size]:
            return MAX_EXPERT_TABLE[batch_size][chips]

    # 策略 2: 线性插值
    interpolated = _try_interpolate(batch_size, chips)
    if interpolated is not None:
        return interpolated

    # 策略 3: 蒙特卡洛模拟（兜底）
    if allow_simulation:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"MoE 负载均衡表中未找到 batch={batch_size}, chips={chips}，"
            f"使用蒙特卡洛模拟（iterations={simulation_iterations}）"
        )
        return monte_carlo_max_experts(
            batch_size=batch_size,
            chips=chips,
            iterations=simulation_iterations
        )

    # 如果不允许模拟，返回保守估计（理论极限）
    import logging
    logger = logging.getLogger(__name__)
    logger.error(
        f"MoE 负载均衡表中未找到 batch={batch_size}, chips={chips}，"
        f"且禁用了模拟，返回保守估计"
    )
    return 256.0 / chips  # 理论极限：所有专家都被激活


# ============================================================================
# 便捷接口（针对不同场景）
# ============================================================================

def get_max_expert_load_for_moe_layer(
    batch_size: int,
    ep_parallelism: int,
    num_experts: int = 256,
    topk: int = 8
) -> float:
    """
    针对 MoE 层的便捷接口

    参数验证 + 语义化命名

    Args:
        batch_size: 全局 batch 大小（会除以 DP）
        ep_parallelism: EP 并行度（芯片数）
        num_experts: 专家总数（默认 256）
        topk: Top-K 专家数（默认 8）

    Returns:
        最忙芯片需要加载的专家数

    Raises:
        ValueError: 如果配置不是 DeepSeek V3（256 专家，Top-8）
    """
    # 验证配置
    if num_experts != 256 or topk != 8:
        raise ValueError(
            f"当前查找表仅支持 DeepSeek V3 配置（256 专家，Top-8），"
            f"当前配置：num_experts={num_experts}, topk={topk}"
        )

    return get_max_expert_load(batch_size, ep_parallelism)


def estimate_moe_expert_load_impact(
    batch_size: int,
    chips: int
) -> Dict[str, float]:
    """
    评估负载不均的影响程度

    返回详细的负载统计，用于分析和调试

    Args:
        batch_size: token 数量
        chips: 芯片数

    Returns:
        包含以下字段的字典：
        - max_experts: 最忙芯片的专家数
        - avg_experts: 平均每芯片的专家数（理论值）
        - load_factor: 负载因子（max / avg）
        - imbalance_ratio: 负载不均衡比例（1.0 = 完全均衡）
    """
    max_experts = get_max_expert_load(batch_size, chips)

    # 理论平均值
    total_calls = batch_size * 8  # 8 = topk
    avg_experts_theoretical = total_calls / chips

    # 负载因子
    load_factor = max_experts / avg_experts_theoretical if avg_experts_theoretical > 0 else 1.0

    return {
        "max_experts": max_experts,
        "avg_experts": avg_experts_theoretical,
        "load_factor": load_factor,
        "imbalance_ratio": load_factor,
    }


# ============================================================================
# 测试和验证
# ============================================================================

if __name__ == "__main__":
    # 快速验证
    print("MoE 负载均衡查询示例：")
    print("=" * 60)

    test_cases = [
        (4, 1, "极小 batch + 单芯片"),
        (4, 32, "极小 batch + 中等并行"),
        (64, 32, "中等 batch + 中等并行"),
        (256, 32, "大 batch + 中等并行"),
        (256, 256, "大 batch + 极高并行"),
    ]

    for batch, chips, desc in test_cases:
        result = get_max_expert_load(batch, chips)
        impact = estimate_moe_expert_load_impact(batch, chips)
        print(f"\n{desc}")
        print(f"  batch={batch}, chips={chips}")
        print(f"  最忙芯片加载专家数: {result:.2f}")
        print(f"  负载因子: {impact['load_factor']:.2f}x")
