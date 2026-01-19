"""
GEMM 评估器工具函数
"""


def ceil_div(x: int, y: int) -> int:
    """向上取整除法"""
    return (x + y - 1) // y


def align_up(x: int, alignment: int) -> int:
    """向上对齐到 alignment 的倍数"""
    if alignment <= 0:
        return x
    return ((x + alignment - 1) // alignment) * alignment


def flops_gemm(m: int, n: int, k: int) -> int:
    """计算 GEMM 的 FLOPs (2×M×N×K)"""
    return 2 * m * n * k
