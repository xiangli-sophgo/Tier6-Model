"""
预定义硬件微架构配置
"""

from .arch_config import AcceleratorMicroArch


# ==================== 算能 SG2260E ====================
def _create_sg2260e() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
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
        # 通信带宽
        intra_bw=504e9,   # 组内带宽 504 GB/s (高速互联)
        inter_bw=100e9,   # 组间带宽 100 GB/s (跨节点)
    )
    # 从 64 TFLOPS 反推频率
    arch.freq_ghz = arch.compute_freq_from_flops(64e12)
    return arch


SG2260E_ARCH = _create_sg2260e()


# ==================== NVIDIA H100 SXM ====================
def _create_h100() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
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
        # 通信带宽
        intra_bw=900e9,   # NVSwitch 900 GB/s
        inter_bw=400e9,   # InfiniBand NDR 400 Gb/s = 50 GB/s per port, 8x = 400 GB/s
    )
    arch.freq_ghz = arch.compute_freq_from_flops(989e12)  # 989 TFLOPS FP16
    return arch


H100_SXM_ARCH = _create_h100()


# ==================== NVIDIA A100 ====================
def _create_a100() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
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
        # 通信带宽
        intra_bw=600e9,   # NVLink 600 GB/s
        inter_bw=200e9,   # InfiniBand HDR 200 Gb/s = 25 GB/s per port, 8x = 200 GB/s
    )
    arch.freq_ghz = arch.compute_freq_from_flops(312e12)  # 312 TFLOPS FP16
    return arch


A100_ARCH = _create_a100()


# 配置查找表
ARCH_PRESETS = {
    'sg2260e': SG2260E_ARCH,
    'sg2260': SG2260E_ARCH,
    'h100': H100_SXM_ARCH,
    'h100_sxm': H100_SXM_ARCH,
    'a100': A100_ARCH,
}


def get_arch_preset(name: str) -> AcceleratorMicroArch:
    """获取预定义配置"""
    name_lower = name.lower()
    if name_lower not in ARCH_PRESETS:
        available = list(ARCH_PRESETS.keys())
        raise ValueError(f"未知硬件类型: {name}，可选: {available}")
    return ARCH_PRESETS[name_lower]
