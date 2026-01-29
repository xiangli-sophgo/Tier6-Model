"""
预定义硬件微架构配置
"""

from .arch_config import AcceleratorMicroArch, CommunicationLatency


# ==================== 算能 SG2260E ====================
# 注意: 此配置与 DS_TPU v1 对齐,用于验证评估器精度
def _create_sg2260e() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
        name="SG2260E",
        flops_dtype="BF16",
        num_cores=64,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        sram_size_bytes=2 * 1024 * 1024,  # 2MB
        sram_utilization=0.45,
        # 与 DS_TPU v1 对齐的带宽和算力配置
        dram_bandwidth_bytes=4 * 4096e9 * 0.70,  # 11468.8 GB/s (与 DS_TPU 对齐)
        lane_num=16,
        align_bytes=32,
        compute_dma_overlap_rate=0.8,
        # 通信带宽
        intra_bw=448e9,  # 组内带宽 448 GB/s (与 DS_TPU 对齐)
        inter_bw=448e9,  # 组间带宽 448 GB/s (与 DS_TPU 对齐)
        c2c_bw_unidirectional_gbps=448.0,  # 单向带宽，双向 = 单向 × 2
        intra_latency_us=1.0,  # 组内延迟 1 us (粗粒度)
        inter_latency_us=2.0,  # 组间延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 对齐 DS_TPU dispatch_eval.py/combine_eval.py
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.2,          # c2c_lat: 芯片间物理互联延迟 (对齐 DS_TPU)
            memory_read_latency_us=0.15,  # ddr_r_lat: 显存读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: 显存写延迟
            noc_latency_us=0.05,          # noc_lat: 片上网络延迟
            die_to_die_latency_us=0.04,   # d2d_lat: Die间延迟
            # start_lat = 2*0.2 + 0.15 + 0.01 + 0.05 + 2*0.04 = 0.69 us
            # dispatch_combine_start_lat = 0.69 + 2*1.0 + 2*0.025 = 2.74 us (对齐 DS_TPU)
        ),
    )
    # 从 768 TFLOPS 反推频率
    arch.freq_ghz = arch.compute_freq_from_flops(768e12)
    return arch


SG2260E_ARCH = _create_sg2260e()


# ==================== 算能 SG2261 ====================
def _create_sg2261() -> AcceleratorMicroArch:
    """
    算能 SG2261 单芯片配置
    参数来源: DS_TPU config/tpu_configs/sg2261.yaml
    """
    arch = AcceleratorMicroArch(
        name="SG2261",
        flops_dtype="BF16",
        num_cores=64,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        sram_size_bytes=2 * 1024 * 1024,  # 2MB per core
        sram_utilization=0.45,
        dram_bandwidth_bytes=4000e9,  # 4000 GB/s (峰值带宽)
        lane_num=16,
        align_bytes=32,
        compute_dma_overlap_rate=0.8,
        eu_num=512,
        # 通信带宽 (单芯片默认)
        intra_bw=448e9,  # 448 GB/s
        inter_bw=100e9,  # 100 GB/s
        c2c_bw_unidirectional_gbps=448.0,  # 单向带宽，双向 = 单向 × 2
        intra_latency_us=1.0,  # SophgoLink 延迟 1 us (粗粒度)
        inter_latency_us=2.0,  # 跨节点延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 对齐 DS_TPU
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: SophgoLink 物理延迟
            memory_read_latency_us=0.15,  # ddr_r_lat: HBM2 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM2 写延迟
            noc_latency_us=0.05,          # noc_lat: 片上网络延迟
            die_to_die_latency_us=0.04,   # d2d_lat: Die间延迟
            # start_lat = 2*0.15 + 0.15 + 0.01 + 0.05 + 2*0.04 = 0.59 us
        ),
    )
    # 256 TFLOPS BF16
    arch.freq_ghz = arch.compute_freq_from_flops(256e12)
    return arch


SG2261_ARCH = _create_sg2261()


# ==================== 算能 SG2262 (多芯片) ====================
def _create_sg2262() -> AcceleratorMicroArch:
    """
    算能 SG2262 多芯片配置
    硬件参数:
    - 算力: 768T FP8, 384T BF16
    - Memory: 128GB 容量, 12TB/s 总带宽, 2MB LMEM
    - C2C BW: 448GB/s 单向, 996GB/s 双向
    """
    arch = AcceleratorMicroArch(
        name="SG2262",
        flops_dtype="BF16",
        num_cores=64,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        sram_size_bytes=2 * 1024 * 1024,  # 2MB LMEM
        sram_utilization=0.45,
        dram_bandwidth_bytes=12e12,  # 12 TB/s (峰值带宽)
        lane_num=16,
        align_bytes=32,
        compute_dma_overlap_rate=0.8,
        eu_num=512,
        # C2C 通信带宽
        intra_bw=448e9,  # 448 GB/s 单向带宽
        inter_bw=448e9,  # 448 GB/s
        c2c_bw_unidirectional_gbps=498.0,  # 单向带宽 498 GB/s，双向 = 单向 × 2 = 996 GB/s
        intra_latency_us=1.0,  # SophgoLink 延迟 1 us (粗粒度)
        inter_latency_us=1.0,  # SG2262 芯片间也是高速互联 1 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 对齐 DS_TPU
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: SophgoLink 物理延迟
            memory_read_latency_us=0.15,  # ddr_r_lat: HBM2 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM2 写延迟
            noc_latency_us=0.05,          # noc_lat: 片上网络延迟
            die_to_die_latency_us=0.04,   # d2d_lat: Die间延迟
            # start_lat = 2*0.15 + 0.15 + 0.01 + 0.05 + 2*0.04 = 0.59 us
        ),
    )
    # SG2262: 384 TFLOPS BF16 (768 TFLOPS FP8)
    arch.freq_ghz = arch.compute_freq_from_flops(384e12)
    return arch


SG2262_ARCH = _create_sg2262()


# 配置查找表 (仅保留 SG 系列芯片)
ARCH_PRESETS = {
    "sg2260e": SG2260E_ARCH,
    "sg2261": SG2261_ARCH,
    "sg2262": SG2262_ARCH,
}


def get_arch_preset(name: str) -> AcceleratorMicroArch:
    """获取预定义配置"""
    name_lower = name.lower()
    if name_lower not in ARCH_PRESETS:
        available = list(ARCH_PRESETS.keys())
        raise ValueError(f"未知硬件类型: {name}，可选: {available}")
    return ARCH_PRESETS[name_lower]
