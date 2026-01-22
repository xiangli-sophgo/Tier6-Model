"""
预定义硬件微架构配置
"""

from .arch_config import AcceleratorMicroArch, CommunicationLatency


# ==================== 算能 SG2260E ====================
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
        dram_bandwidth_bytes=273e9,  # 273 GB/s (峰值带宽)
        lane_num=16,
        align_bytes=32,
        compute_dma_overlap_rate=0.8,
        # 通信带宽
        intra_bw=504e9,  # 组内带宽 504 GB/s (高速互联)
        inter_bw=100e9,  # 组间带宽 100 GB/s (跨节点)
        intra_latency_us=1.0,  # 组内延迟 1 us (粗粒度)
        inter_latency_us=2.0,  # 组间延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 对齐 DS_TPU
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: 芯片间物理互联延迟
            memory_read_latency_us=0.15,  # ddr_r_lat: 显存读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: 显存写延迟
            noc_latency_us=0.05,          # noc_lat: 片上网络延迟
            die_to_die_latency_us=0.04,   # d2d_lat: Die间延迟
            # start_lat = 2*0.15 + 0.15 + 0.01 + 0.05 + 2*0.04 = 0.59 us
        ),
    )
    # 从 64 TFLOPS 反推频率
    arch.freq_ghz = arch.compute_freq_from_flops(64e12)
    return arch


SG2260E_ARCH = _create_sg2260e()


# ==================== NVIDIA H100 SXM ====================
def _create_h100() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
        name="H100 SXM",
        flops_dtype="FP16",
        num_cores=132,  # SM 数量
        cube_m=16,
        cube_k=16,
        cube_n=16,  # Tensor Core: 16×16×16
        sram_size_bytes=256 * 1024,  # 每 SM 共享内存 256KB
        sram_utilization=0.5,
        dram_bandwidth_bytes=3350e9,  # HBM3 3.35 TB/s (峰值带宽)
        lane_num=32,
        align_bytes=128,  # 128B 对齐
        compute_dma_overlap_rate=0.9,
        # 通信带宽
        intra_bw=900e9,  # NVSwitch 900 GB/s
        inter_bw=400e9,  # InfiniBand NDR 400 Gb/s = 50 GB/s per port, 8x = 400 GB/s
        intra_latency_us=1.0,  # NVLink 4.0 延迟 1 us (粗粒度)
        inter_latency_us=2.0,  # InfiniBand NDR 延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - H100 单 Die 芯片
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.10,         # c2c_lat: NVLink 4.0 物理延迟更低
            memory_read_latency_us=0.10,  # ddr_r_lat: HBM3 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM3 写延迟
            noc_latency_us=0.03,          # noc_lat: H100 片上网络延迟 (推算)
            die_to_die_latency_us=0.03,   # d2d_lat: HBM 接口延迟 (推算)
            # start_lat = 2*0.10 + 0.10 + 0.01 + 0.03 + 2*0.03 = 0.40 us
        ),
    )
    arch.freq_ghz = arch.compute_freq_from_flops(989e12)  # 989 TFLOPS FP16
    return arch


H100_SXM_ARCH = _create_h100()


# ==================== NVIDIA A100 ====================
def _create_a100() -> AcceleratorMicroArch:
    arch = AcceleratorMicroArch(
        name="A100",
        flops_dtype="FP16",
        num_cores=108,
        cube_m=16,
        cube_k=16,
        cube_n=8,
        sram_size_bytes=192 * 1024,
        sram_utilization=0.5,
        dram_bandwidth_bytes=2039e9,  # HBM2e 2.039 TB/s (峰值带宽)
        lane_num=32,
        align_bytes=128,
        compute_dma_overlap_rate=0.85,
        # 通信带宽
        intra_bw=600e9,  # NVLink 600 GB/s
        inter_bw=200e9,  # InfiniBand HDR 200 Gb/s = 25 GB/s per port, 8x = 200 GB/s
        intra_latency_us=2.0,  # NVLink 3.0 延迟 2 us (粗粒度)
        inter_latency_us=2.0,  # InfiniBand HDR 延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - A100 单 Die 芯片
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: NVLink 3.0 物理延迟
            memory_read_latency_us=0.12,  # ddr_r_lat: HBM2e 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM2e 写延迟
            noc_latency_us=0.05,          # noc_lat: A100 片上网络延迟 (推算)
            die_to_die_latency_us=0.035,  # d2d_lat: HBM 接口延迟 (推算)
            # start_lat = 2*0.15 + 0.12 + 0.01 + 0.05 + 2*0.035 = 0.55 us
        ),
    )
    arch.freq_ghz = arch.compute_freq_from_flops(312e12)  # 312 TFLOPS FP16
    return arch


A100_ARCH = _create_a100()


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


# ==================== NVIDIA A800 ====================
def _create_a800() -> AcceleratorMicroArch:
    """A800 (中国特供版 A100，NVLink 带宽受限)"""
    arch = AcceleratorMicroArch(
        name="A800",
        flops_dtype="FP16",
        num_cores=108,
        cube_m=16,
        cube_k=16,
        cube_n=8,
        sram_size_bytes=192 * 1024,
        sram_utilization=0.5,
        dram_bandwidth_bytes=2039e9,  # HBM2e 2.039 TB/s (峰值带宽)
        lane_num=32,
        align_bytes=128,
        compute_dma_overlap_rate=0.85,
        intra_bw=400e9,  # NVLink 受限 400 GB/s
        inter_bw=200e9,
        intra_latency_us=2.0,  # NVLink 延迟 2 us (粗粒度)
        inter_latency_us=2.0,  # InfiniBand HDR 延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 与 A100 相同 (带宽受限不影响延迟)
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: NVLink 3.0 物理延迟
            memory_read_latency_us=0.12,  # ddr_r_lat: HBM2e 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM2e 写延迟
            noc_latency_us=0.05,          # noc_lat: 片上网络延迟 (推算)
            die_to_die_latency_us=0.035,  # d2d_lat: HBM 接口延迟 (推算)
            # start_lat = 2*0.15 + 0.12 + 0.01 + 0.05 + 2*0.035 = 0.55 us
        ),
    )
    arch.freq_ghz = arch.compute_freq_from_flops(312e12)
    return arch


A800_ARCH = _create_a800()


# ==================== 华为 Ascend 910B ====================
def _create_ascend_910b() -> AcceleratorMicroArch:
    """华为昇腾 910B"""
    arch = AcceleratorMicroArch(
        name="Ascend-910B",
        flops_dtype="BF16",
        num_cores=32,  # AI Core 数量
        cube_m=16,
        cube_k=16,
        cube_n=16,
        sram_size_bytes=512 * 1024,  # 每核 512KB
        sram_utilization=0.5,
        dram_bandwidth_bytes=1600e9,  # HBM2e 1.6 TB/s (峰值带宽)
        lane_num=16,
        align_bytes=32,
        compute_dma_overlap_rate=0.85,
        intra_bw=392e9,  # HCCS 7×56 GB/s
        inter_bw=200e9,
        intra_latency_us=2.0,  # HCCS 延迟 2 us (粗粒度)
        inter_latency_us=2.0,  # 跨节点延迟 2 us (粗粒度)
        # 细粒度通信延迟 (单位: us) - 推算值
        comm_latency=CommunicationLatency(
            chip_to_chip_us=0.15,         # c2c_lat: HCCS 物理延迟 (推算)
            memory_read_latency_us=0.13,  # ddr_r_lat: HBM2e 读延迟
            memory_write_latency_us=0.01, # ddr_w_lat: HBM2e 写延迟
            noc_latency_us=0.04,          # noc_lat: 片上网络延迟 (推算)
            die_to_die_latency_us=0.035,  # d2d_lat: Die间延迟 (推算)
            # start_lat = 2*0.15 + 0.13 + 0.01 + 0.04 + 2*0.035 = 0.55 us
        ),
    )
    arch.freq_ghz = arch.compute_freq_from_flops(320e12)  # 320 TFLOPS BF16
    return arch


ASCEND_910B_ARCH = _create_ascend_910b()


# 配置查找表
ARCH_PRESETS = {
    "sg2260e": SG2260E_ARCH,
    "sg2261": SG2261_ARCH,
    "sg2262": SG2262_ARCH,
    "h100": H100_SXM_ARCH,
    "h100-sxm": H100_SXM_ARCH,  # 前端使用横线
    "a100": A100_ARCH,
    "a800": A800_ARCH,
    "ascend-910b": ASCEND_910B_ARCH,  # 前端使用横线
}


def get_arch_preset(name: str) -> AcceleratorMicroArch:
    """获取预定义配置"""
    name_lower = name.lower()
    if name_lower not in ARCH_PRESETS:
        available = list(ARCH_PRESETS.keys())
        raise ValueError(f"未知硬件类型: {name}，可选: {available}")
    return ARCH_PRESETS[name_lower]
