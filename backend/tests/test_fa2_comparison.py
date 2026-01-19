"""
FA2 评估器对比测试

对比 Tier6+Model/evaluators/fa2_eval.py 与 DS_TPU_1209/performance/evaluate/compute/flash_attention/fa2_eval.py
确保两者功能一致、结果一致
"""

import sys
import math

sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

# ==================== DS_TPU 实现 (内联) ====================

FP8_BYTES = 1
BF16_BYTES = 2
FP32_BYTES = 4


def ceil_div(x, y):
    return (x + y - 1) // y


def align_up(x, y):
    return ((x + y - 1) // y) * y


# Softmax 步骤
SOFTMAX_STEPS = [
    ('add', 0, 1),
    ('reduce_max', 1, 1),
    ('max', 1, 1),
    ('fuse_exp', 0, 35),
    ('fuse_exp', 1, 35),
    ('reduce_sum', 1, 1),
    ('mul', 1, 1),
    ('add', 1, 1),
    ('copy', 1, 1),
    ('data_convert', 0, 1),
]


def softmax_theoretical_and_real_ds(QS, KS, lane_num, eu_num, dtype_bytes):
    """DS_TPU softmax 估算"""
    def calc_step_theo(shape_type, op_count):
        if shape_type == 0:
            return (
                align_up(QS, lane_num) *
                align_up(KS, eu_num // lane_num // dtype_bytes) *
                op_count
            )
        else:
            return align_up(QS, lane_num) * op_count

    def calc_step_real(shape_type, op_count):
        if shape_type == 0:
            return QS * 1 * KS * op_count
        else:
            return QS * 1 * 1 * op_count

    vector_theo, vector_real = 0, 0
    for _, shape_type, op_count in SOFTMAX_STEPS:
        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)
    return vector_theo, vector_real


class TPUConfigBase:
    """TPU 配置基类"""
    def __init__(self, **kwargs):
        self.core = kwargs.get('core', 32)
        self.flops = kwargs.get('flops', 256.0 * 1024 * 1e9)
        self.dram_bw = kwargs.get('dram_bw', 4000e9 * 0.80)

        self.tpu_cores = self.core
        self.cube_m = kwargs.get('cube_m', 16)
        self.cube_k = kwargs.get('cube_k', 32)
        self.cube_n = kwargs.get('cube_n', 8)
        self.sram_size = kwargs.get('sram_size', 2 * (1 << 20))
        self.lane_num = kwargs.get('lane_num', 16)
        self.eu_num = kwargs.get('eu_num', 512)
        self.align_bytes = kwargs.get('align_bytes', 32)

        self.dma_bw = self.dram_bw / self.tpu_cores
        self.macs_per_cycle = self.cube_m * self.cube_k * self.cube_n
        self.freq = self.flops / (2.0 * self.core * self.macs_per_cycle * 1e9)
        self.tpu_gdma_overlap_rate = kwargs.get('tpu_gdma_overlap_rate', 0.8)


class FA2Eval_DS:
    """DS_TPU FA2 评估器 (简化单线程版本)"""

    def __init__(self, arch):
        self.arch = arch
        self.all_valid_partitions = self._valid_partition()

    def _valid_partition(self):
        blocks = []
        for P_B in range(1, self.arch.tpu_cores + 1):
            if self.arch.tpu_cores % P_B == 0:
                blocks.append(P_B)
        return blocks

    def legal_tiles(self, QS, KS, QD, VD):
        all_tiles = []
        cube_m = self.arch.cube_m
        cube_n = self.arch.cube_n
        cube_k = self.arch.cube_k
        max_cube_nk = max(cube_n, cube_k)
        sram_limit = self.arch.sram_size * 0.45

        for q_t in range(align_up(QS, cube_m), 0, -cube_m):
            align_row_q = align_up(q_t, self.arch.lane_num)
            for k_t in range(align_up(KS, max_cube_nk), 0, -max_cube_nk):
                align_row_k = align_up(k_t, self.arch.lane_num)
                q_block = align_row_q * align_up(QD * FP8_BYTES, self.arch.align_bytes)
                k_block = align_row_k * align_up(QD * FP8_BYTES, self.arch.align_bytes)
                v_block = align_row_k * align_up(VD * FP8_BYTES, self.arch.align_bytes)
                p_block = align_row_q * align_up(k_t * BF16_BYTES, self.arch.align_bytes)
                o_block = align_row_q * align_up(VD * BF16_BYTES, self.arch.align_bytes)
                occupied = q_block + k_block + v_block + 2 * p_block + 4 * o_block

                if occupied > sram_limit:
                    continue
                if self._is_pareto_max(all_tiles, q_t, k_t):
                    all_tiles.append((q_t, k_t))

        return all_tiles if all_tiles else [(min(64, QS), min(64, KS))]

    def _is_pareto_max(self, conds, q_t, k_t):
        for q0, k0 in conds:
            if q0 >= q_t and k0 >= k_t:
                return False
        return True

    def dram_traffic(self, QS, KS, QD, VD, q_t, k_t):
        tile_num_q = ceil_div(QS, q_t)
        load_q = QS * QD * FP8_BYTES
        load_k = KS * QD * FP8_BYTES * tile_num_q
        load_v = KS * VD * FP8_BYTES * tile_num_q
        store_o = QS * VD * BF16_BYTES
        return load_q + load_k + load_v + store_o

    def calc_arch_urate(self, b_blk, QS, KS, QD, VD):
        if b_blk == 0:
            return 0, 0

        gemm_real = QS * KS * (QD + VD)
        gemm_theo = (
            align_up(QS, self.arch.cube_m) * align_up(QD, self.arch.cube_k) * align_up(KS, self.arch.cube_n)
            + align_up(QS, self.arch.cube_m) * align_up(KS, self.arch.cube_k) * align_up(VD, self.arch.cube_n)
        )

        vector_theo, vector_real = softmax_theoretical_and_real_ds(
            QS, KS, self.arch.lane_num, self.arch.eu_num, BF16_BYTES
        )

        arch_urate = (gemm_real + vector_real) / (gemm_theo + vector_theo) if (gemm_theo + vector_theo) > 0 else 0

        gemm_t_us = gemm_theo * b_blk / self.arch.macs_per_cycle / self.arch.freq / 1e3
        vector_t_us = vector_theo * b_blk / self.arch.eu_num / self.arch.freq / BF16_BYTES / 1e3
        t_us = gemm_t_us + vector_t_us

        return arch_urate, t_us

    def evaluate_partition(self, P_B, B, QS, KS, QD, VD):
        # 搜索最佳 tile
        min_traffic = math.inf
        best_tile = None
        for (q_t, k_t) in self.legal_tiles(QS, KS, QD, VD):
            traffic = self.dram_traffic(QS, KS, QD, VD, q_t, k_t)
            if traffic < min_traffic:
                min_traffic = traffic
                best_tile = (q_t, k_t)

        if best_tile is None:
            best_tile = (min(64, QS), min(64, KS))

        q_t, k_t = best_tile
        b_nom = ceil_div(B, P_B)

        total_flops = 0
        total_traffic = 0.0
        max_time = 0
        best_comp_elapse = 0
        best_dma_elapse = 0

        for i_b in range(P_B):
            b_start = i_b * b_nom
            b_blk = max(min(B - b_start, b_nom), 0)

            if b_blk == 0:
                continue

            flops = 2 * b_blk * QS * KS * (QD + VD)
            arch_urate, comp_elapse = self.calc_arch_urate(b_blk, QS, KS, QD, VD)
            traffic = b_blk * self.dram_traffic(QS, KS, QD, VD, q_t, k_t)
            dma_elapse = 1e6 * traffic / self.arch.dma_bw if self.arch.dma_bw > 0 else 1e6
            t_total = (min(comp_elapse, dma_elapse) * (1 - self.arch.tpu_gdma_overlap_rate) +
                       max(comp_elapse, dma_elapse))

            if t_total > max_time:
                best_comp_elapse = comp_elapse
                best_dma_elapse = dma_elapse
            max_time = max(max_time, t_total)
            total_flops += flops
            total_traffic += traffic

        if max_time == 0:
            urate = 0
        else:
            urate = total_flops / (max_time * 1e3 * self.arch.tpu_cores *
                                   self.arch.macs_per_cycle * self.arch.freq * 2)

        return {
            'elapse': max_time,
            'urate': urate,
            'flops': total_flops,
            'traffic': total_traffic,
            'comp_elapse': best_comp_elapse,
            'dma_elapse': best_dma_elapse,
            'best_tile': best_tile,
            'P_B': P_B,
        }

    def eval_p(self, B, QS, KS, QD, VD):
        min_time = math.inf
        best_result = None

        for P_B in self.all_valid_partitions:
            result = self.evaluate_partition(P_B, B, QS, KS, QD, VD)
            if result['elapse'] < min_time and result['elapse'] > 0:
                min_time = result['elapse']
                best_result = result

        return best_result


# ==================== 对比测试 ====================

from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
from llm_simulator.evaluators.fa2_eval import FA2Evaluator


def create_matching_configs():
    """创建匹配的配置"""
    ds_config = TPUConfigBase(
        core=64,
        flops=64.0 * 1e12,
        dram_bw=500e9,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        sram_size=16 * (1 << 20),
        lane_num=64,
        eu_num=512,
        align_bytes=128,
        tpu_gdma_overlap_rate=0.8,
    )

    freq_ghz = ds_config.freq

    tier6_config = AcceleratorMicroArch(
        num_cores=64,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        freq_ghz=freq_ghz,
        eu_num=512,
        sram_size_bytes=16 * (1 << 20),
        sram_utilization=0.45,
        dram_bandwidth_bytes=500e9,
        lane_num=64,
        align_bytes=128,
        compute_dma_overlap_rate=0.8,
    )

    return ds_config, tier6_config


def compare_valid_partitions(ds_eval, tier6_eval):
    """对比有效分区枚举"""
    print("=" * 60)
    print("1. 对比有效分区枚举")
    print("=" * 60)

    ds_parts = set(ds_eval.all_valid_partitions)
    tier6_parts = set(tier6_eval._valid_partitions)

    print(f"DS_TPU 分区数: {len(ds_parts)}")
    print(f"Tier6+ 分区数: {len(tier6_parts)}")

    if ds_parts == tier6_parts:
        print("✓ 分区枚举完全一致!")
        return True
    else:
        print("✗ 分区枚举不一致!")
        return False


def compare_full_evaluation(ds_eval, tier6_eval, B, QS, KS, QD, VD):
    """对比完整评估"""
    print(f"\n完整评估 FA2[B={B}, QS={QS}, KS={KS}, QD={QD}, VD={VD}]:")

    # DS_TPU
    ds_result = ds_eval.eval_p(B, QS, KS, QD, VD)

    # Tier6+
    tier6_result = tier6_eval.evaluate(B, QS, KS, QD, VD)

    if ds_result:
        print(f"\n  DS_TPU 结果:")
        print(f"    延迟: {ds_result['elapse']:.3f} μs")
        print(f"    利用率: {ds_result['urate']:.4f}")
        print(f"    计算时间: {ds_result['comp_elapse']:.3f} μs")
        print(f"    DMA时间: {ds_result['dma_elapse']:.3f} μs")
        print(f"    FLOPS: {ds_result['flops']:,}")
        print(f"    流量: {ds_result['traffic']:.0f} bytes")
        print(f"    最佳Tile: {ds_result['best_tile']}")
        print(f"    最佳分区: P_B={ds_result['P_B']}")

    print(f"\n  Tier6+ 结果:")
    print(f"    延迟: {tier6_result.latency_us:.3f} μs")
    print(f"    利用率: {tier6_result.effective_utilization:.4f}")
    print(f"    计算时间: {tier6_result.compute_time_us:.3f} μs")
    print(f"    DMA时间: {tier6_result.memory_time_us:.3f} μs")
    print(f"    FLOPS: {tier6_result.flops:,}")
    print(f"    流量: {tier6_result.dram_traffic_bytes:,} bytes")
    print(f"    最佳Tile: {tier6_result.best_tile}")
    print(f"    最佳分区: P_B={tier6_result.best_partition}")

    # 对比
    if ds_result:
        latency_diff = abs(ds_result['elapse'] - tier6_result.latency_us)
        latency_rel_diff = latency_diff / ds_result['elapse'] * 100 if ds_result['elapse'] > 0 else 0

        print(f"\n  对比:")
        print(f"    延迟差异: {latency_diff:.3f} μs ({latency_rel_diff:.2f}%)")

        is_match = latency_rel_diff < 10.0
        print(f"    结果: {'✓ 匹配' if is_match else '✗ 不匹配'}")
        return is_match

    return False


def run_comprehensive_comparison():
    """运行全面的对比测试"""
    print("=" * 60)
    print("FA2 评估器对比测试")
    print("DS_TPU_1209 vs Tier6+Model")
    print("=" * 60)

    ds_config, tier6_config = create_matching_configs()

    print("\n配置信息:")
    print(f"  核心数: {ds_config.tpu_cores}")
    print(f"  FLOPS: {ds_config.flops / 1e12:.1f} TFLOPS")
    print(f"  频率: {ds_config.freq:.3f} GHz")
    print(f"  DRAM带宽: {ds_config.dram_bw / 1e9:.1f} GB/s")
    print(f"  EU数量: {ds_config.eu_num}")

    ds_eval = FA2Eval_DS(ds_config)
    tier6_eval = FA2Evaluator(tier6_config)

    # 1. 对比分区枚举
    partition_match = compare_valid_partitions(ds_eval, tier6_eval)

    # 2. 完整评估对比
    print("\n" + "=" * 60)
    print("2. 完整评估对比")
    print("=" * 60)

    test_cases = [
        (32, 512, 512, 128, 128),    # 标准 attention
        (32, 1024, 1024, 128, 128),  # 长序列
        (64, 256, 256, 64, 64),      # 小 head_dim
        (8, 2048, 2048, 128, 128),   # 少 heads, 长序列
        (128, 128, 512, 128, 128),   # Prefill decode 混合
    ]

    results = []
    for B, QS, KS, QD, VD in test_cases:
        is_match = compare_full_evaluation(ds_eval, tier6_eval, B, QS, KS, QD, VD)
        results.append((B, QS, KS, QD, VD, is_match))

    # 3. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for r in results if r[5])
    total = len(results)

    print(f"\n分区枚举: {'✓' if partition_match else '✗'}")
    print(f"完整评估: {passed}/{total} 通过")

    print("\n详细结果:")
    for B, QS, KS, QD, VD, is_match in results:
        status = "✓" if is_match else "✗"
        print(f"  FA2[B={B}, QS={QS}, KS={KS}, QD={QD}, VD={VD}]: {status}")

    if partition_match and passed == total:
        print("\n✓ 所有测试通过! FA2 评估器实现一致。")
        return True
    else:
        print("\n✗ 存在不一致，需要检查。")
        return False


if __name__ == "__main__":
    run_comprehensive_comparison()
