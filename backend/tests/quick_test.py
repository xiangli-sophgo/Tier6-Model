#!/usr/bin/env python3
"""快速验证 Tier6 评估器的计算时间是否正确"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
from llm_simulator.evaluators.gemm_eval import GEMMEvaluator

# 创建测试架构
arch = AcceleratorMicroArch(
    name="Test",
    num_cores=64,
    cube_m=32, cube_n=32, cube_k=32,
    sram_size_bytes=8*1024*1024,
    sram_utilization=0.45,
    lane_num=32, align_bytes=64,
    freq_ghz=1.2,
    dram_bandwidth_bytes=273e9,
    compute_dma_overlap_rate=0.5,
)

evaluator = GEMMEvaluator(arch, enable_partition_search=True, enable_tile_search=True)

print("测试算子: (1, 384, 7168, 2048)")
result = evaluator.evaluate(
    1, 384, 7168, 2048,
    input_dtype='fp8',
    output_dtype='bf16',
    use_multiprocess=False  # 串行，便于调试
)

print(f"\n结果:")
print(f"  partition: {result.best_partition}")
print(f"  tile: {result.best_tile}")
print(f"  计算时间: {result.compute_time_us:.2f} μs")
print(f"  搬运时间: {result.memory_time_us:.2f} μs")
print(f"  总延迟: {result.latency_us:.2f} μs")

# 预期值（根据公式验证）
print(f"\n预期值（partition 1,1,8,8）:")
print(f"  计算时间: 71.68 μs")
print(f"  搬运时间: 180.52 μs")

# 检查
if abs(result.compute_time_us - 71.68) < 1:
    print(f"\n✅ 计算时间正确！")
else:
    print(f"\n❌ 计算时间错误！预期 71.68，实际 {result.compute_time_us:.2f}")
