"""
测试GEMM评估速度
"""
import time
from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator

# 1. 创建评估器
arch = get_arch_preset("SG2262")
print(f"芯片: SG2262, 核心数: {arch.num_cores}")

# 2. 测试FastGEMMEvaluator（关闭tile搜索）
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("测试 FastGEMMEvaluator (fast_mode=True)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

fast_eval = create_gemm_evaluator(arch, fast_mode=True)
print(f"评估器类型: {fast_eval.__class__.__name__}")

# DeepSeek-V3 MoE专家的典型GEMM形状
# gate: (1, 8192, 2048)
# up:   (1, 8192, 2048)
# down: (1, 2048, 7168)

test_shapes = [
    ("gate", 1, 8192, 7168, 2048),
    ("up",   1, 8192, 7168, 2048),
    ("down", 1, 8192, 2048, 7168),
]

print("\n首次评估（需要搜索分区）:")
for name, G, M, K, N in test_shapes:
    start = time.time()
    result = fast_eval.evaluate(G, M, K, N, input_dtype="fp8", output_dtype="bf16", use_multiprocess=True)
    elapsed = (time.time() - start) * 1000
    print(f"  {name:8s}: {elapsed:7.2f}ms, 延迟={result.latency_us:.2f}us")

print("\n二次评估（从缓存读取）:")
for name, G, M, K, N in test_shapes:
    start = time.time()
    result = fast_eval.evaluate(G, M, K, N, input_dtype="fp8", output_dtype="bf16", use_multiprocess=True)
    elapsed = (time.time() - start) * 1000
    print(f"  {name:8s}: {elapsed:7.2f}ms, 延迟={result.latency_us:.2f}us")

# 3. 打印缓存统计
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
fast_eval.print_cache_stats()

print("\n结论:")
print("  - 如果首次评估>5秒，说明分区搜索是瓶颈")
print("  - 如果二次评估<1ms，说明缓存生效")
print("  - MoE第一层需要3次首次评估 + 765次缓存读取")
print("  - 预计总时间: ~15-20秒")
