"""
测试 GEMM 离线预调优效果
"""

import time
import logging
import sys
sys.path.insert(0, 'backend')

from llm_simulator.simulator import run_simulation

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 简单测试配置
topology = {
    "pods": [{
        "id": "pod0",
        "racks": [{
            "id": "rack0",
            "boards": [{
                "id": "board0",
                "chips": [
                    {"id": "chip0", "name": "SG2260E", "compute_tflops_fp16": 64, "memory_gb": 64, "memory_bandwidth_gbps": 273}
                ]
            }]
        }]
    }],
    "connections": []
}

model = {
    "model_name": "DeepSeek-V3",
    "hidden_size": 2048,
    "num_layers": 10,
    "num_attention_heads": 16,
    "num_kv_heads": 16,
    "intermediate_size": 8192,
}

inference = {
    "batch_size": 1,
    "input_seq_length": 128,
    "output_seq_length": 100,
}

parallelism = {
    "dp": 1,
    "tp": 1,
    "pp": 1,
    "ep": 1,
}

hardware = {
    "chip": {
        "chip_type": "SG2260E",
        "compute_tflops_fp16": 64,
        "memory_gb": 64,
        "memory_bandwidth_gbps": 273,
    }
}

print("=" * 80)
print("🧪 测试 GEMM 离线预调优")
print("=" * 80)

# 测试1: 启用预热
print("\n【测试1】启用 GEMM 预热")
config_with_prewarm = {
    "maxSimulatedTokens": 4,
    "enableDataTransferSimulation": True,
}

start = time.time()
result1 = run_simulation(
    topology_dict=topology,
    model_dict=model,
    inference_dict=inference,
    parallelism_dict=parallelism,
    hardware_dict=hardware,
    config_dict=config_with_prewarm,
)
time_with_prewarm = time.time() - start

print(f"✅ 耗时: {time_with_prewarm*1000:.2f}ms")

# 测试2: 禁用预热（重启模拟器，清空缓存）
print("\n【测试2】禁用 GEMM 预热")
# 注意：这里需要通过环境变量或配置禁用预热，暂时跳过

print("=" * 80)
print(f"📊 测试完成")
print(f"   启用预热: {time_with_prewarm*1000:.2f}ms")
print("=" * 80)
