"""
性能测试脚本

用于快速测试模拟器的性能，找出瓶颈
"""

import time
import logging
from simulator import run_simulation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 简单的测试配置
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
    "model_name": "test",
    "hidden_size": 2048,
    "num_layers": 10,  # 只测试10层
    "num_attention_heads": 16,
    "intermediate_size": 8192,
}

inference = {
    "batch_size": 1,
    "input_seq_length": 128,
    "output_seq_length": 100,  # 但只会模拟4个token
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

config = {
    "maxSimulatedTokens": 4,
    "enableDataTransferSimulation": True,
}

print("=" * 80)
print("🚀 开始性能测试...")
print("=" * 80)

start = time.time()
result = run_simulation(
    topology_dict=topology,
    model_dict=model,
    inference_dict=inference,
    parallelism_dict=parallelism,
    hardware_dict=hardware,
    config_dict=config,
)
elapsed = time.time() - start

print("=" * 80)
print(f"✅ 测试完成，总耗时: {elapsed*1000:.2f}ms")
print("=" * 80)
