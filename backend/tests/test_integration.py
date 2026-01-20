#!/usr/bin/env python3
"""
整合验证测试

测试新评估器系统与 simulator 的整合是否成功
"""

import sys
from pathlib import Path

# 添加路径
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from llm_simulator.simulator import run_simulation


def test_basic_simulation():
    """测试基本模拟功能"""
    print("=" * 80)
    print("测试 1: 基本模拟功能")
    print("=" * 80)

    # 简单的拓扑配置 - 正确格式：pods -> racks -> boards -> chips
    topology = {
        "pods": [
            {
                "id": "pod_0",
                "label": "Pod 0",
                "grid_size": [1, 1],
                "racks": [
                    {
                        "id": "rack_0",
                        "label": "Rack 0",
                        "position": [0, 0],
                        "boards": [
                            {
                                "id": "board_0",
                                "label": "Board 0",
                                "u_position": 0,
                                "u_height": 1,
                                "chips": [
                                    {
                                        "id": f"chip_{i}",
                                        "type": "chip",
                                        "position": [i % 4, i // 4],
                                        "label": f"Chip {i}"
                                    } for i in range(8)
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
        "connections": []  # 可选：手动定义连接
    }

    # 模型配置 - 小模型快速测试
    model = {
        "model_name": "test-model",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "dtype": "fp16",
    }

    # 推理配置
    inference = {
        "batch_size": 1,
        "input_seq_length": 128,
        "output_seq_length": 128,
    }

    # 并行策略
    parallelism = {
        "dp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "sp": 1,
    }

    # 硬件配置 - 使用 SG2260E
    hardware = {
        "chip": {
            "chip_type": "SG2260E",
            "compute_tflops_fp16": 64,
            "memory_gb": 64,
            "memory_bandwidth_gbps": 273,
        },
        "node": {
            "chips_per_node": 8,
            "intra_node_bandwidth_gbps": 64,
            "intra_node_latency_us": 1,
        },
        "cluster": {
            "num_nodes": 1,
            "inter_node_bandwidth_gbps": 16,
            "inter_node_latency_us": 2,
        },
    }

    # 模拟配置 - 使用新评估器
    sim_config = {
        "maxSimulatedTokens": 4,  # 少量 token 快速测试
        "enableDataTransferSimulation": True,
        "enableDetailedTransformerOps": True,
        "enableKVCacheAccessSimulation": True,
    }

    try:
        print("\n运行模拟...")
        result = run_simulation(
            topology_dict=topology,
            model_dict=model,
            inference_dict=inference,
            parallelism_dict=parallelism,
            hardware_dict=hardware,
            config_dict=sim_config,
        )

        print("\n✓ 模拟成功完成!")
        print(f"\n统计信息:")
        print(f"  - TTFT: {result['stats']['ttft']:.2f} ms")
        print(f"  - 平均 TPOT: {result['stats']['avgTpot']:.2f} ms")
        print(f"  - MFU: {result['stats']['dynamicMfu']:.2%}")
        print(f"  - MBU: {result['stats']['dynamicMbu']:.2%}")
        print(f"  - 总事件数: {result['stats']['totalEvents']}")

        return True

    except Exception as e:
        print(f"\n✗ 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mla_simulation():
    """测试 MLA 模型模拟"""
    print("\n" + "=" * 80)
    print("测试 2: MLA 模型模拟")
    print("=" * 80)

    topology = {
        "pods": [
            {
                "id": "pod_0",
                "label": "Pod 0",
                "grid_size": [1, 1],
                "racks": [
                    {
                        "id": "rack_0",
                        "label": "Rack 0",
                        "position": [0, 0],
                        "boards": [
                            {
                                "id": "board_0",
                                "label": "Board 0",
                                "u_position": 0,
                                "u_height": 1,
                                "chips": [
                                    {
                                        "id": f"chip_{i}",
                                        "type": "chip",
                                        "position": [i % 4, i // 4],
                                        "label": f"Chip {i}"
                                    } for i in range(8)
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
        "connections": []
    }

    # DeepSeek-V3 风格 MLA 配置
    model = {
        "model_name": "test-mla",
        "hidden_size": 7168,
        "num_layers": 12,  # 减少层数快速测试
        "num_attention_heads": 128,
        "intermediate_size": 18432,
        "vocab_size": 151936,
        "dtype": "bf16",
        "attention_type": "mla",
        "mla_config": {
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        },
    }

    inference = {
        "batch_size": 1,
        "input_seq_length": 64,
        "output_seq_length": 64,
    }

    parallelism = {
        "dp": 1,
        "tp": 2,  # 使用 TP=2
        "pp": 1,
        "ep": 1,
        "sp": 1,
    }

    hardware = {
        "chip": {
            "chip_type": "SG2260E",
            "compute_tflops_fp16": 64,
            "memory_gb": 64,
            "memory_bandwidth_gbps": 273,
        },
        "node": {
            "chips_per_node": 8,
            "intra_node_bandwidth_gbps": 64,
            "intra_node_latency_us": 1,
        },
        "cluster": {
            "num_nodes": 1,
            "inter_node_bandwidth_gbps": 16,
            "inter_node_latency_us": 2,
        },
    }

    sim_config = {
        "maxSimulatedTokens": 2,
        "enableDataTransferSimulation": True,
        "enableDetailedTransformerOps": True,
        "enableKVCacheAccessSimulation": True,
    }

    try:
        print("\n运行 MLA 模拟...")
        result = run_simulation(
            topology_dict=topology,
            model_dict=model,
            inference_dict=inference,
            parallelism_dict=parallelism,
            hardware_dict=hardware,
            config_dict=sim_config,
        )

        print("\n✓ MLA 模拟成功完成!")
        print(f"\n统计信息:")
        print(f"  - TTFT: {result['stats']['ttft']:.2f} ms")
        print(f"  - 平均 TPOT: {result['stats']['avgTpot']:.2f} ms")
        print(f"  - MFU: {result['stats']['dynamicMfu']:.2%}")

        return True

    except Exception as e:
        print(f"\n✗ MLA 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 80)
    print("Tier6+ Simulator 整合验证测试")
    print("=" * 80)

    results = []

    # 测试 1: 基本模拟
    results.append(("基本模拟", test_basic_simulation()))

    # 测试 2: MLA 模拟
    results.append(("MLA 模拟", test_mla_simulation()))

    # 总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！整合成功！")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
