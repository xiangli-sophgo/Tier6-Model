"""内存分析集成测试

验证内存分解功能在完整评估流程中的工作情况。
"""

import sys
import json
from pathlib import Path

# 添加 backend 到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def test_memory_breakdown_integration():
    """集成测试：验证内存分解在评估请求中正常工作"""
    print("\n=== 集成测试: 内存分解功能 ===")

    from math_model.L0_entry.engine import run_evaluation_from_request

    # 构造一个简化的测试配置（LLaMA-7B）
    config = {
        "experiment_name": "test_memory_integration",
        "benchmark_name": "LLaMA-7B-Test",
        "topology_config_name": "P1-R1-B1-C8",
        "benchmark_config": {
            "model": {
                "name": "LLaMA-7B",
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "intermediate_size": 11008,
                "num_dense_layers": 32,
                "num_moe_layers": 0,
                "MLA": {
                    "q_lora_rank": 0,
                    "kv_lora_rank": 0,
                    "qk_nope_head_dim": 0,
                    "qk_rope_head_dim": 0,
                    "v_head_dim": 0,
                },
                "MoE": {
                    "num_routed_experts": 0,
                    "num_shared_experts": 0,
                    "num_activated_experts": 0,
                    "intermediate_size": 0,
                },
            },
            "inference": {
                "batch_size": 8,
                "input_seq_length": 512,
                "output_seq_length": 128,
                "weight_dtype": "bf16",
                "activation_dtype": "bf16",
            },
        },
        "topology_config": {
            "name": "P1-R1-B1-C8",
            "pod_count": 1,
            "racks_per_pod": 1,
            "rack_config": {
                "boards": [
                    {
                        "chips": [
                            {
                                "name": "SG2262",
                                "count": 8,
                            }
                        ],
                        "count": 1,
                    }
                ]
            },
            "chips": {
                "SG2262": {
                    "name": "SG2262",
                    "num_cores": 64,
                    "compute_tflops_fp8": 768,
                    "compute_tflops_bf16": 384,
                    "compute_tflops_fp32": 192,
                    "memory_capacity_gb": 64,
                    "memory_bandwidth_gbps": 12000,
                    "memory_bandwidth_utilization": 0.85,
                    "lmem_capacity_mb": 64,
                    "lmem_bandwidth_gbps": 6400,
                    "cube_m": 16,
                    "cube_k": 32,
                    "cube_n": 8,
                    "sram_size_kb": 2048,
                    "sram_utilization": 0.45,
                    "lane_num": 16,
                    "align_bytes": 32,
                    "compute_dma_overlap_rate": 0.8,
                    "compute_efficiency": 0.9,
                },
            },
            "interconnect": {
                "links": {
                    "c2c": {"bandwidth_gbps": 448, "latency_us": 0.2},
                    "b2b": {"bandwidth_gbps": 400, "latency_us": 2.0},
                    "r2r": {"bandwidth_gbps": 400, "latency_us": 3.0},
                    "p2p": {"bandwidth_gbps": 400, "latency_us": 5.0},
                },
                "comm_params": {
                    "bandwidth_utilization": 0.85,
                    "sync_latency_us": 5.0,
                    "switch_latency_us": 0.5,
                    "cable_latency_us": 0.1,
                    "memory_read_latency_us": 0.1,
                    "memory_write_latency_us": 0.1,
                    "noc_latency_us": 0.05,
                    "die_to_die_latency_us": 0.1,
                },
            },
        },
        "manual_parallelism": {
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "ep": 1,
            "moe_tp": 1,
            "seq_len": 640,
            "enable_tp_sp": False,
            "enable_ring_attention": False,
            "enable_zigzag": False,
            "embed_tp": 4,
            "lmhead_tp": 4,
            "comm_protocol": 0,
            "kv_cache_rate": 1.0,
            "is_prefill": False,
        },
        "search_mode": "manual",
    }

    try:
        # 运行评估
        print("  运行评估请求...")
        result = run_evaluation_from_request(config)

        # 检查返回结果
        if not result or "top_k_plans" not in result:
            print(f"  [FAIL] 评估返回结果格式错误")
            return False

        plans = result["top_k_plans"]
        if not plans:
            print(f"  [FAIL] 没有可行的评估方案")
            if "infeasible_plans" in result and result["infeasible_plans"]:
                reason = result["infeasible_plans"][0].get("infeasible_reason", "Unknown")
                print(f"  原因: {reason}")
            return False

        plan = plans[0]

        # 验证 memory 字段存在
        if "memory" not in plan:
            print(f"  [FAIL] 评估结果中缺少 'memory' 字段")
            return False

        memory = plan["memory"]

        # 验证 memory 格式
        required_fields = [
            "model_memory_gb",
            "kv_cache_memory_gb",
            "activation_memory_gb",
            "overhead_gb",
            "total_per_chip_gb",
            "is_memory_sufficient",
            "memory_utilization",
        ]

        for field in required_fields:
            if field not in memory:
                print(f"  [FAIL] 内存分解缺少字段: {field}")
                return False

        # 打印内存分解结果
        print(f"\n  内存分解结果:")
        print(f"    模型权重:    {memory['model_memory_gb']:.2f} GB")
        print(f"    KV Cache:    {memory['kv_cache_memory_gb']:.2f} GB")
        print(f"    激活值:      {memory['activation_memory_gb']:.2f} GB")
        print(f"    开销:        {memory['overhead_gb']:.2f} GB")
        print(f"    总计:        {memory['total_per_chip_gb']:.2f} GB")
        print(f"    是否足够:    {memory['is_memory_sufficient']}")
        print(f"    利用率:      {memory['memory_utilization'] * 100:.1f}%")

        # 验证数值合理性
        total = memory["total_per_chip_gb"]
        if total <= 0:
            print(f"  [FAIL] 总内存为 0")
            return False

        # 验证总计等于各部分之和
        calculated_total = (
            memory["model_memory_gb"]
            + memory["kv_cache_memory_gb"]
            + memory["activation_memory_gb"]
            + memory["overhead_gb"]
        )

        if abs(calculated_total - total) > 0.01:
            print(f"  [WARN] 总计不等于各部分之和: {calculated_total:.2f} vs {total:.2f}")

        # 验证利用率计算
        chip_memory_gb = 64  # SG2262
        expected_utilization = total / chip_memory_gb
        if abs(memory["memory_utilization"] - expected_utilization) > 0.01:
            print(f"  [WARN] 利用率计算错误: {memory['memory_utilization']:.3f} vs {expected_utilization:.3f}")

        # 验证内存充足性
        expected_sufficient = total <= chip_memory_gb
        if memory["is_memory_sufficient"] != expected_sufficient:
            print(f"  [FAIL] 内存充足性判断错误")
            return False

        print(f"\n  [OK] 内存分解功能集成测试通过")
        return True

    except Exception as e:
        print(f"  [FAIL] 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行集成测试"""
    print("=" * 60)
    print("内存分析集成测试")
    print("=" * 60)

    success = test_memory_breakdown_integration()

    print("\n" + "=" * 60)
    if success:
        print("[OK] 集成测试通过")
    else:
        print("[FAIL] 集成测试失败")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
