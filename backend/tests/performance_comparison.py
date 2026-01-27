#!/usr/bin/env python3
"""
性能对比测试脚本

对比 Tier6+Model 和 DS_TPU 的评估性能
使用相同的模型配置、部署配置和硬件配置
"""

import sys
import time
import json
from pathlib import Path

# 添加路径
tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def load_ds_tpu_config():
    """加载 DS_TPU 的模型配置"""
    from config.config_loader import load_model_config

    model_config = load_model_config("deepseek-v3.2")
    return model_config


def create_tier6_config(ds_model_config, deployment_config):
    """将 DS_TPU 配置转换为 Tier6+Model 格式"""

    # 模型配置
    model_dict = {
        "model_name": ds_model_config.get("name", "DeepSeek-V3.2"),
        "model_type": "moe",  # DS-V3.2 是 MoE 模型
        "hidden_size": ds_model_config["hidden_dim"],
        "num_layers": ds_model_config["n_layers"],
        "num_attention_heads": ds_model_config["n_heads"],
        "num_kv_heads": ds_model_config.get("n_kv_heads", ds_model_config["n_heads"]),
        "intermediate_size": ds_model_config["inter_dim"],
        "vocab_size": ds_model_config.get("vocab_size", 32000),
        "dtype": "bf16",
        "max_seq_length": 8192,
        "attention_type": "mla",
        "mla_config": {
            "qk_nope_head_dim": ds_model_config["qk_nope_head_dim"],
            "qk_rope_head_dim": ds_model_config["qk_rope_head_dim"],
            "v_head_dim": ds_model_config["v_head_dim"],
            "kv_lora_rank": ds_model_config["kv_lora_rank"],
            "q_lora_rank": ds_model_config["q_lora_rank"],
            "variant": "mla_absorb_v32",  # DeepSeek V3.2 使用 absorb 优化
        },
        "moe_config": {
            "num_experts": ds_model_config["n_routed_experts"],
            "num_experts_per_tok": ds_model_config["n_activated_experts"],
            "num_shared_experts": ds_model_config.get("n_shared_experts", 0),
            "expert_intermediate_size": ds_model_config["moe_inter_dim"],
            "first_k_dense_replace": ds_model_config.get("n_dense_layers", 0),
        }
    }

    # 推理配置
    # 注意：Tier6 会先 Prefill (input_seq_length 个 token)，再 Decode (output_seq_length 个 token)
    # 为了对齐 DS_TPU 的 Decode 模式 (context=8192)，设置：
    inference_dict = {
        "batch_size": deployment_config["batch_size"],
        "input_seq_length": deployment_config["kv_len"],  # Prefill 处理 8192 个 token（建立 KV cache）
        "output_seq_length": 1,  # Decode 生成 1 个 token
        "max_seq_length": deployment_config["kv_len"],
    }

    # 并行策略
    parallelism_dict = {
        "dp": deployment_config["dp"],
        "tp": deployment_config["tp"],
        "pp": 1,  # DS_TPU 默认不用 PP
        "ep": deployment_config["ep"],
        "sp": 1,
    }

    # 硬件配置（使用 SG2260E 参数）
    hardware_dict = {
        "chip": {
            "chip_type": "SG2260E",
            "compute_tflops_fp16": 64,
            "memory_gb": 64,
            "memory_bandwidth_gbps": 273,
            "num_cores": deployment_config["tpu_cores"],
        },
        "node": {
            "chips_per_node": 8,
            "intra_node_bandwidth_gbps": 64,
            "intra_node_latency_us": 0.35,
        },
        "cluster": {
            "num_nodes": 1,
            "inter_node_bandwidth_gbps": 16,
            "inter_node_latency_us": 2,
        }
    }

    # 拓扑配置（简单的单节点拓扑）
    total_chips = deployment_config["dp"] * deployment_config["tp"] * deployment_config["ep"]
    topology_dict = {
        "pods": [
            {
                "id": "pod_0",
                "racks": [
                    {
                        "id": "rack_0",
                        "boards": [
                            {
                                "id": f"board_{i}",
                                "chips": [
                                    {
                                        "id": f"chip_{i * 8 + j}",
                                        "name": "SG2260E",
                                    }
                                    for j in range(min(8, total_chips - i * 8))
                                ]
                            }
                            for i in range((total_chips + 7) // 8)
                        ]
                    }
                ]
            }
        ],
        "connections": []
    }

    return topology_dict, model_dict, inference_dict, parallelism_dict, hardware_dict


def run_ds_tpu_benchmark(model_config, deployment_config, tpu_kwargs):
    """运行 DS_TPU 评估并计时"""
    print("\n" + "="*80)
    print("🚀 DS_TPU 评估开始")
    print("="*80)

    from top.simulator import TPUSimulator
    from config.deployment_config import DeploymentConfig

    # 创建部署配置对象
    deploy_cfg = DeploymentConfig(
        batch_size=deployment_config["batch_size"],
        q_seq_len=deployment_config["q_len"],
        kv_seq_len=deployment_config["kv_len"],
        tp=deployment_config["tp"],
        dp=deployment_config["dp"],
        moe_tp=deployment_config["moe_tp"],
        ep=deployment_config["ep"],
        is_prefill=deployment_config["is_prefill"],
        enable_tp_sp=deployment_config["enable_tp_sp"],
        comm_protocol=deployment_config["comm_protocol"],
    )

    # 计时开始
    start_time = time.time()

    # 运行模拟
    global_cache = {}
    simulator = TPUSimulator()
    results = simulator.run_simulation(
        model_cfg=model_config,
        tpu_kwargs=tpu_kwargs,
        deploy_cfg=deploy_cfg,
        model_version="v3.2",
        global_cache=global_cache
    )

    # 计时结束
    elapsed = time.time() - start_time

    print(f"\n✅ DS_TPU 评估完成")
    print(f"⏱️  总耗时: {elapsed:.3f}s")

    perf = results.get("performance", {})
    print(f"📊 性能指标:")
    print(f"   - 执行时间: {perf.get('total_elapse_us', 0):.2f} μs")
    print(f"   - 吞吐量: {perf.get('tps', 0):.2f} tokens/s")
    print(f"   - MFU: {perf.get('mfu', 0)*100:.2f}%")

    return elapsed, results


def run_tier6_benchmark(topology_dict, model_dict, inference_dict, parallelism_dict, hardware_dict):
    """运行 Tier6+Model 评估并计时"""
    print("\n" + "="*80)
    print("🚀 Tier6+Model 评估开始")
    print("="*80)

    from llm_simulator.core.simulator import run_simulation

    # 计时开始
    start_time = time.time()

    # 运行模拟（参数对齐 DS_TPU）
    results = run_simulation(
        topology_dict=topology_dict,
        model_dict=model_dict,
        inference_dict=inference_dict,
        parallelism_dict=parallelism_dict,
        hardware_dict=hardware_dict,
        enable_tile_search=True,  # ✅ 对齐 DS_TPU：开启 tile 搜索
        enable_partition_search=True,  # ✅ 对齐 DS_TPU：开启分区搜索
        max_simulated_tokens=1,  # ✅ 对齐 DS_TPU：只模拟1个 decode token
    )

    # 计时结束
    elapsed = time.time() - start_time

    print(f"\n✅ Tier6+Model 评估完成")
    print(f"⏱️  总耗时: {elapsed:.3f}s")

    stats = results.get("stats", {})
    print(f"📊 性能指标:")
    print(f"   - TTFT: {stats.get('ttft', 0):.2f} ms")
    print(f"   - Avg TPOT: {stats.get('avgTpot', 0):.2f} ms")
    print(f"   - MFU: {stats.get('dynamicMfu', 0)*100:.2f}%")

    return elapsed, results


def main():
    """主函数"""
    print("="*80)
    print("🔬 Tier6+Model vs DS_TPU 性能对比测试")
    print("="*80)

    # 配置参数（对齐DS_TPU的默认配置）
    deployment_config = {
        "batch_size": 48 * 32,
        "q_len": 1,  # Decode 阶段
        "kv_len": 8192,
        "tp": 1,
        "dp": 32,
        "moe_tp": 1,
        "ep": 32,
        "is_prefill": False,
        "enable_tp_sp": True,
        "comm_protocol": 1,
        "tpu_cores": 64,
    }

    tpu_kwargs = {"core": deployment_config["tpu_cores"]}

    print("\n📋 配置参数:")
    print(f"   - Batch Size: {deployment_config['batch_size']}")
    print(f"   - Seq Len: {deployment_config['q_len']} (q) / {deployment_config['kv_len']} (kv)")
    print(f"   - 并行度: TP={deployment_config['tp']}, DP={deployment_config['dp']}, EP={deployment_config['ep']}")
    print(f"   - TPU Cores: {deployment_config['tpu_cores']}")
    print(f"   - Prefill: {deployment_config['is_prefill']}")

    # 加载 DS_TPU 配置
    print("\n📥 加载 DS_TPU 模型配置...")
    ds_model_config = load_ds_tpu_config()

    # 转换为 Tier6 配置
    print("🔄 转换为 Tier6+Model 配置格式...")
    topology_dict, model_dict, inference_dict, parallelism_dict, hardware_dict = create_tier6_config(
        ds_model_config, deployment_config
    )

    # 运行 DS_TPU 基准测试
    try:
        ds_time, ds_results = run_ds_tpu_benchmark(ds_model_config, deployment_config, tpu_kwargs)
    except Exception as e:
        print(f"\n❌ DS_TPU 评估失败: {e}")
        import traceback
        traceback.print_exc()
        ds_time = None
        ds_results = None

    # 运行 Tier6+Model 基准测试
    try:
        tier6_time, tier6_results = run_tier6_benchmark(
            topology_dict, model_dict, inference_dict, parallelism_dict, hardware_dict
        )
    except Exception as e:
        print(f"\n❌ Tier6+Model 评估失败: {e}")
        import traceback
        traceback.print_exc()
        tier6_time = None
        tier6_results = None

    # 对比结果
    print("\n" + "="*80)
    print("📊 性能对比总结")
    print("="*80)

    if ds_time and tier6_time:
        print(f"\n⏱️  耗时对比:")
        print(f"   DS_TPU:        {ds_time:.3f}s")
        print(f"   Tier6+Model:   {tier6_time:.3f}s")
        print(f"   差距:          {tier6_time - ds_time:.3f}s ({tier6_time/ds_time:.2f}x)")

        if tier6_time > ds_time:
            print(f"\n⚠️  Tier6+Model 比 DS_TPU 慢 {(tier6_time/ds_time - 1)*100:.1f}%")
        else:
            print(f"\n✅ Tier6+Model 比 DS_TPU 快 {(1 - tier6_time/ds_time)*100:.1f}%")

    # 保存详细结果
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)

    if ds_results:
        ds_output = output_dir / "ds_tpu_result.json"
        with open(ds_output, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"\n💾 DS_TPU 结果已保存: {ds_output}")

    if tier6_results:
        tier6_output = output_dir / "tier6_result.json"
        with open(tier6_output, "w") as f:
            json.dump(tier6_results, f, indent=2)
        print(f"💾 Tier6+Model 结果已保存: {tier6_output}")


if __name__ == "__main__":
    main()
