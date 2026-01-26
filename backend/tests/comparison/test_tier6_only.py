#!/usr/bin/env python3
"""
测试 Tier6 评估器 (单独运行，验证调用正确)
"""

import sys
from pathlib import Path

# 添加 Tier6 路径
TIER6_PATH = Path(r"C:\Users\DELL\Documents\code\Tier6-Model\backend")
sys.path.insert(0, str(TIER6_PATH))


def test_tier6_deepseek():
    """测试 Tier6 DeepSeek 模型评估"""
    from llm_simulator.evaluators import get_arch_preset
    from llm_simulator.analyzer import PerformanceAnalyzer
    from llm_simulator.models.deepseek import DeepSeekModel
    from llm_simulator.types import ProtocolConfig, NetworkInfraConfig

    print("=" * 60)
    print("Tier6 DeepSeek 模型评估测试")
    print("=" * 60)

    # 配置参数 (对齐 DS_TPU 默认配置)
    # batch_size 语义已对齐: 现在是全局 batch (与 DS_TPU 一致)
    model_config = {
        # 模型结构
        'hidden_dim': 7168,
        'inter_dim': 18432,
        'vocab_size': 129280,
        'n_layers': 4,  # 3 dense + 1 moe (对齐 DS_TPU)
        'n_dense_layers': 3,
        'n_moe_layers': 1,
        'num_heads': 128,

        # MLA 参数
        'qk_nope_dim': 128,
        'qk_rope_dim': 64,
        'v_head_dim': 128,
        'kv_lora_rank': 512,
        'q_lora_rank': 1536,
        'mla_type': 'mla_absorb',
        'enable_tp_sp': True,

        # MoE 参数
        'num_experts': 256,
        'num_activated_experts': 8,
        'num_shared_experts': 1,
        'expert_inter_dim': 2048,

        # 部署参数 (batch_size 现在是全局 batch，与 DS_TPU 对齐)
        'batch_size': 64,  # 全局 batch (与 DS_TPU 一致)
        'seq_len': 1,  # Decode 模式
        'kv_seq_len': 4096,
        'tp': 1,
        'dp': 32,  # 数据并行度 (内部计算 local_batch = 64 // 32 = 2)
        'moe_tp': 1,
        'ep': 32,
        'comm_protocol': 1,
        'is_prefill': False,
    }

    print(f"\n配置: batch={model_config['batch_size']} (全局), "
          f"dp={model_config['dp']}, "
          f"local_batch={model_config['batch_size'] // model_config['dp']}, "
          f"seq_len={model_config['seq_len']}, "
          f"tp={model_config['tp']}, "
          f"ep={model_config['ep']}")

    # 创建模型
    print("\n[1] 创建 DeepSeek 模型...")
    model = DeepSeekModel(name="deepseek-v3-test", config=model_config)
    print(f"  层数: {len(model.layers)}")
    print(f"  算子类型: {list(model.operator_types)}")

    # 获取架构
    print("\n[2] 获取 SG2260E 架构...")
    arch = get_arch_preset('SG2260E')
    print(f"  核心数: {arch.num_cores}")
    print(f"  DRAM 带宽: {arch.dram_bandwidth_bytes / 1e9:.1f} GB/s")

    # 创建协议配置
    protocol_config = ProtocolConfig(
        rtt_tp_us=0.35,
        rtt_ep_us=0.85,
        bandwidth_utilization=0.95,
    )

    network_config = NetworkInfraConfig(
        switch_delay_us=1.0,
        cable_delay_us=0.025,
    )

    # 运行分析
    print("\n[3] 运行性能分析...")
    analyzer = PerformanceAnalyzer(
        model=model,
        arch=arch,
        global_cache={},
        protocol_config=protocol_config,
        network_config=network_config,
    )

    # 获取结果
    results = analyzer.get_summary(
        batch_size=model_config['batch_size'],
        seq_len=model_config['seq_len'],
    )

    # 打印结果
    perf = results.get('performance', {})
    print("\n" + "=" * 60)
    print("性能结果")
    print("=" * 60)
    print(f"  总延迟: {perf.get('total_elapse_us', 0):.2f} μs")
    print(f"  通信延迟: {perf.get('comm_elapse_us', 0):.2f} μs")
    print(f"  MFU: {perf.get('mfu', 0):.4f}" if perf.get('mfu') else "  MFU: N/A")
    print(f"  TPS: {perf.get('tps', 0):.2f}")

    # 打印层级详情
    layers = perf.get('layers', {})
    print("\n层级详情:")
    print("-" * 60)
    for layer_name, layer_info in layers.items():
        layer_perf = layer_info.get('perf', {})
        count = layer_info.get('count', 1)
        print(f"  {layer_name} (x{count}):")
        print(f"    延迟: {layer_perf.get('elapse', 0):.2f} μs")
        print(f"    计算: {layer_perf.get('comp_elapse', 0):.2f} μs")
        print(f"    通信: {layer_perf.get('comm_elapse', 0):.2f} μs")

        # 打印算子详情
        comp_ops = layer_info.get('comp_operators', [])
        if comp_ops:
            print(f"    计算算子 ({len(comp_ops)} 个):")
            for op in comp_ops[:5]:  # 只显示前 5 个
                print(f"      - {op.get('name', 'unknown')}: {op.get('elapse', 0):.2f} μs")
            if len(comp_ops) > 5:
                print(f"      ... 还有 {len(comp_ops) - 5} 个")

        comm_ops = layer_info.get('comm_operators', [])
        if comm_ops:
            print(f"    通信算子 ({len(comm_ops)} 个):")
            for op in comm_ops:
                print(f"      - {op.get('name', 'unknown')}: {op.get('comm_elapse', 0):.2f} μs")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return results


if __name__ == '__main__':
    test_tier6_deepseek()
