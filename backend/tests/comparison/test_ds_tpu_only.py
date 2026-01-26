#!/usr/bin/env python3
"""
测试 DS_TPU 评估器 (单独运行，验证调用正确)
"""

import sys
from pathlib import Path

# 添加 DS_TPU 路径
DS_TPU_PATH = Path(r"c:\Users\DELL\Documents\code\DS_TPU_1209")
sys.path.insert(0, str(DS_TPU_PATH))


def test_ds_tpu_deepseek():
    """测试 DS_TPU DeepSeek 模型评估"""
    from config.config_loader import load_model_config
    from config.deployment_config import DeploymentConfig
    from top.simulator import TPUSimulator

    print("=" * 60)
    print("DS_TPU DeepSeek 模型评估测试")
    print("=" * 60)

    # 加载模型配置
    print("\n[1] 加载 DeepSeek V3.2 模型配置...")
    model_config = load_model_config('deepseek-v3.2')

    # 修改为单层测试 (保持一个 MoE 层)
    model_config['n_layers'] = 4  # 3 dense + 1 moe
    model_config['n_moe_layers'] = 1
    model_config['n_dense_layers'] = 3

    print(f"  模型名称: {model_config.get('name')}")
    print(f"  层数: {model_config.get('n_layers')}")

    # 创建部署配置 (对齐 Tier6 测试)
    # batch_size 需要足够大使 local_batch = batch_size / dp >= 1
    print("\n[2] 创建部署配置...")
    deploy_config = DeploymentConfig(
        batch_size=64,  # 增大 batch 确保 local_batch = 64/32 = 2
        q_seq_len=1,  # Decode 模式
        kv_seq_len=4096,
        tp=1,
        dp=32,
        moe_tp=1,
        ep=32,
        is_prefill=False,
        enable_tp_sp=True,
        comm_protocol=1,
    )
    print(f"  batch_size: {deploy_config.batch_size}")
    print(f"  q_seq_len: {deploy_config.q_seq_len}")
    print(f"  tp: {deploy_config.tp}, ep: {deploy_config.ep}")

    # TPU 参数
    tpu_kwargs = {'core': 64}

    # 运行模拟
    print("\n[3] 运行性能模拟...")
    simulator = TPUSimulator(verbose=False)
    results = simulator.run_simulation(
        model_cfg=model_config,
        deploy_cfg=deploy_config,
        tpu_kwargs=tpu_kwargs,
        model_version='v3.2',
        global_cache={},
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
        print(f"  {layer_name}:")
        print(f"    延迟: {layer_perf.get('elapse', 0):.2f} μs")
        print(f"    计算: {layer_perf.get('comp_elapse', 0):.2f} μs")
        print(f"    通信: {layer_perf.get('comm_elapse', 0):.2f} μs")

        # 打印算子详情
        comp_ops = layer_info.get('comp_operators', [])
        if comp_ops:
            print(f"    计算算子 ({len(comp_ops)} 个):")
            for op in comp_ops[:5]:  # 只显示前 5 个
                print(f"      - {op.get('name', 'unknown')}: {op.get('elapsed', 0):.2f} μs")
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
    test_ds_tpu_deepseek()
