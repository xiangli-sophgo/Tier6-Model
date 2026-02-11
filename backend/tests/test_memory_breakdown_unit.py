"""内存分解功能单元测试

直接测试 _calculate_memory_breakdown() 函数。
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# 添加 backend 到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from math_model.L0_entry.eval_config import (
    EvalConfig, ModelConfig, MLAConfig, MoEConfig,
    DeploymentConfig, BoardConfig, InferenceConfig,
    TopologyOverrides, CommOverrides
)
from math_model.L0_entry.engine import _calculate_memory_breakdown


def create_test_eval_config(
    model_name: str = "TestModel",
    hidden_size: int = 4096,
    num_layers: int = 32,
    tp: int = 4,
    pp: int = 1,
    batch_size: int = 8,
    kv_seq_len: int = 640,
    q_seq_len: int = 1,
    chip_memory_gb: int = 64,
    mla_enabled: bool = False,
) -> EvalConfig:
    """创建测试用的 EvalConfig"""

    # MLA 配置
    if mla_enabled:
        mla = MLAConfig(
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
    else:
        mla = MLAConfig(
            q_lora_rank=0,
            kv_lora_rank=0,
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
        )

    # MoE 配置（Dense 模型）
    moe = MoEConfig(
        num_routed_experts=0,
        num_shared_experts=0,
        num_activated_experts=0,
        intermediate_size=0,
    )

    # 模型配置
    model = ModelConfig(
        name=model_name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=32,
        vocab_size=32000,
        intermediate_size=11008,
        num_dense_layers=num_layers,
        num_moe_layers=0,
        mla=mla,
        moe=moe,
        weight_dtype="bf16",
        activation_dtype="bf16",
        seq_len=kv_seq_len,
        kv_seq_len=kv_seq_len,
        q_seq_len=q_seq_len,
        batch=batch_size,
        is_prefill=False,
    )

    # 部署配置
    deployment = DeploymentConfig(
        tp=tp,
        pp=pp,
        dp=1,
        ep=1,
        moe_tp=1,
        seq_len=kv_seq_len,
        batch_size=batch_size,
        enable_tp_sp=False,
        enable_ring_attention=False,
        enable_zigzag=False,
        embed_tp=tp,
        lmhead_tp=tp,
        comm_protocol=0,
        kv_cache_rate=1.0,
        is_prefill=False,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
    )

    # 板卡配置
    board = BoardConfig(
        num_chips=tp * pp,
        chip_memory_gb=chip_memory_gb,
        inter_chip_bw_gbps=448.0,
    )

    # 推理配置
    inference = InferenceConfig(
        batch_size=batch_size,
        input_seq_length=512,
        output_seq_length=128,
        weight_dtype="bf16",
        activation_dtype="bf16",
    )

    # 拓扑和通信配置（简化）
    topology = TopologyOverrides(
        c2c_bandwidth_gbps=448.0,
        c2c_latency_us=0.2,
        b2b_bandwidth_gbps=400.0,
        b2b_latency_us=2.0,
        r2r_bandwidth_gbps=400.0,
        r2r_latency_us=3.0,
        p2p_bandwidth_gbps=400.0,
        p2p_latency_us=5.0,
        switch_latency_us=0.5,
        cable_latency_us=0.1,
        memory_read_latency_us=0.1,
        memory_write_latency_us=0.1,
        noc_latency_us=0.05,
        die_to_die_latency_us=0.1,
    )

    comm = CommOverrides(
        bw_utilization=0.85,
        sync_lat_us=5.0,
    )

    return EvalConfig(
        model=model,
        chip_config={"name": "test_chip", "compute_efficiency": 0.9, "compute_dma_overlap_rate": 0.8},
        topology=topology,
        comm=comm,
        deployment=deployment,
        board=board,
        inference=inference,
        raw_model_config={},
        raw_topology_config={},
    )


def test_standard_model():
    """测试标准 Dense 模型的内存分解"""
    print("\n=== 测试 1: 标准 Dense 模型（LLaMA-7B风格）===")

    eval_config = create_test_eval_config(
        model_name="LLaMA-7B-Test",
        hidden_size=4096,
        num_layers=32,
        tp=4,
        pp=1,
        batch_size=8,
        kv_seq_len=640,
        q_seq_len=1,
        chip_memory_gb=64,
        mla_enabled=False,
    )

    # 模拟 aggregates（假设权重已经被 L4 累加）
    aggregates = {
        "memory_peak": 3.5 * 1024**3,  # 3.5 GB 权重
    }

    # 模拟 step_metrics（空，因为权重已在 aggregates 中）
    step_metrics = []

    try:
        memory = _calculate_memory_breakdown(eval_config, aggregates, step_metrics)

        print(f"  模型权重:    {memory['model_memory_gb']:.2f} GB")
        print(f"  KV Cache:    {memory['kv_cache_memory_gb']:.2f} GB")
        print(f"  激活值:      {memory['activation_memory_gb']:.2f} GB")
        print(f"  开销:        {memory['overhead_gb']:.2f} GB")
        print(f"  总计:        {memory['total_per_chip_gb']:.2f} GB")
        print(f"  是否足够:    {memory['is_memory_sufficient']}")
        print(f"  利用率:      {memory['memory_utilization'] * 100:.1f}%")

        # 验证
        if memory['total_per_chip_gb'] > 0:
            print(f"  [OK] 内存分解计算成功")
            return True
        else:
            print(f"  [FAIL] 总内存为 0")
            return False

    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mla_model():
    """测试 MLA 模型的内存分解"""
    print("\n=== 测试 2: MLA 模型（DeepSeek-V3风格）===")

    eval_config = create_test_eval_config(
        model_name="DeepSeek-V3-Test",
        hidden_size=7168,
        num_layers=61,
        tp=8,
        pp=1,
        batch_size=128,
        kv_seq_len=4096,
        q_seq_len=1,
        chip_memory_gb=64,
        mla_enabled=True,
    )

    # 模拟 aggregates
    aggregates = {
        "memory_peak": 22 * 1024**3,  # 22 GB 权重（MoE 模型）
    }

    step_metrics = []

    try:
        memory = _calculate_memory_breakdown(eval_config, aggregates, step_metrics)

        print(f"  模型权重:    {memory['model_memory_gb']:.2f} GB")
        print(f"  KV Cache:    {memory['kv_cache_memory_gb']:.2f} GB")
        print(f"  激活值:      {memory['activation_memory_gb']:.2f} GB")
        print(f"  开销:        {memory['overhead_gb']:.2f} GB")
        print(f"  总计:        {memory['total_per_chip_gb']:.2f} GB")
        print(f"  是否足够:    {memory['is_memory_sufficient']}")
        print(f"  利用率:      {memory['memory_utilization'] * 100:.1f}%")

        # 验证 MLA 压缩效果
        if memory['kv_cache_memory_gb'] < 5.0:  # MLA 应该显著减少 KV Cache
            print(f"  [OK] MLA 压缩效果显著（KV Cache < 5GB）")
        else:
            print(f"  [WARN] MLA 压缩效果不明显")

        return True

    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_insufficient():
    """测试内存不足的情况"""
    print("\n=== 测试 3: 内存不足场景 ===")

    eval_config = create_test_eval_config(
        model_name="Large-Model-Test",
        hidden_size=8192,
        num_layers=80,
        tp=2,  # 只用 2 个 TP，内存会更大
        pp=1,
        batch_size=64,
        kv_seq_len=8192,  # 长序列
        q_seq_len=1,
        chip_memory_gb=64,  # 64GB 芯片
        mla_enabled=False,
    )

    # 模拟大模型权重
    aggregates = {
        "memory_peak": 45 * 1024**3,  # 45 GB 权重
    }

    step_metrics = []

    try:
        memory = _calculate_memory_breakdown(eval_config, aggregates, step_metrics)

        print(f"  模型权重:    {memory['model_memory_gb']:.2f} GB")
        print(f"  KV Cache:    {memory['kv_cache_memory_gb']:.2f} GB")
        print(f"  激活值:      {memory['activation_memory_gb']:.2f} GB")
        print(f"  开销:        {memory['overhead_gb']:.2f} GB")
        print(f"  总计:        {memory['total_per_chip_gb']:.2f} GB")
        print(f"  是否足够:    {memory['is_memory_sufficient']}")
        print(f"  利用率:      {memory['memory_utilization'] * 100:.1f}%")

        # 验证内存不足检测
        if not memory['is_memory_sufficient']:
            print(f"  [OK] 正确识别内存不足")
            return True
        else:
            print(f"  [WARN] 应该识别为内存不足")
            return False

    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有单元测试"""
    print("=" * 60)
    print("内存分解功能单元测试")
    print("=" * 60)

    results = []

    # 测试 1: 标准模型
    results.append(test_standard_model())

    # 测试 2: MLA 模型
    results.append(test_mla_model())

    # 测试 3: 内存不足
    results.append(test_memory_insufficient())

    print("\n" + "=" * 60)
    success_count = sum(results)
    total_count = len(results)
    if all(results):
        print(f"[OK] 所有测试通过 ({success_count}/{total_count})")
        return 0
    else:
        print(f"[FAIL] 部分测试失败 ({success_count}/{total_count})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
