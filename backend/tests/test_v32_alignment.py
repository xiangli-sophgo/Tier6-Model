"""
DeepSeek V3.2 Tier6 vs DS_TPU 延迟对齐测试

对比配置 (DS_TPU 默认):
- batch_size: 1536 (全局)
- dp: 32, ep: 32, tp: 1
- seq_len: 1 (decode)
- hidden_dim: 7168
- num_experts: 256, activated: 8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_simulator.models.deepseek import DeepSeekModel
from llm_simulator.evaluators import get_arch_preset
from llm_simulator.analyzer import PerformanceAnalyzer
from llm_simulator.evaluators import get_max_expert_load_for_moe_layer
import math


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_v32_decode_alignment():
    """测试 V3.2 Decode 阶段的延迟对齐"""

    print_section("DeepSeek V3.2 Decode 延迟对齐测试")

    # DS_TPU 默认配置
    config = {
        # 模型参数
        'hidden_dim': 7168,
        'n_layers': 61,
        'n_dense_layers': 3,
        'num_heads': 128,
        'head_dim': 128,

        # MLA 参数
        'q_lora_rank': 1536,
        'kv_lora_rank': 512,
        'qk_nope_head_dim': 128,
        'qk_rope_head_dim': 64,
        'v_head_dim': 128,

        # MoE 参数
        'num_experts': 256,
        'num_activated_experts': 8,
        'num_shared_experts': 1,
        'expert_inter_dim': 2048,
        'dense_inter_dim': 18432,

        # 部署参数 (DS_TPU 默认)
        'batch_size': 1536,  # 全局 batch
        'seq_len': 1,        # decode
        'kv_seq_len': 4096,
        'tp': 1,
        'dp': 32,
        'moe_tp': 1,
        'ep': 32,
        'comm_protocol': 1,
        'is_prefill': False,
        'enable_dsa': True,  # V3.2 特有的 DSA 层
    }

    print("\n[配置信息]")
    print(f"  全局 batch: {config['batch_size']}")
    print(f"  本地 batch: {config['batch_size'] // config['dp']}")
    print(f"  DP: {config['dp']}, EP: {config['ep']}, TP: {config['tp']}")
    print(f"  seq_len: {config['seq_len']} (decode)")
    print(f"  hidden_dim: {config['hidden_dim']}")
    print(f"  num_experts: {config['num_experts']}, activated: {config['num_activated_experts']}")

    # 计算 MoE 参数
    global_batch = config['batch_size']
    seq_len = config['seq_len']
    ep = config['ep']
    num_activated = config['num_activated_experts']
    num_experts = config['num_experts']

    token_per_ep_group = math.ceil(global_batch * seq_len * num_activated / ep)
    expert_per_ep_group = math.ceil(num_experts / ep)
    m_per_group = math.ceil(token_per_ep_group / expert_per_ep_group)

    # 查表获取最忙芯片的专家数
    max_experts = get_max_expert_load_for_moe_layer(
        batch_size=global_batch,
        ep_parallelism=ep,
        num_experts=num_experts,
        topk=num_activated
    )

    print("\n[MoE 计算参数]")
    print(f"  token_per_ep_group: {token_per_ep_group}")
    print(f"  expert_per_ep_group: {expert_per_ep_group}")
    print(f"  m_per_group (M): {m_per_group}")
    print(f"  max_experts (G): {max_experts:.2f} → {math.ceil(max_experts)}")

    # 创建模型
    model = DeepSeekModel(
        name="DeepSeek-V3.2",
        config=config,
    )

    # 获取架构配置
    arch = get_arch_preset('SG2260E')

    # 分析性能
    print("\n[开始评估...]")
    analyzer = PerformanceAnalyzer(model, arch)

    # 获取结果
    summary = analyzer.get_summary(
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
    )

    print_section("层级延迟详情")

    # 收集层级信息
    layers_info = []
    total_elapse = 0
    total_comm = 0

    for layer in model.layers:
        info = layer.get_info()
        perf = info.get('perf', {})
        elapse = perf.get('elapse', 0)  # elapse not elapse_us
        comm = perf.get('comm_elapse', 0)  # comm_elapse not comm_elapse_us

        # 层重复次数
        count = info.get('count', 1)

        layers_info.append({
            'name': info.get('name', 'unknown'),
            'type': info.get('layer_type', 'unknown'),
            'count': count,
            'elapse': elapse,
            'comm': comm,
            'total': (elapse + comm) * count,
        })

        total_elapse += elapse * count
        total_comm += comm * count

    # 打印层级信息
    print(f"\n{'层名称':<20} {'类型':<12} {'x数量':<6} {'延迟(us)':<12} {'通信(us)':<12} {'总计(us)':<12}")
    print("-" * 80)

    for info in layers_info:
        print(f"{info['name']:<20} {info['type']:<12} x{info['count']:<5} "
              f"{info['elapse']:>10.2f}  {info['comm']:>10.2f}  {info['total']:>10.2f}")

    print("-" * 80)
    print(f"{'总计':<20} {'':<12} {'':<6} {total_elapse:>10.2f}  {total_comm:>10.2f}  {total_elapse + total_comm:>10.2f}")

    # 按类型汇总
    print_section("按层类型汇总")

    type_summary = {}
    for info in layers_info:
        t = info['type']
        if t not in type_summary:
            type_summary[t] = {'count': 0, 'elapse': 0, 'comm': 0}
        type_summary[t]['count'] += info['count']
        type_summary[t]['elapse'] += info['elapse'] * info['count']
        type_summary[t]['comm'] += info['comm'] * info['count']

    print(f"\n{'类型':<15} {'层数':<8} {'计算延迟(us)':<15} {'通信延迟(us)':<15} {'占比':<10}")
    print("-" * 70)

    total = total_elapse + total_comm
    for t, s in sorted(type_summary.items(), key=lambda x: -x[1]['elapse']):
        type_total = s['elapse'] + s['comm']
        pct = type_total / total * 100 if total > 0 else 0
        print(f"{t:<15} {s['count']:<8} {s['elapse']:>13.2f}  {s['comm']:>13.2f}  {pct:>8.1f}%")

    # MoE 层详细信息
    print_section("MoE 层算子详情")

    for layer in model.layers:
        if layer.layer_type == 'MoE':
            print(f"\n层: {layer.name}")
            print(f"{'算子':<25} {'类型':<15} {'延迟(us)':<12}")
            print("-" * 55)

            for op in layer.comp_ops:
                elapse = getattr(op, 'elapse', 0)
                print(f"{op.name:<25} {op.operator_type:<15} {elapse:>10.2f}")

            for op in layer.comm_ops:
                elapse = getattr(op, 'comm_elapse', 0)
                print(f"{op.name:<25} {op.comm_kind:<15} {elapse:>10.2f}")
            break  # 只打印第一个 MoE 层

    # DS_TPU 参考值对比
    print_section("与 DS_TPU 参考值对比")

    # DS_TPU 参考值 (修正后: 768 TFLOPS, batch=1536, dp=32, ep=32)
    # 原始 DS_TPU 使用 768*1024*1e9=786.4 TFLOPS，已修正为 768 TFLOPS
    ds_tpu_refs = {
        'MoE routed_gate': {'gemm': 'G=8, M=48', 'elapsed': 16.93, 'dma': 11.27},
        'MoE routed_up': {'gemm': 'G=8, M=48', 'elapsed': 16.93, 'dma': 11.27},
        'MoE routed_down': {'gemm': 'G=8, M=48', 'elapsed': 16.93, 'dma': 11.27},
    }

    print("\nDS_TPU 参考值 (修正后 768 TFLOPS):")
    print(f"  配置: batch=1536, dp=32, ep=32")
    print(f"  MoE GEMM: G=8, M=48, K=7168, N=2048")
    print(f"  参考延迟: ~16.93 us (总), ~11.27 us (DMA)")

    # 获取 Tier6 的 MoE GEMM 延迟
    tier6_moe_gemm = None
    for layer in model.layers:
        if layer.layer_type == 'MoE':
            for op in layer.comp_ops:
                if 'routed_gate' in op.name:
                    tier6_moe_gemm = getattr(op, 'elapse', 0)
                    break
            break

    if tier6_moe_gemm:
        ratio = tier6_moe_gemm / 16.93
        print(f"\nTier6 routed_gate 延迟: {tier6_moe_gemm:.2f} us")
        print(f"对比 DS_TPU 16.93 us: {ratio:.2f}x")

        if 0.9 <= ratio <= 1.1:
            print("[OK] Aligned within 10%")
        elif 0.8 <= ratio <= 1.2:
            print("[WARN] Difference within 20%")
        else:
            print("[FAIL] Large difference, needs investigation")

    print_section("总结")

    print(f"\nTier6 V3.2 Decode 总延迟: {total_elapse + total_comm:.2f} us")
    print(f"  - 计算延迟: {total_elapse:.2f} us")
    print(f"  - 通信延迟: {total_comm:.2f} us")

    return summary


def test_moe_gemm_direct():
    """直接测试 MoE GEMM 与 DS_TPU 对比"""

    print_section("MoE GEMM 直接对比测试")

    from llm_simulator.evaluators import GEMMEvaluator

    arch = get_arch_preset('SG2260E')
    evaluator = GEMMEvaluator(arch)

    # DS_TPU 配置: G=8, M=48, K=7168, N=2048
    # DS_TPU 参考值已修正 (768 TFLOPS)
    test_cases = [
        {'name': 'DS_TPU 默认', 'G': 8, 'M': 48, 'K': 7168, 'N': 2048, 'ds_ref': 16.93},
        {'name': '小 batch', 'G': 4, 'M': 1, 'K': 7168, 'N': 2048, 'ds_ref': None},
    ]

    print(f"\n{'测试用例':<15} {'GEMM 参数':<25} {'Tier6 (us)':<12} {'DS_TPU (us)':<12} {'比率':<8}")
    print("-" * 80)

    for tc in test_cases:
        # 使用 FP8 输入精度，对齐 DS_TPU W8A8 模式
        result = evaluator.evaluate(
            G=tc['G'], M=tc['M'], K=tc['K'], N=tc['N'],
            input_dtype='fp8', output_dtype='bf16'
        )

        gemm_str = f"G={tc['G']}, M={tc['M']}"
        ds_ref_str = f"{tc['ds_ref']:.2f}" if tc['ds_ref'] else "N/A"
        ratio_str = f"{result.latency_us / tc['ds_ref']:.2f}x" if tc['ds_ref'] else "N/A"

        print(f"{tc['name']:<15} {gemm_str:<25} {result.latency_us:>10.2f}  {ds_ref_str:>10}  {ratio_str:>6}")

        # 打印详细信息
        print(f"  → Tile: {result.best_tile}, Partition: {result.best_partition}")
        print(f"  → 计算: {result.compute_time_us:.2f} us, DMA: {result.memory_time_us:.2f} us")
        print(f"  → DRAM 流量: {result.dram_traffic_bytes / 1e6:.2f} MB")


if __name__ == '__main__':
    # 运行对齐测试
    test_v32_decode_alignment()

    # 运行 GEMM 直接对比
    test_moe_gemm_direct()
