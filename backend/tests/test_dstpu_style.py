#!/usr/bin/env python3
"""
DS_TPU 风格架构测试

验证新的三层抽象体系 (Operator -> Layer -> Model) 和 PerformanceAnalyzer
"""

import sys
import json
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

from llm_simulator.models import DeepSeekModel, LlamaModel
from llm_simulator.models.deepseek import create_deepseek_v3
from llm_simulator.models.llama import create_llama_7b
from llm_simulator.analyzer import PerformanceAnalyzer, analyze_model
from llm_simulator.evaluators import get_arch_preset


def test_deepseek_model_structure():
    """测试 DeepSeek 模型结构"""
    print("=" * 80)
    print("测试 DeepSeek V3 模型结构")
    print("=" * 80)

    model = create_deepseek_v3(
        batch_size=1,
        seq_len=4096,
        tp=4,
        moe_tp=1,
        ep=32,
        comm_protocol=1,
        is_prefill=True,
    )

    print(f"\n模型名称: {model.name}")
    print(f"模型类型: {model.model_type}")
    print(f"层数: {len(model.layers)}")
    print(f"层数映射: {model.layer_counts}")
    print(f"算子类型: {model.operator_types}")

    # 统计算子数量
    total_ops = 0
    for op_type, ops in model.operator_map.items():
        print(f"  {op_type}: {len(ops)} 个算子")
        total_ops += len(ops)
    print(f"总算子数: {total_ops}")

    return True


def test_llama_model_structure():
    """测试 Llama 模型结构"""
    print("\n" + "=" * 80)
    print("测试 Llama 7B 模型结构")
    print("=" * 80)

    model = create_llama_7b(
        batch_size=1,
        seq_len=2048,
        tp=2,
        comm_protocol=1,
    )

    print(f"\n模型名称: {model.name}")
    print(f"模型类型: {model.model_type}")
    print(f"层数: {len(model.layers)}")
    print(f"层数映射: {model.layer_counts}")
    print(f"算子类型: {model.operator_types}")

    # 统计算子数量
    total_ops = 0
    for op_type, ops in model.operator_map.items():
        print(f"  {op_type}: {len(ops)} 个算子")
        total_ops += len(ops)
    print(f"总算子数: {total_ops}")

    return True


def test_performance_analyzer():
    """测试性能分析器"""
    print("\n" + "=" * 80)
    print("测试 PerformanceAnalyzer")
    print("=" * 80)

    # 创建小规模 DeepSeek 模型用于测试
    model = DeepSeekModel(
        name="deepseek-test",
        config={
            'hidden_dim': 4096,
            'inter_dim': 11008,
            'vocab_size': 32000,
            'n_layers': 4,
            'n_dense_layers': 1,
            'n_moe_layers': 3,
            'num_heads': 32,
            'head_dim': 128,
            'kv_lora_rank': 256,
            'q_lora_rank': 512,
            'num_experts': 8,
            'num_activated_experts': 2,
            'num_shared_experts': 1,
            'expert_inter_dim': 1024,
            'batch_size': 1,
            'seq_len': 512,
            'tp': 2,
            'moe_tp': 1,
            'ep': 2,
            'comm_protocol': 1,
            'is_prefill': True,
        }
    )

    print(f"\n模型: {model.name}")
    print(f"算子类型: {model.operator_types}")

    # 运行分析
    arch = get_arch_preset('SG2260E')
    analyzer = PerformanceAnalyzer(model, arch)
    summary = analyzer.get_summary(batch_size=1, seq_len=512)

    print(f"\n性能摘要:")
    print(f"  总延迟: {summary['performance']['total_elapse_ms']:.3f} ms")
    print(f"  通信延迟: {summary['performance']['comm_elapse_us']:.3f} us")
    print(f"  总 FLOPs: {summary['performance']['total_flops']:,.0f}")
    print(f"  DRAM 占用: {summary['performance']['dram_occupy']:,.0f} bytes")
    if summary['performance']['mfu']:
        print(f"  MFU: {summary['performance']['mfu']:.4f}")
    print(f"  TPS: {summary['performance']['tps']:.2f}")

    # 打印层信息
    print(f"\n层级性能:")
    for layer_name, layer_info in summary['performance']['layers'].items():
        perf = layer_info['perf']
        count = layer_info.get('count', 1)
        print(f"  {layer_name} (x{count}):")
        print(f"    延迟: {perf['elapse']:.3f} us")
        print(f"    计算: {perf['comp_elapse']:.3f} us")
        print(f"    通信: {perf['comm_elapse']:.3f} us")
        print(f"    计算算子: {len(layer_info['comp_operators'])}")
        print(f"    通信算子: {len(layer_info['comm_operators'])}")

    return True


def test_operator_details():
    """测试算子级详情"""
    print("\n" + "=" * 80)
    print("测试算子级详情输出")
    print("=" * 80)

    model = create_llama_7b(batch_size=1, seq_len=1024, tp=1)

    arch = get_arch_preset('SG2260E')
    analyzer = PerformanceAnalyzer(model, arch)
    summary = analyzer.get_summary(batch_size=1, seq_len=1024)

    # 检查算子详情
    print("\n示例算子详情 (MHA 层):")
    mha_info = summary['performance']['layers'].get('mha')
    if mha_info:
        for op in mha_info['comp_operators'][:3]:  # 前3个
            print(f"\n  算子: {op['name']}")
            print(f"    类型: {op['operator_type']}")
            print(f"    延迟: {op['elapse']:.3f} us")
            print(f"    FLOPs: {op['flops']:,}")
            if op.get('best_tile'):
                print(f"    Best Tile: {op['best_tile']}")
            if op.get('urate'):
                print(f"    利用率: {op['urate']:.2%}")

    return True


def test_json_output():
    """测试 JSON 输出格式"""
    print("\n" + "=" * 80)
    print("测试 JSON 输出格式")
    print("=" * 80)

    model = create_llama_7b(batch_size=1, seq_len=512, tp=1)
    summary = analyze_model(model, 'SG2260E', batch_size=1, seq_len=512)

    # 导出 JSON
    json_str = json.dumps(summary, indent=2, default=str)
    print(f"\nJSON 输出长度: {len(json_str)} 字符")
    print(f"\n前 2000 字符预览:")
    print(json_str[:2000])

    return True


def main():
    """主测试函数"""
    results = []

    try:
        results.append(("DeepSeek 模型结构", test_deepseek_model_structure()))
    except Exception as e:
        print(f"DeepSeek 模型结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DeepSeek 模型结构", False))

    try:
        results.append(("Llama 模型结构", test_llama_model_structure()))
    except Exception as e:
        print(f"Llama 模型结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Llama 模型结构", False))

    try:
        results.append(("性能分析器", test_performance_analyzer()))
    except Exception as e:
        print(f"性能分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("性能分析器", False))

    try:
        results.append(("算子详情", test_operator_details()))
    except Exception as e:
        print(f"算子详情测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("算子详情", False))

    try:
        results.append(("JSON 输出", test_json_output()))
    except Exception as e:
        print(f"JSON 输出测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("JSON 输出", False))

    # 汇总
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
