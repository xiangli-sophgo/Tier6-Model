"""
精确评估器模块

提供 GEMM、FlashAttention 等算子的精确性能评估。
"""

from .arch_config import AcceleratorMicroArch
from .presets import (
    SG2260E_ARCH,
    H100_SXM_ARCH,
    A100_ARCH,
    ARCH_PRESETS,
    get_arch_preset,
)
from .gemm_eval import (
    GEMMResult,
    GEMMEvaluator,
    get_gemm_evaluator,
    eval_gemm,
)
from .fa2_eval import (
    FA2Result,
    FA2Evaluator,
    get_fa2_evaluator,
    eval_fa2,
)
from .softmax_eval import softmax_theoretical_and_real
from .rmsnorm_eval import (
    RMSNormResult,
    RMSNormEvaluator,
    rmsnorm_theoretical_and_real,
    get_rmsnorm_evaluator,
    eval_rmsnorm,
)
from .comm_eval import (
    CommResult,
    AllReduceEval,
    AllGatherEval,
    ReduceScatterEval,
    DispatchEval,
    CombineEval,
    init_comm_evaluators,
    get_allreduce_eval,
    get_allgather_eval,
    get_reducescatter_eval,
    get_dispatch_eval,
    get_combine_eval,
    eval_allreduce,
    eval_allgather,
    eval_reducescatter,
)
from .utils import ceil_div, align_up

__all__ = [
    # 配置
    'AcceleratorMicroArch',
    'SG2260E_ARCH',
    'H100_SXM_ARCH',
    'A100_ARCH',
    'ARCH_PRESETS',
    'get_arch_preset',
    # GEMM
    'GEMMResult',
    'GEMMEvaluator',
    'get_gemm_evaluator',
    'eval_gemm',
    # FA2
    'FA2Result',
    'FA2Evaluator',
    'get_fa2_evaluator',
    'eval_fa2',
    # Softmax
    'softmax_theoretical_and_real',
    # RMSNorm
    'RMSNormResult',
    'RMSNormEvaluator',
    'rmsnorm_theoretical_and_real',
    'get_rmsnorm_evaluator',
    'eval_rmsnorm',
    # 通信
    'CommResult',
    'AllReduceEval',
    'AllGatherEval',
    'ReduceScatterEval',
    'DispatchEval',
    'CombineEval',
    'init_comm_evaluators',
    'get_allreduce_eval',
    'get_allgather_eval',
    'get_reducescatter_eval',
    'get_dispatch_eval',
    'get_combine_eval',
    'eval_allreduce',
    'eval_allgather',
    'eval_reducescatter',
    # 工具
    'ceil_div',
    'align_up',
]
