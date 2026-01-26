"""
配置模块
"""

from .types import (
    # 枚举
    GanttTaskType,
    InferencePhase,
    BottleneckType,
    AllReduceAlgorithm,
    AllToAllAlgorithm,
    # 拓扑配置
    ChipConfig,
    BoardConfig,
    RackConfig,
    PodConfig,
    ConnectionConfig,
    HierarchicalTopology,
    # 硬件配置
    ChipHardwareConfig,
    NodeConfig,
    ClusterConfig,
    HardwareConfig,
    # 模型配置
    MoEConfig,
    MLAConfig,
    LLMModelConfig,
    # 运行时配置
    ProtocolConfig,
    NetworkInfraConfig,
    SimulationConfig,
    InferenceConfig,
    ParallelismStrategy,
    # 甘特图
    GanttTask,
    GanttResource,
    GanttChartData,
    # 统计
    PhaseTimeStats,
    SimulationStats,
    SimulationResult,
    # 互联图
    ChipNode,
    ChipLink,
    InterconnectGraph,
    ChipAssignment,
    ParallelGroupAssignment,
    # 辅助函数
    BYTES_PER_DTYPE,
    get_bytes_per_element,
    # 验证函数
    validate_mla_config,
    validate_moe_config,
    validate_model_config,
    validate_hardware_config,
    validate_parallelism_config,
)

from .config import (
    get_max_global_workers,
    set_max_global_workers,
)

__all__ = [
    # types
    'GanttTaskType',
    'InferencePhase',
    'BottleneckType',
    'AllReduceAlgorithm',
    'AllToAllAlgorithm',
    'ChipConfig',
    'BoardConfig',
    'RackConfig',
    'PodConfig',
    'ConnectionConfig',
    'HierarchicalTopology',
    'ChipHardwareConfig',
    'NodeConfig',
    'ClusterConfig',
    'HardwareConfig',
    'MoEConfig',
    'MLAConfig',
    'LLMModelConfig',
    'ProtocolConfig',
    'NetworkInfraConfig',
    'SimulationConfig',
    'InferenceConfig',
    'ParallelismStrategy',
    'GanttTask',
    'GanttResource',
    'GanttChartData',
    'PhaseTimeStats',
    'SimulationStats',
    'SimulationResult',
    'ChipNode',
    'ChipLink',
    'InterconnectGraph',
    'ChipAssignment',
    'ParallelGroupAssignment',
    'BYTES_PER_DTYPE',
    'get_bytes_per_element',
    'validate_mla_config',
    'validate_moe_config',
    'validate_model_config',
    'validate_hardware_config',
    'validate_parallelism_config',
    # config
    'get_max_global_workers',
    'set_max_global_workers',
]
