"""
FastAPI 接口模块

提供 LLM 推理模拟的 REST API 接口。
"""

import json
import logging
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
from sqlalchemy.orm import Session

from ..core.simulator import run_simulation
from ..core.database import get_db, get_db_session, init_db, Experiment, EvaluationTask, TaskStatus
from ..evaluators import ARCH_PRESETS
from ..config import (
    ProtocolConfig,
    NetworkInfraConfig,
    validate_model_config,
    validate_hardware_config,
    validate_parallelism_config,
    validate_mla_config,
    validate_moe_config,
    get_max_global_workers,
    set_max_global_workers,
)
from .websocket import ws_manager
from ..tasks import manager as task_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Pydantic 模型 - 严格类型定义（无默认值）
# ============================================

class ChipHardwareConfigRequest(BaseModel):
    """芯片硬件配置请求 - 所有核心字段必须明确指定"""
    chip_type: str = Field(..., description="芯片型号，如 SG2260E, A100, H100")
    compute_tflops_fp16: float = Field(..., gt=0, description="FP16 算力（TFLOPS）")
    memory_gb: float = Field(..., gt=0, description="显存容量（GB）")
    memory_bandwidth_gbps: float = Field(..., gt=0, description="显存带宽（GB/s）")

    # 可选的高级参数（有合理默认值）
    compute_tops_int8: float = Field(0.0, ge=0, description="INT8 算力（TOPS）")
    num_cores: int = Field(8, gt=0, description="计算核心数")
    memory_bandwidth_utilization: float = Field(0.9, gt=0, le=1, description="显存带宽利用率")
    l2_cache_mb: float = Field(16.0, gt=0, description="L2缓存容量（MB）")
    l2_bandwidth_gbps: float = Field(512.0, gt=0, description="L2缓存带宽（GB/s）")
    pcie_bandwidth_gbps: float = Field(64.0, gt=0, description="PCIe带宽（GB/s）")
    pcie_latency_us: float = Field(1.0, gt=0, description="PCIe延迟（微秒）")
    hbm_random_access_latency_ns: float = Field(100.0, gt=0, description="HBM随机访问延迟（纳秒）")

    # 微架构参数（可选，用于精确 GEMM 评估）
    cube_m: Optional[int] = Field(None, description="矩阵单元 M 维度")
    cube_k: Optional[int] = Field(None, description="矩阵单元 K 维度（累加维度）")
    cube_n: Optional[int] = Field(None, description="矩阵单元 N 维度")
    sram_size_kb: Optional[float] = Field(None, description="每核 SRAM 大小（KB）")
    sram_utilization: Optional[float] = Field(None, description="SRAM 可用比例（0-1）")
    lane_num: Optional[int] = Field(None, description="SIMD lane 数量")
    align_bytes: Optional[int] = Field(None, description="内存对齐字节数")
    compute_dma_overlap_rate: Optional[float] = Field(None, description="计算-搬运重叠率（0-1）")


class ModelConfigRequest(BaseModel):
    """模型配置请求 - 所有核心字段必须明确指定"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field(..., description="模型类型: dense 或 moe")
    hidden_size: int = Field(..., gt=0, description="隐藏层维度")
    num_layers: int = Field(..., gt=0, description="层数")
    num_attention_heads: int = Field(..., gt=0, description="注意力头数")
    num_kv_heads: int = Field(..., gt=0, description="KV 头数（GQA）")
    intermediate_size: int = Field(..., gt=0, description="FFN 中间层维度")
    vocab_size: int = Field(..., gt=0, description="词表大小")
    dtype: str = Field(..., description="数据类型: fp32, fp16, bf16, int8, int4")
    max_seq_length: int = Field(..., gt=0, description="最大序列长度")

    # 可选配置
    attention_type: str = Field("gqa", description="注意力类型: mha, gqa, mqa, mla")
    norm_type: str = Field("rmsnorm", description="归一化类型: layernorm, rmsnorm")
    moe_config: Optional[dict[str, Any]] = Field(None, description="MoE 配置")
    mla_config: Optional[dict[str, Any]] = Field(None, description="MLA 配置（DeepSeek）")


class InferenceConfigRequest(BaseModel):
    """推理配置请求 - 所有核心字段必须明确指定"""
    batch_size: int = Field(..., gt=0, description="批次大小")
    input_seq_length: int = Field(..., gt=0, description="输入序列长度")
    output_seq_length: int = Field(..., gt=0, description="输出序列长度")
    max_seq_length: int = Field(..., gt=0, description="最大序列长度")
    num_micro_batches: int = Field(1, gt=0, description="微批次数量（Pipeline Parallelism）")


class ParallelismConfigRequest(BaseModel):
    """并行策略配置请求 - manual 模式下所有并行度必须明确指定"""
    dp: int = Field(..., ge=1, description="数据并行度")
    tp: int = Field(..., ge=1, description="张量并行度")
    pp: int = Field(..., ge=1, description="流水线并行度")
    ep: int = Field(..., ge=1, description="专家并行度")
    sp: int = Field(1, ge=1, description="序列并行度")
    moe_tp: int = Field(1, ge=1, description="MoE 专家内张量并行度")


class HardwareConfigRequest(BaseModel):
    """硬件配置请求"""
    chip: ChipHardwareConfigRequest = Field(..., description="芯片配置")
    # node 和 cluster 配置可选，代码中有默认处理逻辑


class SimulationRequest(BaseModel):
    """模拟请求 - 使用严格类型"""
    topology: dict[str, Any]  # 拓扑配置保持灵活，因为结构复杂
    model: ModelConfigRequest
    inference: InferenceConfigRequest
    parallelism: ParallelismConfigRequest
    hardware: HardwareConfigRequest
    config: dict[str, Any] | None = None  # protocol_config 和 network_config 保持灵活


class SimulationResponse(BaseModel):
    """模拟响应"""
    ganttChart: dict[str, Any]
    stats: dict[str, Any]
    timestamp: float


class BenchmarkConfig(BaseModel):
    """Benchmark 配置"""
    id: str
    name: str
    model: dict[str, Any]
    inference: dict[str, Any]


class EvaluationRequest(BaseModel):
    """评估请求 - 保持向后兼容（使用 dict 类型）"""
    experiment_name: str
    description: str = ""

    # 配置文件引用（追溯来源）
    benchmark_name: Optional[str] = None
    topology_config_name: Optional[str] = None

    # 完整配置数据（保持灵活的 dict 类型，兼容现有前端）
    topology: dict[str, Any]
    model: dict[str, Any]  # 暂时保持 dict，避免破坏前端
    hardware: dict[str, Any]  # 暂时保持 dict
    inference: dict[str, Any]  # 暂时保持 dict

    # 搜索配置
    search_mode: str  # 'manual' or 'auto'
    manual_parallelism: Optional[dict[str, Any]] = None
    search_constraints: Optional[dict[str, Any]] = None

    # 任务并发配置
    max_workers: int = Field(4, ge=1, le=32, description="本任务的最大并发数")


class TaskSubmitResponse(BaseModel):
    """任务提交响应"""
    task_id: str
    message: str


class ExecutorConfigResponse(BaseModel):
    """全局资源池配置响应"""
    max_workers: int  # 全局资源池最大 worker 数量
    running_tasks: int  # 当前已分配的 worker 总数
    active_tasks: int  # 活跃任务数量
    note: str = "修改 max_workers 需要重启服务后生效"


class ExecutorConfigUpdateRequest(BaseModel):
    """全局资源池配置更新请求"""
    max_workers: int  # 全局资源池最大 worker 数量（1-32）


# 配置文件存储目录 (backend/configs/)
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
BENCHMARKS_DIR = CONFIGS_DIR / "benchmarks"
TOPOLOGIES_DIR = CONFIGS_DIR / "topologies"

# 确保目录存在
CONFIGS_DIR.mkdir(exist_ok=True)
BENCHMARKS_DIR.mkdir(exist_ok=True)
TOPOLOGIES_DIR.mkdir(exist_ok=True)


# ============================================
# FastAPI 应用
# ============================================

app = FastAPI(
    title="LLM 推理模拟器 API",
    description="基于拓扑的 GPU/加速器侧精细模拟服务",
    version="1.0.0",
)

# 配置 CORS - 使用环境变量控制允许的来源
allowed_origins_str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3100,http://localhost:3000,http://127.0.0.1:3100,http://127.0.0.1:3000"  # 默认：开发环境前端端口（支持 localhost 和 127.0.0.1）
)
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# 启动事件：初始化数据库和 WebSocket 回调
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("初始化数据库...")
    init_db()

    logger.info("恢复孤儿任务状态...")
    _recover_orphaned_tasks()

    logger.info("设置 WebSocket 广播回调...")
    task_manager.set_ws_broadcast_callback(ws_manager.broadcast_task_update)
    logger.info("应用启动完成")


def _recover_orphaned_tasks():
    """
    恢复孤儿任务状态

    服务重启后，将所有处于 RUNNING 或 PENDING 状态的任务标记为 FAILED，
    避免状态不一致问题。
    """
    with get_db_session() as db:
        orphaned_tasks = db.query(EvaluationTask).filter(
            EvaluationTask.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING])
        ).all()

        if orphaned_tasks:
            logger.info(f"发现 {len(orphaned_tasks)} 个孤儿任务，标记为失败")
            for task in orphaned_tasks:
                task.status = TaskStatus.FAILED
                task.error = "服务重启导致任务中断"
                if task.started_at and not task.completed_at:
                    from datetime import datetime
                    task.completed_at = datetime.utcnow()

            db.commit()
            logger.info("孤儿任务状态恢复完成")
        else:
            logger.info("未发现孤儿任务")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "LLM 推理模拟器 API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/presets/chips")
async def get_chip_presets():
    """
    获取所有芯片预设配置

    返回后端定义的所有硬件架构配置，包括：
    - 芯片名称
    - 算力精度 (BF16/FP16)
    - 峰值算力 (TFLOPS)
    - 核心数量
    - 内存带宽等
    """
    result = []
    for chip_id, arch in ARCH_PRESETS.items():
        result.append({
            "id": chip_id,
            "name": arch.name,
            "flops_dtype": arch.flops_dtype,
            "compute_tflops": round(arch.flops_per_second / 1e12, 2),  # 四舍五入到2位小数
            "num_cores": arch.num_cores,
            "sram_size_mb": round(arch.sram_size_bytes / (1024 * 1024), 2),
            "dram_bandwidth_gbps": round(arch.dram_bandwidth_bytes / 1e9, 2),
            "intra_bw_gbps": round(arch.intra_bw / 1e9, 2),
            "inter_bw_gbps": round(arch.inter_bw / 1e9, 2),
            # 粗粒度延迟（向后兼容）
            "intra_latency_us": round(arch.intra_latency_us, 2),
            "inter_latency_us": round(arch.inter_latency_us, 2),
            # 细粒度通信延迟（新增，单位: us）
            "comm_latency": {
                "chip_to_chip_us": round(arch.comm_latency.chip_to_chip_us, 2),
                "comm_start_overhead_us": round(arch.comm_latency.comm_start_overhead_us, 2),
                "memory_read_latency_us": round(arch.comm_latency.memory_read_latency_us, 2),
                "memory_write_latency_us": round(arch.comm_latency.memory_write_latency_us, 2),
            },
        })
    return {"chips": result}


@app.get("/api/presets/runtime")
async def get_runtime_presets():
    """
    获取运行时配置预设

    返回协议和网络基础设施的默认配置值，供前端显示和配置。
    """
    # 使用默认值创建配置对象
    protocol = ProtocolConfig()
    network = NetworkInfraConfig()

    return {
        "protocol": {
            "rtt_tp_us": protocol.rtt_tp_us,
            "rtt_ep_us": protocol.rtt_ep_us,
            "bandwidth_utilization": protocol.bandwidth_utilization,
            "sync_latency_us": protocol.sync_latency_us,
        },
        "network": {
            "switch_delay_us": network.switch_delay_us,
            "cable_delay_us": network.cable_delay_us,
            "link_delay_us": network.link_delay_us,
        },
    }


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    运行 LLM 推理模拟

    Args:
        request: 模拟请求，包含拓扑、模型、推理、并行策略、硬件配置

    Returns:
        模拟结果，包含甘特图数据和统计信息
    """
    try:
        # 将 Pydantic 模型转换为 dict
        model_dict = request.model.model_dump()
        inference_dict = request.inference.model_dump()
        parallelism_dict = request.parallelism.model_dump()
        hardware_dict = request.hardware.model_dump()

        logger.info(f"开始模拟: model={model_dict['model_name']}")
        result = run_simulation(
            topology_dict=request.topology,
            model_dict=model_dict,
            inference_dict=inference_dict,
            parallelism_dict=parallelism_dict,
            hardware_dict=hardware_dict,
            config_dict=request.config,
        )
        logger.info("模拟完成")
        return SimulationResponse(**result)
    except ValueError as e:
        logger.warning(f"配置验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        logger.warning(f"配置缺少必要字段: {e}")
        raise HTTPException(status_code=400, detail=f"配置缺少必要字段: {e}")
    except TypeError as e:
        logger.warning(f"配置类型错误: {e}")
        raise HTTPException(status_code=400, detail=f"配置类型错误: {e}")
    except Exception as e:
        logger.error(f"模拟失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模拟失败: {str(e)}")


@app.post("/api/validate")
async def validate_config(request: SimulationRequest):
    """
    验证配置是否有效

    检查：
    - 模型配置的有效性
    - 硬件配置的合理性
    - 并行策略的正确性
    - MLA/MoE 配置的完整性
    - 拓扑中芯片数量是否满足并行策略需求
    """
    errors = []

    # 将 Pydantic 模型转换为 dict（验证函数期望 dict 类型）
    model_dict = request.model.model_dump()
    hardware_dict = request.hardware.model_dump()
    parallelism_dict = request.parallelism.model_dump()

    # 验证模型配置
    try:
        validate_model_config(model_dict)
    except ValueError as e:
        errors.append(f"模型配置: {e}")

    # 验证硬件配置
    try:
        validate_hardware_config(hardware_dict)
    except ValueError as e:
        errors.append(f"硬件配置: {e}")

    # 验证并行策略
    try:
        validate_parallelism_config(parallelism_dict, model_dict)
    except ValueError as e:
        errors.append(f"并行策略: {e}")

    # 验证 MLA 配置（如果存在）
    mla_dict = model_dict.get("mla_config")
    if mla_dict:
        try:
            validate_mla_config(mla_dict)
        except ValueError as e:
            errors.append(f"MLA 配置: {e}")

    # 验证 MoE 配置（如果存在）
    moe_dict = model_dict.get("moe_config")
    if moe_dict:
        try:
            validate_moe_config(moe_dict)
        except ValueError as e:
            errors.append(f"MoE 配置: {e}")

    # 验证芯片数量
    topology = request.topology

    required_chips = (
        parallelism_dict["dp"] *
        parallelism_dict["tp"] *
        parallelism_dict["pp"] *
        parallelism_dict["ep"]
    )

    available_chips = 0
    for pod in topology.get("pods", []):
        for rack in pod.get("racks", []):
            for board in rack.get("boards", []):
                available_chips += len(board.get("chips", []))

    if available_chips < required_chips:
        errors.append(f"芯片数量不足: 需要 {required_chips} 个，拓扑中只有 {available_chips} 个")

    if errors:
        logger.warning(f"配置验证失败: {errors}")
        return {
            "valid": False,
            "errors": errors,
            "required_chips": required_chips,
            "available_chips": available_chips,
        }

    logger.info("配置验证通过")
    return {
        "valid": True,
        "required_chips": required_chips,
        "available_chips": available_chips,
    }


# ============================================
# 拓扑配置管理 API
# ============================================

class TopologyConfigRequest(BaseModel):
    """拓扑配置请求"""
    name: str = Field(..., description="配置名称（唯一标识）")
    description: Optional[str] = Field(None, description="配置描述")
    pod_count: int = Field(1, description="Pod 数量")
    racks_per_pod: int = Field(1, description="每个 Pod 的 Rack 数量")
    board_configs: Optional[dict] = Field(None, description="Board 配置")
    rack_config: Optional[dict] = Field(None, description="Rack 配置")
    switch_config: Optional[dict] = Field(None, description="交换机配置")
    manual_connections: Optional[dict] = Field(None, description="手动连接配置")
    generated_topology: Optional[dict] = Field(None, description="生成的完整拓扑数据")
    chip_configs: Optional[list] = Field(None, description="芯片硬件配置列表")
    network_config: Optional[dict] = Field(None, description="网络配置")
    comm_latency_config: Optional[dict] = Field(None, description="通信延迟配置")


@app.get("/api/topologies")
async def list_topologies():
    """
    获取所有保存的拓扑配置列表
    """
    topologies = []
    for file_path in TOPOLOGIES_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 返回摘要信息
                topologies.append({
                    "name": data.get("name", file_path.stem),
                    "description": data.get("description"),
                    "pod_count": data.get("pod_count"),
                    "racks_per_pod": data.get("racks_per_pod"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
        except Exception as e:
            logger.warning(f"读取拓扑配置文件失败 {file_path}: {e}")

    # 按更新时间倒序排列
    topologies.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return {"topologies": topologies}


@app.get("/api/topologies/{name}")
async def get_topology_config(name: str):
    """
    获取指定名称的拓扑配置
    """
    file_path = TOPOLOGIES_DIR / f"{name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"拓扑配置不存在: {name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取拓扑配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/topologies")
async def create_topology_config(config: TopologyConfigRequest):
    """
    创建新的拓扑配置
    """
    from datetime import datetime

    file_path = TOPOLOGIES_DIR / f"{config.name}.json"

    # 检查是否已存在
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"拓扑配置已存在: {config.name}")

    try:
        data = config.model_dump()
        now = datetime.now().isoformat()
        data["created_at"] = now
        data["updated_at"] = now

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"创建拓扑配置: {config.name}")
        return {"success": True, "name": config.name}
    except Exception as e:
        logger.error(f"保存拓扑配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/topologies/{name}")
async def update_topology_config(name: str, config: TopologyConfigRequest):
    """
    更新拓扑配置
    """
    from datetime import datetime

    file_path = TOPOLOGIES_DIR / f"{name}.json"

    # 读取现有配置以保留 created_at
    created_at = None
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("created_at")
        except Exception:
            pass

    try:
        data = config.model_dump()
        data["created_at"] = created_at or datetime.now().isoformat()
        data["updated_at"] = datetime.now().isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"更新拓扑配置: {name}")
        return {"success": True, "name": name}
    except Exception as e:
        logger.error(f"更新拓扑配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/topologies/{name}")
async def delete_topology_config(name: str):
    """
    删除拓扑配置
    """
    file_path = TOPOLOGIES_DIR / f"{name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"拓扑配置不存在: {name}")

    try:
        file_path.unlink()
        logger.info(f"删除拓扑配置: {name}")
        return {"success": True, "name": name}
    except Exception as e:
        logger.error(f"删除拓扑配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Benchmark 管理 API
# ============================================

@app.get("/api/benchmarks")
async def list_benchmarks():
    """
    获取所有自定义 Benchmark 列表

    从 benchmarks 目录读取所有 JSON 文件
    """
    benchmarks = []
    for file_path in BENCHMARKS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                benchmarks.append(data)
        except Exception as e:
            logger.warning(f"读取 benchmark 文件失败 {file_path}: {e}")
    return {"benchmarks": benchmarks}


@app.get("/api/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """
    获取单个 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmarks")
async def create_benchmark(benchmark: BenchmarkConfig):
    """
    创建新的 Benchmark 配置

    保存为 JSON 文件到 benchmarks 目录
    """
    file_path = BENCHMARKS_DIR / f"{benchmark.id}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"创建 benchmark: {benchmark.id}")
        return {"success": True, "id": benchmark.id}
    except Exception as e:
        logger.error(f"保存 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/benchmarks/{benchmark_id}")
async def update_benchmark(benchmark_id: str, benchmark: BenchmarkConfig):
    """
    更新 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(benchmark.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"更新 benchmark: {benchmark_id}")
        return {"success": True, "id": benchmark_id}
    except Exception as e:
        logger.error(f"更新 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/benchmarks/{benchmark_id}")
async def delete_benchmark(benchmark_id: str):
    """
    删除 Benchmark 配置
    """
    file_path = BENCHMARKS_DIR / f"{benchmark_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark 不存在: {benchmark_id}")

    try:
        file_path.unlink()
        logger.info(f"删除 benchmark: {benchmark_id}")
        return {"success": True, "id": benchmark_id}
    except Exception as e:
        logger.error(f"删除 benchmark 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 评估任务管理 API
# ============================================

@app.post("/api/evaluation/submit", response_model=TaskSubmitResponse)
async def submit_evaluation(request: EvaluationRequest):
    """
    提交评估任务到后台执行

    Args:
        request: 评估请求，包含实验名称、拓扑、模型、硬件、推理配置等

    Returns:
        任务 ID 和提示消息
    """
    try:
        # EvaluationRequest 使用 dict 类型（向后兼容），直接传递
        task_id = task_manager.create_and_submit_task(
            experiment_name=request.experiment_name,
            description=request.description,
            topology=request.topology,
            model_config=request.model,
            inference_config=request.inference,
            search_mode=request.search_mode,
            max_workers=request.max_workers,
            benchmark_name=request.benchmark_name,
            topology_config_name=request.topology_config_name,
            manual_parallelism=request.manual_parallelism,
            search_constraints=request.search_constraints,
        )
        logger.info(f"评估任务已提交: {task_id}, max_workers={request.max_workers}")
        return TaskSubmitResponse(
            task_id=task_id,
            message=f"评估任务已提交（使用 {request.max_workers} workers），正在后台运行"
        )
    except Exception as e:
        logger.error(f"提交评估任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"提交评估任务失败: {str(e)}")


@app.get("/api/evaluation/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """获取任务状态"""
    status = task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return status


@app.get("/api/evaluation/tasks/{task_id}/results")
async def get_task_results_endpoint(task_id: str):
    """获取任务的完整结果"""
    results = task_manager.get_task_results(task_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return results


@app.post("/api/evaluation/tasks/{task_id}/cancel")
async def cancel_task_endpoint(task_id: str):
    """取消任务"""
    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {"success": True, "message": "任务已取消"}


@app.delete("/api/evaluation/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    """删除任务（及其结果）"""
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {"success": True, "message": "任务已删除"}


@app.get("/api/evaluation/tasks")
async def list_tasks(
    status: Optional[str] = None,
    experiment_name: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    获取任务列表

    Args:
        status: 过滤状态 (pending, running, completed, failed, cancelled)
        experiment_name: 过滤实验名称
        limit: 返回数量限制
    """
    query = db.query(EvaluationTask).join(Experiment)

    if status:
        try:
            status_enum = TaskStatus(status)
            query = query.filter(EvaluationTask.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的状态: {status}")

    if experiment_name:
        query = query.filter(Experiment.name == experiment_name)

    tasks = query.order_by(EvaluationTask.created_at.desc()).limit(limit).all()

    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "experiment_name": task.experiment.name,
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
            for task in tasks
        ]
    }


@app.get("/api/evaluation/running")
async def get_running_tasks_endpoint():
    """获取所有运行中的任务"""
    return {"tasks": task_manager.get_running_tasks()}


@app.get("/api/evaluation/config", response_model=ExecutorConfigResponse)
async def get_executor_config():
    """
    获取全局资源池配置

    返回全局资源池的配置信息，包括最大 worker 数量和当前分配情况
    """
    info = task_manager.get_executor_info()
    return ExecutorConfigResponse(
        max_workers=info["max_workers"],
        running_tasks=info["running_tasks"],
        active_tasks=info["active_tasks"],
    )


@app.put("/api/evaluation/config")
async def update_executor_config(request: ExecutorConfigUpdateRequest):
    """
    更新全局资源池配置

    更新全局资源池最大 worker 数量，需要重启服务后生效

    Args:
        request: 配置更新请求，包含 max_workers (1-32)

    Returns:
        更新后的配置信息
    """
    try:
        set_max_global_workers(request.max_workers)
        info = task_manager.get_executor_info()
        return {
            "success": True,
            "message": f"全局资源池最大 worker 数量已设置为 {request.max_workers}，重启服务后生效",
            "current_max_workers": info["max_workers"],  # 当前运行中的配置
            "new_max_workers": request.max_workers,  # 新配置（重启后生效）
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@app.get("/api/evaluation/experiments")
async def list_experiments(db: Session = Depends(get_db)):
    """获取所有实验列表"""
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return {
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "total_tasks": exp.total_tasks,
                "completed_tasks": exp.completed_tasks,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            for exp in experiments
        ]
    }


@app.get("/api/evaluation/experiments/{experiment_id}")
async def get_experiment_details(experiment_id: int, db: Session = Depends(get_db)):
    """获取实验详情（包含所有任务）"""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

        tasks = db.query(EvaluationTask).filter(EvaluationTask.experiment_id == experiment_id).all()

        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "total_tasks": experiment.total_tasks,
            "completed_tasks": experiment.completed_tasks,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "updated_at": experiment.updated_at.isoformat() if experiment.updated_at else None,
            "tasks": [
                {
                    "id": task.id,
                    "task_id": task.task_id,
                    "experiment_id": task.experiment_id,
                    "status": task.status.value,
                    "progress": task.progress,
                    "message": task.message,
                    "error": task.error,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "config_snapshot": task.config_snapshot,
                    "benchmark_name": task.benchmark_name,
                    "topology_config_name": task.topology_config_name,
                    "search_mode": task.search_mode,
                    "manual_parallelism": task.manual_parallelism,
                    "search_constraints": task.search_constraints,
                    "search_stats": task.search_stats,
                }
                for task in tasks
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实验详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@app.delete("/api/evaluation/experiments/{experiment_id}")
async def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """删除实验及其所有任务和结果"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    try:
        # 删除实验（级联删除任务和结果）
        db.delete(experiment)
        db.commit()
        return {"message": f"实验 '{experiment.name}' 已删除"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================
# WebSocket 端点
# ============================================

@app.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 端点：实时推送任务状态更新

    客户端连接后会收到所有任务的状态变化推送
    消息格式:
    {
        "type": "task_update",
        "task_id": "...",
        "status": "running",
        "progress": 50.0,
        "message": "...",
        ...
    }
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # 保持连接，等待客户端消息（目前不处理客户端消息）
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import os
    import uvicorn
    from pathlib import Path
    from dotenv import load_dotenv

    # 加载 Tier6+model/.env 共享配置
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    port = int(os.environ.get("VITE_API_PORT", "8001"))
    print(f"Tier6+互联建模平台启动在端口: {port}")
    uvicorn.run("llm_simulator.api:app", host="0.0.0.0", port=port, reload=True)
