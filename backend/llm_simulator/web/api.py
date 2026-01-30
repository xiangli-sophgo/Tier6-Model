"""
FastAPI 接口模块

提供 LLM 推理模拟的 REST API 接口。
"""
# Route order fixed: static routes before dynamic routes

import asyncio
import json
import logging
import os
import uuid
import yaml
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional
from sqlalchemy.orm import Session

from ..core.simulator import run_simulation
from ..core.database import get_db, get_db_session, init_db, Experiment, EvaluationTask, EvaluationResult, TaskStatus
from ..core.model_utils import calculate_params_from_dict, format_params
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
from ..tasks import manager as task_manager
from ..tasks.deployment import count_topology_chips, calculate_required_chips
from .column_presets import router as column_presets_router

# 导入 Pydantic 模型（从 schemas.py）
from .schemas import (
    ChipHardwareConfigRequest,
    HardwareConfigRequest,
    ModelConfigRequest,
    InferenceConfigRequest,
    ParallelismConfigRequest,
    SimulationRequest,
    SimulationResponse,
    BenchmarkConfig,
    EvaluationRequest,
    TaskSubmitResponse,
    ExecutorConfigResponse,
    ExecutorConfigUpdateRequest,
    ExperimentUpdateRequest,
    BatchDeleteExperimentsRequest,
    BatchDeleteResultsRequest,
    ExperimentExportData,
    ExportInfo,
    CheckImportResult,
    ImportConfigItem,
    ImportExecuteRequest,
    ImportResult,
    TopologyConfigRequest,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

# 注册列配置方案路由
app.include_router(column_presets_router)


# 启动事件：初始化数据库和 WebSocket
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    import asyncio

    logger.info("初始化数据库...")
    init_db()

    logger.info("恢复孤儿任务状态...")
    _recover_orphaned_tasks()

    logger.info("注册事件循环到任务管理器...")
    task_manager.set_main_loop(asyncio.get_running_loop())
    print("[DEBUG STARTUP] Main event loop registered to task manager!")
    print("[DEBUG STARTUP] Application startup complete!")
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
            # C2C 带宽
            "c2c_bw_unidirectional_gbps": round(arch.c2c_bw_unidirectional_gbps, 1),
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


@app.post("/api/model/calculate-params")
async def calculate_model_params_api(model: dict[str, Any]):
    """
    计算模型参数量

    Args:
        model: 模型配置字典

    Returns:
        参数量信息，包括原始数值和格式化字符串
    """
    try:
        params = calculate_params_from_dict(model)
        return {
            "params": params,
            "formatted": format_params(params),
        }
    except Exception as e:
        logger.error(f"计算模型参数量失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


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

    # 使用统一的芯片计算函数
    required_chips = calculate_required_chips(parallelism_dict, model_dict)
    available_chips = count_topology_chips(topology)

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

@app.get("/api/topologies")
async def list_topologies():
    """
    获取所有保存的拓扑配置列表
    """
    topologies = []
    for file_path in TOPOLOGIES_DIR.glob("*.yaml"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                # 从 generated_topology 中统计实际数量
                total_pods = 0
                total_racks = 0
                total_boards = 0
                total_chips = 0

                if "generated_topology" in data and data["generated_topology"]:
                    topology = data["generated_topology"]
                    # 使用公共函数统计芯片数
                    total_chips = count_topology_chips(topology)
                    # 统计其他层级
                    if "pods" in topology and topology["pods"]:
                        total_pods = len(topology["pods"])
                        for pod in topology["pods"]:
                            if "racks" in pod and pod["racks"]:
                                total_racks += len(pod["racks"])
                                for rack in pod["racks"]:
                                    if "boards" in rack and rack["boards"]:
                                        total_boards += len(rack["boards"])

                # 返回摘要信息
                topologies.append({
                    "name": data.get("name", file_path.stem),
                    "description": data.get("description"),
                    "pod_count": data.get("pod_count"),
                    "racks_per_pod": data.get("racks_per_pod"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    # 添加统计数量
                    "total_pods": total_pods,
                    "total_racks": total_racks,
                    "total_boards": total_boards,
                    "total_chips": total_chips,
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
    file_path = TOPOLOGIES_DIR / f"{name}.yaml"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"拓扑配置不存在: {name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"读取拓扑配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/topologies")
async def create_topology_config(config: TopologyConfigRequest):
    """
    创建新的拓扑配置
    """
    from datetime import datetime

    file_path = TOPOLOGIES_DIR / f"{config.name}.yaml"

    # 检查是否已存在
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"拓扑配置已存在: {config.name}")

    try:
        data = config.model_dump(exclude_none=True)
        now = datetime.now().isoformat()
        data["created_at"] = now
        data["updated_at"] = now

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

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

    file_path = TOPOLOGIES_DIR / f"{name}.yaml"

    # 读取现有配置以保留 created_at
    created_at = None
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f)
                created_at = existing.get("created_at")
        except Exception:
            pass

    try:
        data = config.model_dump(exclude_none=True)
        data["created_at"] = created_at or datetime.now().isoformat()
        data["updated_at"] = datetime.now().isoformat()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

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
    file_path = TOPOLOGIES_DIR / f"{name}.yaml"
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

@app.get("/api/debug/paths")
async def debug_paths():
    """调试端点：显示配置文件路径信息"""
    import os
    json_files = list(BENCHMARKS_DIR.glob("*.json"))
    return {
        "api_file": str(Path(__file__).resolve()),
        "working_directory": os.getcwd(),
        "CONFIGS_DIR": str(CONFIGS_DIR.resolve()),
        "BENCHMARKS_DIR": str(BENCHMARKS_DIR.resolve()),
        "TOPOLOGIES_DIR": str(TOPOLOGIES_DIR.resolve()),
        "benchmarks_dir_exists": BENCHMARKS_DIR.exists(),
        "benchmarks_dir_is_dir": BENCHMARKS_DIR.is_dir(),
        "json_file_count": len(json_files),
        "json_files": [f.name for f in json_files[:5]],
    }

@app.get("/api/benchmarks")
async def list_benchmarks():
    """
    获取所有自定义 Benchmark 列表

    从 benchmarks 目录读取所有 JSON 文件
    """
    benchmarks = []
    json_files = list(BENCHMARKS_DIR.glob("*.json"))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                benchmarks.append(data)
        except Exception as e:
            logger.warning(f"读取 benchmark 文件失败 {file_path}: {e}")

    logger.info(f"返回 {len(benchmarks)} 个 benchmarks")
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
            enable_tile_search=request.enable_tile_search,
            enable_partition_search=request.enable_partition_search,
            max_simulated_tokens=request.max_simulated_tokens,
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
                # 统计所有评估结果数量（自动搜索中每个方案算一个任务）
                "total_tasks": sum(len(t.results) for t in exp.tasks),
                # 统计已完成任务的结果数量
                "completed_tasks": sum(len(t.results) for t in exp.tasks if t.status == TaskStatus.COMPLETED),
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            for exp in experiments
        ]
    }


# ============================================
# 实验静态路由（必须在动态路由 {experiment_id} 之前）
# ============================================

# 存储临时导入文件
_import_temp_files: dict[str, dict[str, Any]] = {}


@app.post("/api/evaluation/experiments/batch-delete")
async def batch_delete_experiments(
    request: BatchDeleteExperimentsRequest,
    db: Session = Depends(get_db)
):
    """批量删除实验"""
    try:
        if not request.experiment_ids:
            raise HTTPException(status_code=400, detail="实验 ID 列表不能为空")

        # 查询要删除的实验
        experiments = db.query(Experiment).filter(
            Experiment.id.in_(request.experiment_ids)
        ).all()

        if not experiments:
            raise HTTPException(status_code=404, detail="未找到指定的实验")

        # 删除实验（级联删除任务和结果）
        deleted_count = len(experiments)
        for exp in experiments:
            db.delete(exp)

        db.commit()
        return {
            "success": True,
            "message": f"成功删除 {deleted_count} 个实验",
            "deleted_count": deleted_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"批量删除实验失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/api/evaluation/experiments/export")
async def export_experiments(experiment_ids: str = "", db: Session = Depends(get_db)):
    """导出实验配置为 JSON"""
    try:
        # 解析实验 ID 列表
        if experiment_ids:
            ids = [int(id_str) for id_str in experiment_ids.split(",")]
        else:
            ids = []

        # 查询实验
        query = db.query(Experiment)
        if ids:
            experiments = query.filter(Experiment.id.in_(ids)).all()
        else:
            experiments = query.all()

        if not experiments:
            raise HTTPException(status_code=404, detail="未找到指定的实验")

        # 构建导出数据
        export_data = {
            "version": "1.0",
            "export_time": datetime.utcnow().isoformat(),
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
            ],
        }

        return export_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出实验失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.post("/api/evaluation/experiments/check-import")
async def check_import_experiments(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """检查导入文件的有效性"""
    temp_file_id = str(uuid.uuid4())[:8]

    try:
        # 读取文件内容
        content = await file.read()
        import_data = json.loads(content.decode("utf-8"))

        # 验证格式
        if not isinstance(import_data, dict) or "experiments" not in import_data:
            return {
                "valid": False,
                "error": "无效的导入文件格式，缺少 'experiments' 字段",
            }

        # 检查现有实验的名称冲突
        existing_names = {
            exp.name: exp.id
            for exp in db.query(Experiment).all()
        }

        experiments_info = []
        for exp in import_data.get("experiments", []):
            exp_info = {
                "id": exp.get("id"),
                "name": exp.get("name"),
                "description": exp.get("description"),
                "total_tasks": exp.get("total_tasks", 0),
                "completed_tasks": exp.get("completed_tasks", 0),
                "conflict": exp.get("name") in existing_names,
                "existing_id": existing_names.get(exp.get("name")),
            }
            experiments_info.append(exp_info)

        # 保存临时文件
        _import_temp_files[temp_file_id] = {
            "data": import_data,
            "created_at": datetime.utcnow(),
        }

        return {
            "valid": True,
            "experiments": experiments_info,
            "temp_file_id": temp_file_id,
        }
    except json.JSONDecodeError:
        return {
            "valid": False,
            "error": "无法解析 JSON 文件",
        }
    except Exception as e:
        logger.error(f"检查导入文件失败: {e}", exc_info=True)
        return {
            "valid": False,
            "error": f"检查失败: {str(e)}",
        }


@app.post("/api/evaluation/experiments/execute-import")
async def execute_import_experiments(
    request: ImportExecuteRequest,
    db: Session = Depends(get_db)
):
    """执行导入操作"""
    try:
        # 获取临时文件数据
        if request.temp_file_id not in _import_temp_files:
            raise HTTPException(status_code=404, detail="导入会话已过期，请重新上传文件")

        import_data = _import_temp_files[request.temp_file_id]["data"]

        # 处理导入配置
        imported_count = 0
        skipped_count = 0
        overwritten_count = 0
        error_messages = []

        # 建立原始 ID 到新 ID 的映射
        id_mapping = {}

        for config in request.configs:
            try:
                original_name = config.original_name
                action = config.action

                # 从导入数据中找到对应的实验
                import_exp = None
                for exp in import_data.get("experiments", []):
                    if exp.get("name") == original_name:
                        import_exp = exp
                        break

                if not import_exp:
                    skipped_count += 1
                    continue

                if action == "skip":
                    skipped_count += 1
                    continue

                elif action == "overwrite":
                    # 删除现有实验
                    existing_exp = db.query(Experiment).filter(
                        Experiment.name == original_name
                    ).first()
                    if existing_exp:
                        db.delete(existing_exp)
                        overwritten_count += 1

                    # 创建新实验
                    new_exp = Experiment(
                        name=original_name,
                        description=import_exp.get("description"),
                        total_tasks=import_exp.get("total_tasks", 0),
                        completed_tasks=import_exp.get("completed_tasks", 0),
                    )
                    db.add(new_exp)
                    db.flush()
                    id_mapping[import_exp.get("id")] = new_exp.id
                    imported_count += 1

                elif action == "rename":
                    # 以新名称导入
                    new_name = config.new_name or f"{original_name}_imported"
                    new_exp = Experiment(
                        name=new_name,
                        description=import_exp.get("description"),
                        total_tasks=import_exp.get("total_tasks", 0),
                        completed_tasks=import_exp.get("completed_tasks", 0),
                    )
                    db.add(new_exp)
                    db.flush()
                    id_mapping[import_exp.get("id")] = new_exp.id
                    imported_count += 1

            except Exception as e:
                error_messages.append(f"导入 '{original_name}' 失败: {str(e)}")
                logger.error(f"导入实验失败: {e}", exc_info=True)

        db.commit()

        # 清理临时文件
        del _import_temp_files[request.temp_file_id]

        return {
            "success": True,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "overwritten_count": overwritten_count,
            "message": f"导入完成：{imported_count} 个成功，{skipped_count} 个跳过，{overwritten_count} 个覆盖",
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"执行导入失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


# ============================================
# 实验动态路由（必须在静态路由之后）
# ============================================

@app.get("/api/evaluation/experiments/{experiment_id}")
async def get_experiment_details(experiment_id: int, db: Session = Depends(get_db)):
    """获取实验详情（包含所有评估结果，每个结果作为单独的行）"""
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

        tasks = db.query(EvaluationTask).filter(EvaluationTask.experiment_id == experiment_id).all()

        # 将每个任务的所有结果展开为单独的条目
        tasks_with_results = []
        for task in tasks:
            # 查询该任务的所有结果，按得分降序排列
            results = db.query(EvaluationResult).filter(
                EvaluationResult.task_id == task.id
            ).order_by(EvaluationResult.score.desc()).all()

            if results:
                # 为每个结果创建一个任务条目
                for idx, result in enumerate(results):
                    # 从 full_result 中提取 tpot 和 ttft
                    full_result = result.full_result or {}
                    stats = full_result.get('stats', {})

                    task_dict = {
                        "id": task.id,
                        "task_id": task.task_id,
                        "result_id": result.id,  # 新增：结果 ID
                        "result_rank": idx + 1,  # 新增：结果排名
                        "experiment_id": task.experiment_id,
                        "status": task.status.value,
                        "progress": task.progress,
                        "message": task.message,
                        "error": task.error,
                        "created_at": result.created_at.isoformat() if result.created_at else (task.created_at.isoformat() if task.created_at else None),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "config_snapshot": task.config_snapshot,
                        "benchmark_name": task.benchmark_name,
                        "topology_config_name": task.topology_config_name,
                        "search_mode": task.search_mode,
                        "manual_parallelism": task.manual_parallelism,
                        "search_constraints": task.search_constraints,
                        "search_stats": task.search_stats,
                        "result": {
                            'tps': result.tps,
                            'tps_per_chip': result.tps_per_chip,
                            'tps_per_batch': result.tps_per_batch,
                            'tpot': result.tpot,
                            'ttft': result.ttft,
                            'mfu': result.mfu,
                            'mbu': full_result.get('mbu', 0),  # 从 full_result 获取
                            'score': result.score,
                            'chips': result.chips,
                            'dram_occupy': result.dram_occupy,
                            'flops': result.flops,
                            'cost': full_result.get('cost'),  # 成本分析结果
                            'parallelism': {
                                'dp': result.dp,
                                'tp': result.tp,
                                'pp': result.pp,
                                'ep': result.ep,
                                'sp': result.sp,
                                'moe_tp': result.moe_tp,
                            }
                        }
                    }
                    tasks_with_results.append(task_dict)
            # 注意：不再显示没有结果的已完成任务，避免空行

        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "total_tasks": experiment.total_tasks,
            "completed_tasks": experiment.completed_tasks,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "updated_at": experiment.updated_at.isoformat() if experiment.updated_at else None,
            "tasks": tasks_with_results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实验详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@app.patch("/api/evaluation/experiments/{experiment_id}")
async def update_experiment(
    experiment_id: int,
    request: ExperimentUpdateRequest,
    db: Session = Depends(get_db)
):
    """更新实验信息（支持 inline 编辑）"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    try:
        if request.name is not None:
            experiment.name = request.name
        if request.description is not None:
            experiment.description = request.description

        db.commit()
        db.refresh(experiment)

        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "updated_at": experiment.updated_at.isoformat() if experiment.updated_at else None,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"更新实验失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


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


@app.post("/api/evaluation/experiments/{experiment_id}/results/batch-delete")
async def batch_delete_results(
    experiment_id: int,
    request: BatchDeleteResultsRequest,
    db: Session = Depends(get_db)
):
    """批量删除评估结果"""
    try:
        # 验证实验存在
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

        if not request.result_ids:
            raise HTTPException(status_code=400, detail="结果 ID 列表不能为空")

        # 查询要删除的结果
        results = db.query(EvaluationResult).filter(
            EvaluationResult.id.in_(request.result_ids)
        ).all()

        if not results:
            raise HTTPException(status_code=404, detail="未找到指定的结果")

        # 删除结果
        deleted_count = len(results)
        for result in results:
            db.delete(result)

        db.commit()

        return {
            "success": True,
            "message": f"成功删除 {deleted_count} 个结果",
            "deleted_count": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"批量删除结果失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================
# WebSocket 端点
# ============================================

@app.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 端点：实时推送任务状态更新（队列订阅模式）

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
    await websocket.accept()

    # 订阅全局任务更新队列（使用 task_manager 的订阅功能）
    queue = task_manager.subscribe_global()

    try:
        # 持续从队列获取更新并推送给客户端
        while True:
            try:
                # 等待队列中的更新，超时后发送心跳
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(message)
                logger.info(f"[WS] Sent message to client: task_id={message.get('task_id')}, progress={message.get('progress')}")
            except asyncio.TimeoutError:
                # 发送心跳保持连接
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        task_manager.unsubscribe_global(queue)


if __name__ == "__main__":
    import os
    import uvicorn
    from pathlib import Path
    from dotenv import load_dotenv

    # 加载项目根目录 .env 配置
    # web/api.py -> web -> llm_simulator -> backend -> Tier6-Model
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    port = int(os.environ.get("VITE_API_PORT", "8001"))
    print(f"Tier6+互联建模平台启动在端口: {port}")
    uvicorn.run("llm_simulator.web.api:app", host="0.0.0.0", port=port, reload=True)
