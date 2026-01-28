"""
列配置方案管理 API

提供列显示配置的持久化存储和管理功能。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/column-presets", tags=["column-presets"])

# 配置文件存储路径
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config" / "column_presets"
PRESETS_FILE = CONFIG_DIR / "presets.json"


# ============================================
# 数据模型
# ============================================

class ColumnPreset(BaseModel):
    """列配置方案"""
    name: str = Field(..., description="配置方案名称")
    experiment_id: int = Field(..., description="实验ID")
    visible_columns: List[str] = Field(..., description="显示的列")
    column_order: List[str] = Field(..., description="列顺序")
    fixed_columns: List[str] = Field(..., description="固定的列")
    created_at: str = Field(..., description="创建时间 ISO8601")


class PresetsFile(BaseModel):
    """配置文件结构"""
    version: int = Field(default=1, description="文件版本")
    presets: List[ColumnPreset] = Field(default_factory=list, description="配置方案列表")


# ============================================
# 工具函数
# ============================================

def ensure_config_dir():
    """确保配置目录存在"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_presets() -> PresetsFile:
    """从文件加载配置方案"""
    ensure_config_dir()

    if not PRESETS_FILE.exists():
        return PresetsFile(version=1, presets=[])

    try:
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return PresetsFile(**data)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        # 备份损坏的文件
        if PRESETS_FILE.exists():
            backup_file = PRESETS_FILE.with_suffix('.json.bak')
            PRESETS_FILE.rename(backup_file)
            logger.info(f"已备份损坏的配置文件至: {backup_file}")
        return PresetsFile(version=1, presets=[])


def save_presets(presets_file: PresetsFile):
    """保存配置方案到文件"""
    ensure_config_dir()

    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets_file.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"配置文件已保存，共 {len(presets_file.presets)} 个方案")
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


# ============================================
# API 端点
# ============================================

@router.get("/")
async def get_presets():
    """获取所有配置方案"""
    try:
        presets_file = load_presets()
        return presets_file.model_dump()
    except Exception as e:
        logger.error(f"获取配置方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.post("/")
async def save_all_presets(presets_file: PresetsFile):
    """保存所有配置方案（完全覆盖）"""
    try:
        save_presets(presets_file)
        return {"message": "配置已保存", "count": len(presets_file.presets)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存配置方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@router.post("/add")
async def add_preset(preset: ColumnPreset):
    """添加或更新单个配置方案"""
    try:
        presets_file = load_presets()

        # 检查是否存在同名同实验ID的配置
        existing_idx = None
        for idx, p in enumerate(presets_file.presets):
            if p.name == preset.name and p.experiment_id == preset.experiment_id:
                existing_idx = idx
                break

        if existing_idx is not None:
            # 更新现有配置
            presets_file.presets[existing_idx] = preset
            message = f"配置方案「{preset.name}」已更新"
        else:
            # 添加新配置
            presets_file.presets.append(preset)
            message = f"配置方案「{preset.name}」已保存"

        save_presets(presets_file)
        return {"message": message, "preset": preset.model_dump()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加配置方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@router.delete("/{experiment_id}/{name}")
async def delete_preset(experiment_id: int, name: str):
    """删除指定配置方案"""
    try:
        presets_file = load_presets()

        # 查找并删除
        original_count = len(presets_file.presets)
        presets_file.presets = [
            p for p in presets_file.presets
            if not (p.name == name and p.experiment_id == experiment_id)
        ]

        if len(presets_file.presets) == original_count:
            raise HTTPException(status_code=404, detail=f"配置方案「{name}」不存在")

        save_presets(presets_file)
        return {"message": f"配置方案「{name}」已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除配置方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/{experiment_id}")
async def get_presets_by_experiment(experiment_id: int):
    """获取指定实验的所有配置方案"""
    try:
        presets_file = load_presets()
        experiment_presets = [
            p for p in presets_file.presets
            if p.experiment_id == experiment_id
        ]
        return {"presets": [p.model_dump() for p in experiment_presets]}
    except Exception as e:
        logger.error(f"获取实验配置方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")
