"""
预设配置管理

提供芯片预设的加载和管理功能
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_chip_preset(preset_id: str) -> Dict[str, Any]:
    """
    从预设文件加载芯片配置

    Args:
        preset_id: 芯片预设ID (例如: 'sg2262', 'h100')

    Returns:
        芯片配置字典

    Raises:
        FileNotFoundError: 预设文件不存在
        ValueError: 预设文件格式错误
    """
    # 获取预设文件路径
    preset_file = Path(__file__).parent.parent / "configs" / "chip_presets" / f"{preset_id}.yaml"

    if not preset_file.exists():
        raise FileNotFoundError(f"芯片预设文件不存在: {preset_file}")

    # 加载YAML文件
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"芯片预设文件YAML解析失败: {preset_file}, 错误: {e}")
    except Exception as e:
        raise ValueError(f"芯片预设文件读取失败: {preset_file}, 错误: {e}")

    # 验证数据格式
    if not isinstance(preset_data, dict):
        raise ValueError(f"芯片预设文件格式错误 (期望dict): {preset_file}")

    logger.info(f"成功加载芯片预设: {preset_id}")
    return preset_data
