"""
全局配置管理

管理任务执行器、数据库等的配置参数
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent / "data"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.json"

# 默认配置
DEFAULT_CONFIG = {
    "max_global_workers": 8,  # 全局资源池最大 worker 数量
}


class ConfigManager:
    """配置管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config = self._load_config()
        self._initialized = True

    def _load_config(self) -> dict:
        """从文件加载配置"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    logger.info(f"配置已加载: {config}")
                    return {**DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}, 使用默认配置")
                return DEFAULT_CONFIG.copy()
        else:
            logger.info("配置文件不存在，使用默认配置")
            return DEFAULT_CONFIG.copy()

    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存: {self._config}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def get(self, key: str, default=None):
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key: str, value):
        """设置配置项并保存"""
        self._config[key] = value
        self._save_config()

    def get_all(self) -> dict:
        """获取所有配置"""
        return self._config.copy()

    def update(self, config: dict):
        """批量更新配置"""
        self._config.update(config)
        self._save_config()


# 全局配置实例
config_manager = ConfigManager()


def get_max_global_workers() -> int:
    """获取全局资源池最大 worker 数量"""
    return config_manager.get("max_global_workers", DEFAULT_CONFIG["max_global_workers"])


def set_max_global_workers(max_workers: int):
    """设置全局资源池最大 worker 数量（重启服务后生效）"""
    if max_workers < 1 or max_workers > 32:
        raise ValueError("max_global_workers 必须在 1-32 之间")
    config_manager.set("max_global_workers", max_workers)
    logger.info(f"全局资源池最大 worker 数量已设置为 {max_workers}（重启服务后生效）")
