"""配置加载模块

加载芯片、模型、拓扑、Benchmark 配置。
四类配置独立管理，通过引用关联。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """配置加载器

    从 YAML 文件加载配置预设。

    目录结构:
        configs/
        ├── chips/           # 芯片配置 (SG2262.yaml)
        ├── models/          # 模型配置 (deepseek_v3.yaml)
        ├── benchmarks/      # Benchmark 配置 (deepseek_v3-S32K-O1K-W16A16-B1.yaml)
        └── topologies/      # 拓扑配置 (P1-R1-B1-C8.yaml)
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        self.chips_dir = self.config_dir / "chips"
        self.models_dir = self.config_dir / "models"
        self.topologies_dir = self.config_dir / "topologies"
        self.benchmarks_dir = self.config_dir / "benchmarks"
        self.evaluation_dir = self.config_dir / "evaluation"

    def load_yaml(self, filepath: str | Path) -> dict[str, Any]:
        """加载 YAML 文件"""
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ==================== 芯片配置 CRUD ====================

    def list_chips(self) -> list[str]:
        """列出所有芯片配置名称"""
        if not self.chips_dir.exists():
            return []
        return [
            f.stem
            for f in self.chips_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def load_chip(self, name: str) -> dict[str, Any]:
        """加载芯片配置

        Args:
            name: 芯片名称 (不含扩展名)

        Returns:
            芯片配置字典
        """
        filepath = self.chips_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Chip config not found: {name} (path: {filepath})")
        return self.load_yaml(filepath)

    def save_chip(self, name: str, config: dict[str, Any]) -> None:
        """保存芯片配置"""
        self.chips_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.chips_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def delete_chip(self, name: str) -> None:
        """删除芯片配置"""
        filepath = self.chips_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Chip config not found: {name}")
        filepath.unlink()

    # ==================== 模型配置 CRUD ====================

    def list_models(self) -> list[str]:
        """列出所有模型配置名称"""
        if not self.models_dir.exists():
            return []
        return [
            f.stem
            for f in self.models_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def load_model(self, name: str) -> dict[str, Any]:
        """加载模型配置

        Args:
            name: 模型名称 (不含扩展名)

        Returns:
            模型配置字典
        """
        filepath = self.models_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Model config not found: {name} (path: {filepath})")
        return self.load_yaml(filepath)

    def save_model(self, name: str, config: dict[str, Any]) -> None:
        """保存模型配置"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.models_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def delete_model(self, name: str) -> None:
        """删除模型配置"""
        filepath = self.models_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Model config not found: {name}")
        filepath.unlink()

    # ==================== 拓扑配置 CRUD ====================

    def list_topologies(self) -> list[str]:
        """列出所有拓扑配置名称"""
        if not self.topologies_dir.exists():
            return []
        return [
            f.stem
            for f in self.topologies_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def load_topology(self, name: str) -> dict[str, Any]:
        """加载拓扑配置，自动解析芯片引用

        遍历 rack_config.boards[].chips[].name，从 chips/ 目录加载对应芯片配置，
        注入到返回结果的 hardware_params.chips 字典中。

        Args:
            name: 拓扑名称 (不含扩展名)

        Returns:
            完整拓扑配置 (含解析后的芯片参数)
        """
        filepath = self.topologies_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Topology config not found: {name} (path: {filepath})")

        config = self.load_yaml(filepath)

        # 解析芯片引用: 从 rack_config.boards[].chips[].name 提取芯片名称
        chip_names = set()
        rack_config = config.get("rack_config", {})
        boards = rack_config.get("boards", [])
        for board in boards:
            for chip_group in board.get("chips", []):
                chip_name = chip_group.get("name")
                if chip_name:
                    chip_names.add(chip_name)

        # 加载芯片配置并注入到 hardware_params.chips
        if chip_names:
            if "hardware_params" not in config:
                config["hardware_params"] = {}
            hw = config["hardware_params"]
            if "chips" not in hw:
                hw["chips"] = {}

            for chip_name in chip_names:
                if chip_name not in hw["chips"]:
                    # 只在 chips 字典中没有时才从文件加载
                    try:
                        chip_config = self.load_chip(chip_name)
                        hw["chips"][chip_name] = chip_config
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            f"Chip '{chip_name}' referenced in topology '{name}' "
                            f"not found in chips directory: {self.chips_dir}"
                        )

        return config

    def save_topology(self, name: str, config: dict[str, Any]) -> None:
        """保存拓扑配置

        保存前移除 hardware_params.chips (解析后的数据不应持久化)
        """
        self.topologies_dir.mkdir(parents=True, exist_ok=True)

        # 深拷贝以避免修改原始数据
        import copy
        save_config = copy.deepcopy(config)

        # 移除解析后的芯片参数 (不应持久化到 YAML)
        hw = save_config.get("hardware_params", {})
        if "chips" in hw:
            del hw["chips"]

        filepath = self.topologies_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(save_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_topology(self, name: str) -> bool:
        """删除拓扑配置"""
        filepath = self.topologies_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # ==================== Benchmark 配置 CRUD ====================

    def list_benchmarks(self) -> list[str]:
        """列出所有 Benchmark 名称 (YAML 格式)"""
        if not self.benchmarks_dir.exists():
            return []
        return [
            f.stem
            for f in self.benchmarks_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def list_all_benchmarks(self) -> list[dict[str, Any]]:
        """列出所有 Benchmark 信息"""
        benchmarks: list[dict[str, Any]] = []

        for name in self.list_benchmarks():
            try:
                config = self.load_yaml(self.benchmarks_dir / f"{name}.yaml")
                benchmarks.append({
                    "id": config.get("id", name),
                    "name": config.get("name", name),
                    "format": "yaml",
                    "filename": name,
                })
            except Exception:
                pass

        return benchmarks

    def load_benchmark(self, name: str) -> dict[str, Any]:
        """加载 Benchmark 配置，自动解析模型引用

        如果 model 字段是字符串，从模型预设加载完整配置。

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            Benchmark 配置 (含解析后的模型参数)
        """
        filepath = self.benchmarks_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark not found: {name} (path: {filepath})")

        config = self.load_yaml(filepath)

        # 解析模型预设引用
        model_ref = config.get("model")
        if isinstance(model_ref, str):
            try:
                model_config = self.load_model(model_ref)
                config["model"] = model_config
                config["model_preset_ref"] = model_ref
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Model '{model_ref}' referenced in benchmark '{name}' "
                    f"not found in models directory: {self.models_dir}"
                )

        return config

    def save_benchmark(self, name: str, config: dict[str, Any]) -> None:
        """保存 Benchmark 配置

        如果 config 中有 model_preset_ref，保存时将 model 恢复为字符串引用。
        """
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        import copy
        save_config = copy.deepcopy(config)

        # 如果有模型引用，保存时恢复为字符串
        model_ref = save_config.pop("model_preset_ref", None)
        if model_ref:
            save_config["model"] = model_ref

        filepath = self.benchmarks_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(save_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_benchmark(self, name: str) -> bool:
        """删除 Benchmark"""
        filepath = self.benchmarks_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    # ==================== 评估配置 ====================

    def load_evaluation_config(self, name: str = "default") -> dict[str, Any]:
        """加载评估配置"""
        filepath = self.evaluation_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Evaluation config not found: {name}")
        return self.load_yaml(filepath)


# 全局配置加载器实例
_config_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def reset_config_loader() -> None:
    """重置全局配置加载器"""
    global _config_loader
    _config_loader = None


# ==================== 便捷函数 ====================


def load_chip(name: str) -> dict[str, Any]:
    """加载芯片配置"""
    return get_config_loader().load_chip(name)


def load_model(name: str) -> dict[str, Any]:
    """加载模型配置"""
    return get_config_loader().load_model(name)


def load_topology(name: str) -> dict[str, Any]:
    """加载拓扑配置 (含芯片引用解析)"""
    return get_config_loader().load_topology(name)


def load_benchmark(name: str) -> dict[str, Any]:
    """加载 Benchmark (含模型引用解析)"""
    return get_config_loader().load_benchmark(name)


def load_evaluation_config(name: str = "default") -> dict[str, Any]:
    """加载评估配置"""
    return get_config_loader().load_evaluation_config(name)

