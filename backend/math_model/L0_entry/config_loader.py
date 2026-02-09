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
        """加载拓扑配置

        格式: grouped_pods (pods[].racks[].boards[].chips[])
        芯片参数以快照形式存储在 chips 字段中。
        如果 YAML 缺少 chips 字段（旧文件兼容），从 chips/ 目录加载。

        Args:
            name: 拓扑名称 (不含扩展名)

        Returns:
            拓扑配置
        """
        from .topology_format import extract_chip_names

        filepath = self.topologies_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Topology config not found: {name} (path: {filepath})")

        config = self.load_yaml(filepath)

        # 兼容: 如果 YAML 中缺少 chips 字段，从 chips/ 目录加载
        if "chips" not in config or not config["chips"]:
            chip_names = extract_chip_names(config)
            if chip_names:
                config["chips"] = {}
                for chip_name in chip_names:
                    if chip_name not in config["chips"]:
                        try:
                            chip_config = self.load_chip(chip_name)
                            config["chips"][chip_name] = chip_config
                        except FileNotFoundError:
                            raise FileNotFoundError(
                                f"Chip '{chip_name}' referenced in topology '{name}' "
                                f"not found in chips directory: {self.chips_dir}"
                            )

        return config

    def save_topology(self, name: str, config: dict[str, Any]) -> None:
        """保存拓扑配置

        芯片参数以快照形式持久化在 chips 字段中。
        """
        self.topologies_dir.mkdir(parents=True, exist_ok=True)

        import copy
        save_config = copy.deepcopy(config)

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
                    "topology": config.get("topology"),
                    "format": "yaml",
                    "filename": name,
                })
            except Exception:
                pass

        return benchmarks

    def load_benchmark(self, name: str) -> dict[str, Any]:
        """加载 Benchmark 配置，自动解析模型和拓扑引用

        如果 model/topology 字段是字符串，从预设加载完整配置。

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            Benchmark 配置 (含解析后的模型和拓扑参数)
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

        # 解析拓扑预设引用
        topology_ref = config.get("topology")
        if isinstance(topology_ref, str):
            try:
                topology_config = self.load_topology(topology_ref)
                config["topology"] = topology_config
                config["topology_preset_ref"] = topology_ref
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Topology '{topology_ref}' referenced in benchmark '{name}' "
                    f"not found in topologies directory: {self.topologies_dir}"
                )

        return config

    def save_benchmark(self, name: str, config: dict[str, Any]) -> None:
        """保存 Benchmark 配置

        如果 config 中有 model_preset_ref/topology_preset_ref，
        保存时将 model/topology 恢复为字符串引用。
        输出键序: id, name, model, topology, inference
        """
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        import copy
        save_config = copy.deepcopy(config)

        # 如果有模型引用，保存时恢复为字符串
        model_ref = save_config.pop("model_preset_ref", None)
        if model_ref:
            save_config["model"] = model_ref

        # 如果有拓扑引用，保存时恢复为字符串
        topology_ref = save_config.pop("topology_preset_ref", None)
        if topology_ref:
            save_config["topology"] = topology_ref

        # 按固定键序输出: id, name, model, topology, inference, 其余
        key_order = ["id", "name", "model", "topology", "inference"]
        ordered: dict[str, Any] = {}
        for k in key_order:
            if k in save_config:
                ordered[k] = save_config.pop(k)
        ordered.update(save_config)

        filepath = self.benchmarks_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(ordered, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_benchmark(self, name: str) -> bool:
        """删除 Benchmark"""
        filepath = self.benchmarks_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            return True
        return False



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



