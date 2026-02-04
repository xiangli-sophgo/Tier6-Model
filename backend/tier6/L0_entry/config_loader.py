"""配置加载模块

加载 CHIPMathica 格式的芯片、模型、板卡、场景等预设配置。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """配置加载器

    从 YAML/JSON 文件加载 CHIPMathica 格式的配置预设。

    目录结构:
        configs/
        ├── chips/           # 芯片配置 (sg2262.yaml, sg2260e.yaml)
        ├── models/          # 模型配置 (deepseek_v3.yaml, llama2_7b.yaml)
        ├── boards/          # 板卡配置 (sg2260e_8chip.yaml)
        ├── scenarios/       # 场景配置 (deepseek_v3_sg2262.yaml)
        └── evaluation/      # 评估配置 (default.yaml)
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """初始化

        Args:
            config_dir: 配置目录路径，默认为 tier6/configs
        """
        if config_dir is None:
            # 默认配置目录: tier6/configs
            self.config_dir = Path(__file__).parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        # CHIPMathica 格式的目录结构
        self.chips_dir = self.config_dir / "chips"
        self.models_dir = self.config_dir / "models"
        self.boards_dir = self.config_dir / "boards"
        self.scenarios_dir = self.config_dir / "scenarios"
        self.evaluation_dir = self.config_dir / "evaluation"

        # 兼容旧格式的目录 (可选)
        self.chip_presets_dir = self.config_dir / "chip_presets"
        self.model_presets_dir = self.config_dir / "model_presets"
        self.topologies_dir = self.config_dir / "topologies"
        self.benchmarks_dir = self.config_dir / "benchmarks"

    def load_yaml(self, filepath: str | Path) -> dict[str, Any]:
        """加载 YAML 文件

        Args:
            filepath: 文件路径

        Returns:
            dict: 配置字典
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_json(self, filepath: str | Path) -> dict[str, Any]:
        """加载 JSON 文件

        Args:
            filepath: 文件路径

        Returns:
            dict: 配置字典
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================== CHIPMathica 格式 API ====================

    def list_chips(self) -> list[str]:
        """列出所有芯片配置

        Returns:
            list[str]: 芯片名称列表
        """
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
            dict: 芯片配置 (CHIPMathica 格式，包含 chip: 根键)
        """
        filepath = self.chips_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Chip config not found: {name}")
        return self.load_yaml(filepath)

    def list_models(self) -> list[str]:
        """列出所有模型配置

        Returns:
            list[str]: 模型名称列表
        """
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
            dict: 模型配置 (CHIPMathica 格式，包含 model: 根键)
        """
        filepath = self.models_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Model config not found: {name}")
        return self.load_yaml(filepath)

    def list_boards(self) -> list[str]:
        """列出所有板卡配置

        Returns:
            list[str]: 板卡名称列表
        """
        if not self.boards_dir.exists():
            return []
        return [
            f.stem
            for f in self.boards_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def load_board(self, name: str) -> dict[str, Any]:
        """加载板卡配置

        Args:
            name: 板卡名称 (不含扩展名)

        Returns:
            dict: 板卡配置 (CHIPMathica 格式，包含 board: 根键)
        """
        filepath = self.boards_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Board config not found: {name}")
        return self.load_yaml(filepath)

    def list_scenarios(self) -> list[str]:
        """列出所有场景配置

        Returns:
            list[str]: 场景名称列表
        """
        if not self.scenarios_dir.exists():
            return []
        return [
            f.stem
            for f in self.scenarios_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def load_scenario(self, name: str) -> dict[str, Any]:
        """加载场景配置

        场景配置包含对芯片和模型的引用，以及部署参数。

        Args:
            name: 场景名称 (不含扩展名)

        Returns:
            dict: 场景配置
        """
        filepath = self.scenarios_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Scenario config not found: {name}")
        return self.load_yaml(filepath)

    def load_scenario_resolved(self, name: str) -> dict[str, Any]:
        """加载场景配置并解析引用

        自动解析场景中对芯片和模型的引用，返回完整配置。

        Args:
            name: 场景名称 (不含扩展名)

        Returns:
            dict: 解析后的完整配置
        """
        scenario = self.load_scenario(name)

        # 解析芯片引用
        chip_name = scenario.get("chip")
        if chip_name:
            chip_config = self.load_chip(chip_name)
            scenario["chip_config"] = chip_config.get("chip", chip_config)

        # 解析模型引用
        model_name = scenario.get("model")
        if model_name:
            model_config = self.load_model(model_name)
            scenario["model_config"] = model_config.get("model", model_config)

        return scenario

    def load_evaluation_config(self, name: str = "default") -> dict[str, Any]:
        """加载评估配置

        Args:
            name: 评估配置名称 (不含扩展名)

        Returns:
            dict: 评估配置
        """
        filepath = self.evaluation_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Evaluation config not found: {name}")
        return self.load_yaml(filepath)

    # ==================== 兼容旧格式 API ====================

    def list_chip_presets(self) -> list[str]:
        """列出所有芯片预设 (兼容旧格式)

        Returns:
            list[str]: 预设名称列表
        """
        # 优先使用新格式
        if self.chips_dir.exists():
            return self.list_chips()
        # 回退到旧格式
        if not self.chip_presets_dir.exists():
            return []
        return [
            f.stem
            for f in self.chip_presets_dir.glob("*.yaml")
        ]

    def load_chip_preset(self, name: str) -> dict[str, Any]:
        """加载芯片预设 (兼容旧格式)

        Args:
            name: 预设名称 (不含扩展名)

        Returns:
            dict: 芯片配置
        """
        # 优先使用新格式
        if self.chips_dir.exists():
            try:
                config = self.load_chip(name)
                # 返回 chip 内部内容，保持接口兼容
                return config.get("chip", config)
            except FileNotFoundError:
                pass

        # 回退到旧格式
        filepath = self.chip_presets_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Chip preset not found: {name}")
        return self.load_yaml(filepath)

    def save_chip_preset(self, name: str, config: dict[str, Any]) -> None:
        """保存芯片预设到 YAML 文件 (Tier6 格式)

        Args:
            name: 预设名称 (不含扩展名)
            config: 芯片配置 (Tier6 ChipPreset 格式)
        """
        # 优先保存到新格式目录
        if self.chips_dir.exists():
            filepath = self.chips_dir / f"{name}.yaml"
        else:
            # 确保目录存在
            self.chip_presets_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.chip_presets_dir / f"{name}.yaml"

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def delete_chip_preset(self, name: str) -> None:
        """删除芯片预设

        Args:
            name: 预设名称 (不含扩展名)
        """
        # 优先在新格式目录查找
        filepath = self.chips_dir / f"{name}.yaml"
        if not filepath.exists():
            filepath = self.chip_presets_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Chip preset not found: {name}")
        filepath.unlink()

    def list_model_presets(self) -> list[str]:
        """列出所有模型预设 (兼容旧格式)

        Returns:
            list[str]: 预设名称列表
        """
        # 优先使用新格式
        if self.models_dir.exists():
            return self.list_models()
        # 回退到旧格式
        if not self.model_presets_dir.exists():
            return []
        return [
            f.stem
            for f in self.model_presets_dir.glob("*.yaml")
        ]

    def load_model_preset(self, name: str) -> dict[str, Any]:
        """加载模型预设 (兼容旧格式)

        Args:
            name: 预设名称 (不含扩展名)

        Returns:
            dict: 模型配置
        """
        # 优先使用新格式
        if self.models_dir.exists():
            try:
                config = self.load_model(name)
                # 返回 model 内部内容，保持接口兼容
                return config.get("model", config)
            except FileNotFoundError:
                pass

        # 回退到旧格式
        filepath = self.model_presets_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Model preset not found: {name}")
        return self.load_yaml(filepath)

    def list_topologies(self) -> list[str]:
        """列出所有拓扑预设

        Returns:
            list[str]: 预设名称列表
        """
        if not self.topologies_dir.exists():
            return []
        return [
            f.stem
            for f in self.topologies_dir.glob("*.yaml")
        ]

    def load_topology(self, name: str) -> dict[str, Any]:
        """加载拓扑预设

        Args:
            name: 预设名称 (不含扩展名)

        Returns:
            dict: 拓扑配置
        """
        filepath = self.topologies_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Topology preset not found: {name}")
        return self.load_yaml(filepath)

    def list_benchmarks(self) -> list[str]:
        """列出所有 Benchmark (JSON 格式)

        Returns:
            list[str]: Benchmark 名称列表
        """
        if not self.benchmarks_dir.exists():
            return []
        return [
            f.stem
            for f in self.benchmarks_dir.glob("*.json")
        ]

    def load_benchmark(self, name: str) -> dict[str, Any]:
        """加载 Benchmark (JSON 格式)

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            dict: Benchmark 配置
        """
        filepath = self.benchmarks_dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark not found: {name}")
        return self.load_json(filepath)

    def list_benchmarks_yaml(self) -> list[str]:
        """列出所有 Benchmark (YAML 格式)

        Returns:
            list[str]: Benchmark 名称列表
        """
        if not self.benchmarks_dir.exists():
            return []
        return [
            f.stem
            for f in self.benchmarks_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]

    def list_all_benchmarks(self) -> list[dict[str, Any]]:
        """列出所有 Benchmark (包含 JSON 和 YAML 格式)

        Returns:
            list[dict]: Benchmark 信息列表，包含 id, name, format 字段
        """
        benchmarks: list[dict[str, Any]] = []

        # 加载 JSON 格式
        for name in self.list_benchmarks():
            try:
                config = self.load_benchmark(name)
                benchmarks.append({
                    "id": config.get("id", name),
                    "name": config.get("name", name),
                    "format": "json",
                    "filename": name,
                })
            except Exception:
                pass

        # 加载 YAML 格式
        for name in self.list_benchmarks_yaml():
            try:
                config = self.load_benchmark_yaml(name)
                benchmarks.append({
                    "id": config.get("id", name),
                    "name": config.get("name", name),
                    "format": "yaml",
                    "filename": name,
                })
            except Exception:
                pass

        return benchmarks

    def load_benchmark_yaml(self, name: str) -> dict[str, Any]:
        """加载 Benchmark (YAML 格式)

        支持模型预设引用解析。如果 model 字段是字符串，则从模型预设加载。

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            dict: Benchmark 配置
        """
        filepath = self.benchmarks_dir / f"{name}.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark YAML not found: {name}")

        config = self.load_yaml(filepath)

        # 解析模型预设引用
        model_ref = config.get("model")
        if isinstance(model_ref, str):
            # 模型是引用，从预设加载
            try:
                model_config = self.load_model_preset(model_ref)
                config["model"] = model_config
                config["model_preset_ref"] = model_ref
            except FileNotFoundError:
                # 保持原样，允许后续处理
                pass

        return config

    def load_benchmark_auto(self, name: str) -> dict[str, Any]:
        """自动加载 Benchmark (优先 YAML，回退 JSON)

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            dict: Benchmark 配置
        """
        # 优先尝试 YAML
        yaml_path = self.benchmarks_dir / f"{name}.yaml"
        if yaml_path.exists():
            return self.load_benchmark_yaml(name)

        # 回退到 JSON
        json_path = self.benchmarks_dir / f"{name}.json"
        if json_path.exists():
            return self.load_benchmark(name)

        raise FileNotFoundError(f"Benchmark not found: {name}")

    def save_benchmark_yaml(self, name: str, config: dict[str, Any]) -> None:
        """保存 Benchmark 为 YAML 格式

        Args:
            name: Benchmark 名称 (不含扩展名)
            config: Benchmark 配置
        """
        if not self.benchmarks_dir.exists():
            self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.benchmarks_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_benchmark(self, name: str) -> bool:
        """删除 Benchmark 文件

        Args:
            name: Benchmark 名称 (不含扩展名)

        Returns:
            bool: 是否成功删除
        """
        # 尝试删除 YAML
        yaml_path = self.benchmarks_dir / f"{name}.yaml"
        if yaml_path.exists():
            yaml_path.unlink()
            return True

        # 尝试删除 JSON
        json_path = self.benchmarks_dir / f"{name}.json"
        if json_path.exists():
            json_path.unlink()
            return True

        return False

    # ==================== 拓扑配置 CRUD ====================

    def save_topology(self, name: str, config: dict[str, Any]) -> None:
        """保存拓扑配置为 YAML 格式

        Args:
            name: 拓扑名称 (不含扩展名)
            config: 拓扑配置
        """
        if not self.topologies_dir.exists():
            self.topologies_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.topologies_dir / f"{name}.yaml"
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_topology(self, name: str) -> bool:
        """删除拓扑配置文件

        Args:
            name: 拓扑名称 (不含扩展名)

        Returns:
            bool: 是否成功删除
        """
        filepath = self.topologies_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            return True
        return False


# 全局配置加载器实例
_config_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器

    Returns:
        ConfigLoader: 配置加载器实例
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def reset_config_loader() -> None:
    """重置全局配置加载器

    用于测试或切换配置目录时清除缓存。
    """
    global _config_loader
    _config_loader = None


# ==================== 便捷函数 (CHIPMathica 格式) ====================


def load_chip(name: str) -> dict[str, Any]:
    """加载芯片配置 (便捷函数)

    Args:
        name: 芯片名称

    Returns:
        dict: 芯片配置
    """
    return get_config_loader().load_chip(name)


def load_model(name: str) -> dict[str, Any]:
    """加载模型配置 (便捷函数)

    Args:
        name: 模型名称

    Returns:
        dict: 模型配置
    """
    return get_config_loader().load_model(name)


def load_board(name: str) -> dict[str, Any]:
    """加载板卡配置 (便捷函数)

    Args:
        name: 板卡名称

    Returns:
        dict: 板卡配置
    """
    return get_config_loader().load_board(name)


def load_scenario(name: str, resolve: bool = True) -> dict[str, Any]:
    """加载场景配置 (便捷函数)

    Args:
        name: 场景名称
        resolve: 是否解析引用

    Returns:
        dict: 场景配置
    """
    loader = get_config_loader()
    if resolve:
        return loader.load_scenario_resolved(name)
    return loader.load_scenario(name)


def load_evaluation_config(name: str = "default") -> dict[str, Any]:
    """加载评估配置 (便捷函数)

    Args:
        name: 评估配置名称

    Returns:
        dict: 评估配置
    """
    return get_config_loader().load_evaluation_config(name)


# ==================== 便捷函数 (兼容旧格式) ====================


def load_chip_preset(name: str) -> dict[str, Any]:
    """加载芯片预设 (便捷函数，兼容旧格式)

    Args:
        name: 预设名称

    Returns:
        dict: 芯片配置
    """
    return get_config_loader().load_chip_preset(name)


def load_model_preset(name: str) -> dict[str, Any]:
    """加载模型预设 (便捷函数，兼容旧格式)

    Args:
        name: 预设名称

    Returns:
        dict: 模型配置
    """
    return get_config_loader().load_model_preset(name)


def load_topology(name: str) -> dict[str, Any]:
    """加载拓扑预设 (便捷函数)

    Args:
        name: 预设名称

    Returns:
        dict: 拓扑配置
    """
    return get_config_loader().load_topology(name)


def load_benchmark(name: str) -> dict[str, Any]:
    """加载 Benchmark (便捷函数)

    Args:
        name: Benchmark 名称

    Returns:
        dict: Benchmark 配置
    """
    return get_config_loader().load_benchmark(name)


def load_benchmark_yaml(name: str) -> dict[str, Any]:
    """加载 Benchmark YAML (便捷函数)

    Args:
        name: Benchmark 名称

    Returns:
        dict: Benchmark 配置
    """
    return get_config_loader().load_benchmark_yaml(name)


def load_benchmark_auto(name: str) -> dict[str, Any]:
    """自动加载 Benchmark (便捷函数)

    Args:
        name: Benchmark 名称

    Returns:
        dict: Benchmark 配置
    """
    return get_config_loader().load_benchmark_auto(name)
