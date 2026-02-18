"""Tier6 自定义异常

定义项目中使用的异常层级结构。
"""

from __future__ import annotations

from typing import Any


class Tier6Error(Exception):
    """Tier6 基础异常类

    所有 Tier6 异常的基类。
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# ============================================================================
# 配置相关异常
# ============================================================================


class ConfigError(Tier6Error):
    """配置错误

    配置文件加载、解析或校验失败时抛出。
    """

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        field: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs
        if config_path:
            details["config_path"] = config_path
        if field:
            details["field"] = field
        super().__init__(message, details)
        self.config_path = config_path
        self.field = field


class ValidationError(ConfigError):
    """校验错误

    配置或数据校验失败时抛出。
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        **kwargs: Any,
    ):
        details = kwargs
        if value is not None:
            details["value"] = repr(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, field=field, **details)
        self.value = value
        self.expected = expected


class SchemaError(ConfigError):
    """Schema 错误

    配置不符合 Schema 定义时抛出。
    """

    pass


# ============================================================================
# 注册表相关异常
# ============================================================================


class RegistryError(Tier6Error):
    """注册表错误基类"""

    pass


class NotRegisteredError(RegistryError):
    """未注册错误

    请求的组件未在注册表中注册时抛出。
    """

    def __init__(
        self,
        name: str,
        registry_name: str,
        available: list[str] | None = None,
    ):
        message = f"'{name}' is not registered in {registry_name}"
        if available:
            message += f". Available: {', '.join(available)}"
        super().__init__(message, {"name": name, "registry": registry_name})
        self.name = name
        self.registry_name = registry_name
        self.available = available or []


class AlreadyRegisteredError(RegistryError):
    """重复注册错误

    尝试注册已存在的组件时抛出。
    """

    def __init__(self, name: str, registry_name: str):
        message = f"'{name}' is already registered in {registry_name}"
        super().__init__(message, {"name": name, "registry": registry_name})
        self.name = name
        self.registry_name = registry_name


# ============================================================================
# 评估相关异常
# ============================================================================


class EvaluationError(Tier6Error):
    """评估错误

    性能评估过程中出现错误时抛出。
    """

    pass


class EngineError(EvaluationError):
    """引擎错误

    评估引擎内部错误。
    """

    def __init__(self, message: str, engine_type: str | None = None, **kwargs: Any):
        details = kwargs
        if engine_type:
            details["engine"] = engine_type
        super().__init__(message, details)
        self.engine_type = engine_type


class MappingError(EvaluationError):
    """映射错误

    工作负载到硬件的映射失败时抛出。
    """

    pass


# ============================================================================
# 模型相关异常
# ============================================================================


class ModelError(Tier6Error):
    """模型错误基类"""

    pass


class ModelNotFoundError(ModelError):
    """模型未找到

    请求的模型不存在时抛出。
    """

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found", {"model": model_name})
        self.model_name = model_name


class ModelConfigError(ModelError):
    """模型配置错误

    模型配置不正确时抛出。
    """

    pass


# ============================================================================
# 芯片相关异常
# ============================================================================


class ChipError(Tier6Error):
    """芯片错误基类"""

    pass


class ChipNotFoundError(ChipError):
    """芯片未找到

    请求的芯片不存在时抛出。
    """

    def __init__(self, chip_name: str):
        super().__init__(f"Chip '{chip_name}' not found", {"chip": chip_name})
        self.chip_name = chip_name


class ChipConfigError(ChipError):
    """芯片配置错误

    芯片配置不正确时抛出。
    """

    pass


# ============================================================================
# IO 相关异常
# ============================================================================


class IOError(Tier6Error):
    """IO 错误基类"""

    pass


class FileNotFoundError(IOError):
    """文件未找到"""

    def __init__(self, path: str):
        super().__init__(f"File not found: {path}", {"path": path})
        self.path = path


class ExportError(IOError):
    """导出错误

    结果导出失败时抛出。
    """

    def __init__(self, message: str, format: str | None = None, **kwargs: Any):
        details = kwargs
        if format:
            details["format"] = format
        super().__init__(message, details)
        self.format = format
