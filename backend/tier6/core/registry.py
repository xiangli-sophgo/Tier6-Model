"""CHIPMathica 通用注册表

提供可扩展的插件注册机制。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generic, TypeVar

from tier6.core.exceptions import AlreadyRegisteredError, NotRegisteredError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """通用注册表

    支持通过装饰器或显式调用注册组件，并提供工厂方法创建实例。

    Example:
        >>> model_registry = Registry[BaseModel]("model")
        >>>
        >>> @model_registry.register("llama2")
        ... class Llama2Model(BaseModel):
        ...     pass
        >>>
        >>> model = model_registry.create("llama2", config=config)
    """

    def __init__(self, name: str):
        """初始化注册表

        Args:
            name: 注册表名称，用于错误信息
        """
        self._name = name
        self._registry: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}  # 别名映射

    @property
    def name(self) -> str:
        """注册表名称"""
        return self._name

    def register(
        self,
        name: str,
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """注册装饰器

        Args:
            name: 注册名称（不区分大小写）
            aliases: 可选的别名列表
            override: 是否允许覆盖已有注册

        Returns:
            装饰器函数

        Example:
            >>> @registry.register("my_component", aliases=["mc", "mycomp"])
            ... class MyComponent:
            ...     pass
        """

        def decorator(cls: type[T]) -> type[T]:
            self.register_class(name, cls, aliases=aliases, override=override)
            return cls

        return decorator

    def register_class(
        self,
        name: str,
        cls: type[T],
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> None:
        """显式注册类

        Args:
            name: 注册名称（不区分大小写）
            cls: 要注册的类
            aliases: 可选的别名列表
            override: 是否允许覆盖已有注册

        Raises:
            AlreadyRegisteredError: 名称已注册且不允许覆盖
        """
        key = name.lower()

        if key in self._registry and not override:
            raise AlreadyRegisteredError(name, self._name)

        self._registry[key] = cls
        logger.debug(f"Registered '{name}' in {self._name} registry")

        # 注册别名
        if aliases:
            for alias in aliases:
                alias_key = alias.lower()
                if alias_key in self._aliases and not override:
                    logger.warning(f"Alias '{alias}' already exists, skipping")
                    continue
                self._aliases[alias_key] = key

    def unregister(self, name: str) -> None:
        """取消注册

        Args:
            name: 要取消注册的名称

        Raises:
            NotRegisteredError: 名称未注册
        """
        key = name.lower()

        if key not in self._registry:
            raise NotRegisteredError(name, self._name, self.list_registered())

        del self._registry[key]

        # 移除相关别名
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == key]
        for alias in aliases_to_remove:
            del self._aliases[alias]

    def get(self, name: str) -> type[T]:
        """获取已注册的类

        Args:
            name: 注册名称或别名（不区分大小写）

        Returns:
            注册的类

        Raises:
            NotRegisteredError: 名称未注册
        """
        key = name.lower()

        # 先检查别名
        if key in self._aliases:
            key = self._aliases[key]

        if key not in self._registry:
            raise NotRegisteredError(name, self._name, self.list_registered())

        return self._registry[key]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """创建实例

        Args:
            name: 注册名称或别名
            *args: 传递给构造函数的位置参数
            **kwargs: 传递给构造函数的关键字参数

        Returns:
            创建的实例

        Raises:
            NotRegisteredError: 名称未注册
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def contains(self, name: str) -> bool:
        """检查名称是否已注册

        Args:
            name: 要检查的名称

        Returns:
            是否已注册
        """
        key = name.lower()
        return key in self._registry or key in self._aliases

    def list_registered(self) -> list[str]:
        """列出所有已注册的名称

        Returns:
            已注册名称列表（按字母排序）
        """
        return sorted(self._registry.keys())

    def list_all(self) -> list[str]:
        """列出所有名称（包括别名）

        Returns:
            所有名称列表（按字母排序）
        """
        all_names = set(self._registry.keys()) | set(self._aliases.keys())
        return sorted(all_names)

    def clear(self) -> None:
        """清空注册表"""
        self._registry.clear()
        self._aliases.clear()

    def __contains__(self, name: str) -> bool:
        """支持 in 操作符"""
        return self.contains(name)

    def __len__(self) -> int:
        """返回注册数量"""
        return len(self._registry)

    def __iter__(self):
        """迭代所有注册名称"""
        return iter(self._registry.keys())

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, count={len(self)})"


class InstanceRegistry(Generic[T]):
    """实例注册表

    用于存储单例实例而非类。适用于硬件规格等需要预创建实例的场景。

    Example:
        >>> chip_registry = InstanceRegistry[ChipSpec]("chip")
        >>> chip_registry.register("sg2260e", chip_instance, aliases=["SG2260E"])
        >>> chip = chip_registry.get("sg2260e")  # 返回相同实例
    """

    def __init__(self, name: str):
        """初始化实例注册表

        Args:
            name: 注册表名称
        """
        self._name = name
        self._instances: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    @property
    def name(self) -> str:
        """注册表名称"""
        return self._name

    def register(
        self,
        name: str,
        instance: T,
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> None:
        """注册实例

        Args:
            name: 注册名称（不区分大小写）
            instance: 要注册的实例
            aliases: 可选的别名列表
            override: 是否允许覆盖已有注册
        """
        key = name.lower()

        if key in self._instances and not override:
            raise AlreadyRegisteredError(name, self._name)

        self._instances[key] = instance
        logger.debug(f"Registered instance '{name}' in {self._name} registry")

        if aliases:
            for alias in aliases:
                alias_key = alias.lower()
                if alias_key in self._aliases and not override:
                    logger.warning(f"Alias '{alias}' already exists, skipping")
                    continue
                self._aliases[alias_key] = key

    def get(self, name: str) -> T:
        """获取已注册的实例

        Args:
            name: 注册名称或别名

        Returns:
            注册的实例

        Raises:
            NotRegisteredError: 名称未注册
        """
        key = name.lower()

        if key in self._aliases:
            key = self._aliases[key]

        if key not in self._instances:
            raise NotRegisteredError(name, self._name, self.list_registered())

        return self._instances[key]

    def contains(self, name: str) -> bool:
        """检查名称是否已注册"""
        key = name.lower()
        return key in self._instances or key in self._aliases

    def list_registered(self) -> list[str]:
        """列出所有已注册的名称"""
        return sorted(self._instances.keys())

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self):
        return iter(self._instances.keys())

    def __repr__(self) -> str:
        return f"InstanceRegistry(name={self._name!r}, count={len(self)})"


class LazyRegistry(Registry[T]):
    """延迟加载注册表

    支持注册工厂函数而非类，在首次创建时才导入和实例化。
    适用于避免循环导入或延迟加载大型依赖。

    Example:
        >>> lazy_registry = LazyRegistry[BaseModel]("lazy_model")
        >>>
        >>> @lazy_registry.register_lazy("large_model")
        ... def load_large_model():
        ...     from heavy_module import LargeModel
        ...     return LargeModel
        >>>
        >>> model = lazy_registry.create("large_model", config=config)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._lazy_factories: dict[str, Callable[[], type[T]]] = {}

    def register_lazy(
        self,
        name: str,
        aliases: list[str] | None = None,
    ) -> Callable[[Callable[[], type[T]]], Callable[[], type[T]]]:
        """注册延迟加载工厂函数

        Args:
            name: 注册名称
            aliases: 可选的别名列表

        Returns:
            装饰器函数
        """

        def decorator(factory: Callable[[], type[T]]) -> Callable[[], type[T]]:
            key = name.lower()
            self._lazy_factories[key] = factory

            # 注册别名
            if aliases:
                for alias in aliases:
                    self._aliases[alias.lower()] = key

            return factory

        return decorator

    def get(self, name: str) -> type[T]:
        """获取类（支持延迟加载）"""
        key = name.lower()

        # 先检查别名
        if key in self._aliases:
            key = self._aliases[key]

        # 检查是否已加载
        if key in self._registry:
            return self._registry[key]

        # 检查延迟工厂
        if key in self._lazy_factories:
            cls = self._lazy_factories[key]()
            self._registry[key] = cls
            del self._lazy_factories[key]
            return cls

        raise NotRegisteredError(name, self._name, self.list_all())

    def list_registered(self) -> list[str]:
        """列出所有已注册的名称（包括未加载的）"""
        all_keys = set(self._registry.keys()) | set(self._lazy_factories.keys())
        return sorted(all_keys)
