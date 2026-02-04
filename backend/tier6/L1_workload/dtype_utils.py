"""dtype 解析与量化映射工具."""

from __future__ import annotations

import re
from typing import Any

from tier6.core.types import DataType


def parse_dtype(value: str | DataType | None, default: DataType) -> DataType:
    if value is None:
        return default
    if isinstance(value, DataType):
        return value
    return _from_token(str(value), default=default)


def resolve_layer_dtypes(
    config: dict[str, Any], *, default: DataType = DataType.FP16
) -> dict[str, DataType]:
    base = parse_dtype(config.get("dtype"), default)

    activation = _maybe_parse(config.get("activation_dtype"))
    weight = _maybe_parse(config.get("weight_dtype"))
    output = _maybe_parse(config.get("output_dtype"))
    accum = _maybe_parse(config.get("accum_dtype"))

    if activation is None or weight is None or output is None or accum is None:
        quant = config.get("quantization")
        if isinstance(quant, str) and quant.strip():
            q_map = parse_quantization(quant, base)
            activation = activation or q_map.get("activation")
            weight = weight or q_map.get("weight")
            output = output or q_map.get("output")
            accum = accum or q_map.get("accum")

    activation = activation or base
    weight = weight or base
    output = output or activation
    accum = accum or DataType.FP32

    return {
        "activation": activation,
        "weight": weight,
        "output": output,
        "accum": accum,
    }


def parse_quantization(quant: str, base_dtype: DataType) -> dict[str, DataType]:
    quant_norm = quant.strip().lower().replace("-", "_")
    match = re.search(r"w([a-z0-9_]+)a([a-z0-9_]+)", quant_norm)
    if not match:
        return {}

    w_token = match.group(1)
    a_token = match.group(2)

    weight = _from_quant_token(w_token, base_dtype, role="weight")
    activation = _from_quant_token(a_token, base_dtype, role="activation")
    output = activation
    accum = DataType.FP32

    return {
        "weight": weight,
        "activation": activation,
        "output": output,
        "accum": accum,
    }


def _maybe_parse(value: str | DataType | None) -> DataType | None:
    if value is None:
        return None
    if isinstance(value, DataType):
        return value
    return _from_token(str(value), default=None)


def _from_quant_token(token: str, base: DataType, *, role: str) -> DataType:
    token = token.strip().lower()
    if token in {"8", "int8", "i8"}:
        return DataType.INT8
    if token in {"4", "int4", "i4"}:
        return DataType.INT4
    if token in {"16"}:
        if base in {DataType.BF16, DataType.FP16}:
            return base
        return DataType.FP16
    if token in {"bf16", "bfloat16"}:
        return DataType.BF16
    if token in {"fp16", "f16"}:
        return DataType.FP16
    if token in {"fp32", "f32"}:
        return DataType.FP32
    if token in {"fp8", "e4m3"}:
        return DataType.FP8_E4M3
    if token in {"e5m2"}:
        return DataType.FP8_E5M2
    if token in {"fp4"}:
        # 若未明确 FP4 编码，回退到 INT4
        return DataType.INT4
    return _from_token(token, default=base if role == "activation" else base)


def _from_token(token: str, *, default: DataType | None) -> DataType:
    token = token.strip().lower().replace("-", "_")
    if token in {"fp8"}:
        return DataType.FP8_E4M3
    if token in {"e4m3", "fp8_e4m3"}:
        return DataType.FP8_E4M3
    if token in {"e5m2", "fp8_e5m2"}:
        return DataType.FP8_E5M2
    if token in {"fp16", "f16"}:
        return DataType.FP16
    if token in {"bf16", "bfloat16"}:
        return DataType.BF16
    if token in {"fp32", "f32"}:
        return DataType.FP32
    if token in {"int8", "i8"}:
        return DataType.INT8
    if token in {"int4", "i4"}:
        return DataType.INT4
    if token in {"uint8", "u8"}:
        return DataType.UINT8
    if token in {"uint4", "u4"}:
        return DataType.UINT4
    if default is None:
        raise ValueError(f"Unknown dtype token: {token}")
    return default
