"""Dispatch operator implementation."""

from __future__ import annotations

from math_model.core.types import DataType
from math_model.L1_workload.operators import op_registry
from math_model.L1_workload.operators.base import OpBase, OpRole
from math_model.L1_workload.operators.utils import bytes_to_elements, get_first_int
from math_model.L1_workload.tensor import TensorDesc


@op_registry.register("dispatch")
class DispatchOp(OpBase):
    def __init__(self, name: str, parallel_params: dict[str, int], *, dtype: DataType = DataType.FP16):
        super().__init__(name, dtype=dtype)
        params = parallel_params or {}
        self._bytes = get_first_int(params, ("comm_size", "bytes", "size"), 0)

    @property
    def op_type(self) -> str:
        return "dispatch"

    @property
    def role(self) -> OpRole:
        return OpRole.COMM

    def get_inputs(self) -> list[TensorDesc]:
        elements = bytes_to_elements(self._bytes, self._dtype.bytes)
        return [TensorDesc(name="payload", shape=[elements], dtype=self._dtype)]

    def get_outputs(self) -> list[TensorDesc]:
        elements = bytes_to_elements(self._bytes, self._dtype.bytes)
        return [TensorDesc(name="dispatched", shape=[elements], dtype=self._dtype)]

    def compute_ops(self) -> int:
        return 0

    def compute_memory_access(self) -> tuple[int, int]:
        return self._bytes, self._bytes
