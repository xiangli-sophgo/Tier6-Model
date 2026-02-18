"""Softmax operator implementation."""

from __future__ import annotations

from perf_model.L0_entry.types import DataType
from perf_model.L1_workload.operators import op_registry
from perf_model.L1_workload.operators.base import OpBase, OpRole
from perf_model.L1_workload.operators.utils import get_first_int
from perf_model.L1_workload.tensor import TensorDesc


@op_registry.register("softmax")
class SoftmaxOp(OpBase):
    """Softmax operator."""

    def __init__(self, name: str, parallel_params: dict[str, int], *, dtype: DataType = DataType.FP16):
        super().__init__(name, dtype=dtype)
        params = parallel_params or {}
        self._b = get_first_int(params, ("B", "b", "batch"), 1)
        self._s = get_first_int(params, ("S", "s", "seq_len"), 1)
        self._d = get_first_int(params, ("D", "d", "dim"), 1)

    @property
    def op_type(self) -> str:
        return "softmax"

    @property
    def role(self) -> OpRole:
        return OpRole.VECTOR

    def get_inputs(self) -> list[TensorDesc]:
        return [TensorDesc(name="input", shape=[self._b, self._s, self._d], dtype=self._dtype)]

    def get_outputs(self) -> list[TensorDesc]:
        return [TensorDesc(name="output", shape=[self._b, self._s, self._d], dtype=self._dtype)]

    def compute_ops(self) -> int:
        return self._b * self._s * (3 * self._d - 1)

    def compute_memory_access(self) -> tuple[int, int]:
        read_bytes = sum(t.bytes for t in self.get_inputs())
        write_bytes = sum(t.bytes for t in self.get_outputs())
        return read_bytes, write_bytes
