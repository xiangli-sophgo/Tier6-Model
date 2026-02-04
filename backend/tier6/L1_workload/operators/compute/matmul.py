"""MatMul operator implementation."""

from __future__ import annotations

from tier6.core.types import DataType
from tier6.L1_workload.operators import op_registry
from tier6.L1_workload.operators.base import OpBase, OpRole
from tier6.L1_workload.operators.utils import get_first_int
from tier6.L1_workload.tensor import TensorDesc


@op_registry.register("matmul")
class MatMulOp(OpBase):
    """Matrix multiplication operator."""

    def __init__(
        self,
        name: str,
        parallel_params: dict[str, int],
        *,
        dtype: DataType = DataType.FP16,
        weight_dtype: DataType | None = None,
        output_dtype: DataType | None = None,
        accum_dtype: DataType | None = None,
    ):
        super().__init__(name, dtype=dtype)
        params = parallel_params or {}
        self._g = get_first_int(params, ("G", "g"), 1)
        self._m = get_first_int(params, ("M", "m"), 1)
        self._k = get_first_int(params, ("K", "k"), 1)
        self._n = get_first_int(params, ("N", "n"), 1)
        self._weight_dtype = weight_dtype or dtype
        self._output_dtype = output_dtype or dtype
        self._accum_dtype = accum_dtype or DataType.FP32

    @property
    def op_type(self) -> str:
        return "matmul"

    @property
    def role(self) -> OpRole:
        return OpRole.CUBE

    def get_inputs(self) -> list[TensorDesc]:
        if self._g > 1:
            a_shape = [self._g, self._m, self._k]
            b_shape = [self._g, self._k, self._n]
        else:
            a_shape = [self._m, self._k]
            b_shape = [self._k, self._n]
        return [
            TensorDesc(name="input", shape=a_shape, dtype=self._dtype),
            TensorDesc(
                name="weight", shape=b_shape, dtype=self._weight_dtype, is_weight=True
            ),
        ]

    def get_outputs(self) -> list[TensorDesc]:
        if self._g > 1:
            out_shape = [self._g, self._m, self._n]
        else:
            out_shape = [self._m, self._n]
        return [TensorDesc(name="output", shape=out_shape, dtype=self._output_dtype)]

    def compute_ops(self) -> int:
        return 2 * self._g * self._m * self._k * self._n

    def compute_memory_access(self) -> tuple[int, int]:
        read_bytes = sum(t.bytes for t in self.get_inputs())
        write_bytes = sum(t.bytes for t in self.get_outputs())
        return read_bytes, write_bytes
