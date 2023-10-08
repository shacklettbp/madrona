R"===(#"

from functools import partial

import jax
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.lib import xla_client
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

custom_call_name = f"{type(sim_obj).__name__}_{id(sim_obj)}_step"

xla_client.register_custom_call_target(
    custom_call_name, custom_call_capsule, platform=custom_call_platform)

def _row_major_layout(shape):
    return tuple(range(len(shape) -1, -1, -1))

def _lowering(ctx, inputs):
    flattened_inputs = jax.tree_util.tree_flatten(inputs)

    operand_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
    operand_layouts = [_row_major_layout(t.shape) for t in operand_types]

    aliases = {idx: idx for idx in range(len(flattened_inputs))}

    return custom_call(
        custom_call_name,
        result_types=operand_types,
        operands=flattened_inputs,
        backend_config=sim_encode,
        operand_output_aliases=aliases,
        operand_layouts=operand_layouts,
        result_layouts=operand_layouts,
    )

def _abstract(inputs):
    return tuple(ShapedArray(
        i.shape, i.dtype, named_shape=i.named_shape) for i in inputs)

_primitive = core.Primitive(custom_call_name)
_primitive.multiple_results = True
_primitive.def_impl(partial(xla.apply_primitive, _primitive))
_primitive.def_abstract_eval(_abstract)


mlir.register_lowering(
    _primitive,
    _lowering,
    platform=custom_call_platform,
)

def step_func(*args, **kwargs):
    return _primitive.bind(*args, **kwargs)


#)==="
