R"===(#"

from functools import partial

import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jax.lib import xla_client
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
import builtins as __builtins__

custom_call_name = f"{type(sim_obj).__name__}_{id(sim_obj)}_step"
del sim_obj

xla_client.register_custom_call_target(
    custom_call_name, custom_call_capsule, platform=custom_call_platform)

def _row_major_layout(shape):
    return tuple(range(len(shape) -1, -1, -1))

def _lowering(ctx, *flattened_data):
    operand_types = [ir.RankedTensorType(i.type) for i in flattened_data]
    operand_layouts = [_row_major_layout(t.shape) for t in operand_types]

    aliases = {idx: idx for idx in range(len(flattened_data))}

    return custom_call(
        custom_call_name,
        result_types=operand_types,
        operands=flattened_data,
        backend_config=sim_encode,
        operand_output_aliases=aliases,
        operand_layouts=operand_layouts,
        result_layouts=operand_layouts,
        has_side_effect=True,
    ).results

def _abstract(*inputs):
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

def step_func(sim_data):
    flattened_in, treedef = jax.tree_util.tree_flatten(sim_data)
    flattened_out = _primitive.bind(*flattened_in)
    return jax.tree_util.tree_unflatten(treedef, flattened_out)

step_func = jax.jit(step_func, donate_argnums=0)

#)==="
