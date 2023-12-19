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
    flattened_in = []

    flattened_in.append(sim_data['actions'])
    flattened_in.append(sim_data['resets'])
    flattened_in.append(sim_data['rewards'])
    flattened_in.append(sim_data['dones'])

    if 'policy_assignments' in sim_iface_shapes:
        flattened_in.append(sim_data['policy_assignments'])

    for k in sim_iface_shapes['obs'].keys():
        flattened_in.append(sim_data['obs'][k])

    for k in sim_iface_shapes['stats'].keys():
        flattened_in.append(sim_data['stats'][k])

    flattened_out = _primitive.bind(*flattened_in)

    out = {}

    cur_idx = 0

    def next_out():
        nonlocal cur_idx
        o = flattened_out[cur_idx]
        cur_idx += 1
        return o

    out['actions'] = next_out()
    out['resets'] = next_out()
    out['rewards'] = next_out()
    out['dones'] = next_out()

    if 'policy_assignments' in sim_iface_shapes:
        out['policy_assignments'] = next_out()

    out['obs'] = {}
    for k in sim_iface_shapes['obs'].keys():
        out['obs'][k] = next_out()

    out['stats'] = {}
    for k in sim_iface_shapes['stats'].keys():
        out['stats'][k] = next_out()

    return out

step_func = jax.jit(step_func, donate_argnums=0)

#)==="
