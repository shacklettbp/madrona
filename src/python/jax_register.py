R"===(#"

from functools import partial

import jax
from jax import core, dtypes
from jax.core import ShapedArray, Effect
from jax.lib import xla_client
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, dtype_to_ir_type
from jaxlib.hlo_helpers import custom_call
import builtins as __builtins__

from jax._src import effects
from jax._src.lib.mlir.dialects import hlo

class SimEffect(Effect):
    __str__ = lambda self: "Sim"
_SimEffect = SimEffect()
mlir.lowerable_effects.add_type(SimEffect)
effects.ordered_effects.add_type(SimEffect)
effects.control_flow_allowed_effects.add_type(SimEffect)

custom_call_prefix = f"{type(sim_obj).__name__}_{id(sim_obj)}"
init_custom_call_name = f"{custom_call_prefix}_init"
step_custom_call_name = f"{custom_call_prefix}_step"
del sim_obj

xla_client.register_custom_call_target(
    init_custom_call_name, init_custom_call_capsule,
    platform=custom_call_platform)

xla_client.register_custom_call_target(
    step_custom_call_name, step_custom_call_capsule,
    platform=custom_call_platform)

def _row_major_layout(shape):
    return tuple(range(len(shape) -1, -1, -1))

def _shape_dtype_to_abstract_vals(vs):
    return tuple(ShapedArray(v.shape, v.dtype) for v in vs)

def _lower_shape_dtypes(shape_dtypes):
    return [ir.RankedTensorType.get(i.shape, dtype_to_ir_type(i.dtype))
        for i in shape_dtypes]

# Below code uses ordered effects, which is internal logic taken from
# jax io_callback code and emit_python_callback code. The idea is a
# token is threaded through the custom_call, which preserves ordering and
# prevents sim_step calls from being elided if their outputs aren't used.
# This code deviates slightly from the jax convention which is to put
# the token in the first input / output on GPU. Instead, we put the token
# in the first input and the *last* output, which means we can just skip
# the first buffer passed to the custom call target (the input token)
# and write to the rest of the buffers normally, leaving the final token
# output buffer untouched.

def _prepend_token_to_inputs(types, layouts):
    return [hlo.TokenType.get(), *types], [(), *layouts]

def _append_token_to_results(types, layouts):
    return [*types, hlo.TokenType.get()], [*layouts, ()]

def _init_lowering(ctx):
    # It seems like the input token shouldn't be necessary for init,
    # but using it allows us to keep parity between init and step
    # for gpuEntryFn, which skips the first buffer input.
    token = ctx.tokens_in.get(_SimEffect)[0]

    result_types = _lower_shape_dtypes(step_outputs_iface['obs'].values())
    result_layouts = [_row_major_layout(t.shape) for t in result_types]

    result_types, result_layouts = _append_token_to_results(
        result_types, result_layouts)

    results = custom_call(
        init_custom_call_name,
        backend_config=sim_encode,
        operands=[token],
        operand_layouts=[()],
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results

    *results, token = results
    ctx.set_tokens_out(mlir.TokenSet({_SimEffect: (token,)}))
    return results

def _init_abstract():
    return _shape_dtype_to_abstract_vals(step_outputs_iface['obs'].values()), {_SimEffect}


def _flatten_step_output_shape_dtypes():
    result_shape_dtypes = list(step_outputs_iface['obs'].values())

    result_shape_dtypes.append(step_outputs_iface['rewards'])
    result_shape_dtypes.append(step_outputs_iface['dones'])

    result_shape_dtypes += step_outputs_iface['pbt'].values()

    return result_shape_dtypes


def _step_lowering(ctx, *flattened_inputs):
    token = ctx.tokens_in.get(_SimEffect)[0]
    inputs = [token, *flattened_inputs]

    input_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
    input_layouts = [_row_major_layout(t.shape) for t in input_types]
    input_types, input_layouts = _prepend_token_to_inputs(
        input_types, input_layouts)

    result_types = _lower_shape_dtypes(_flatten_step_output_shape_dtypes())
    result_layouts = [_row_major_layout(t.shape) for t in result_types]
    result_types, result_layouts = _append_token_to_results(
        result_types, result_layouts)

    results = custom_call(
        step_custom_call_name,
        backend_config=sim_encode,
        operands=inputs,
        operand_layouts=input_layouts,
        result_types=result_types,
        result_layouts=result_layouts,
        has_side_effect=True,
    ).results

    *results, token = results
    ctx.set_tokens_out(mlir.TokenSet({_SimEffect: (token,)}))

    return results


def _step_abstract(*inputs):
    return _shape_dtype_to_abstract_vals(_flatten_step_output_shape_dtypes()), {_SimEffect}


_init_primitive = core.Primitive(init_custom_call_name)
_init_primitive.multiple_results = True
_init_primitive.def_impl(partial(xla.apply_primitive, _init_primitive))
_init_primitive.def_effectful_abstract_eval(_init_abstract)

mlir.register_lowering(
    _init_primitive,
    _init_lowering,
    platform=custom_call_platform,
)

_step_primitive = core.Primitive(step_custom_call_name)
_step_primitive.multiple_results = True
_step_primitive.def_impl(partial(xla.apply_primitive, _step_primitive))
_step_primitive.def_effectful_abstract_eval(_step_abstract)

mlir.register_lowering(
    _step_primitive,
    _step_lowering,
    platform=custom_call_platform,
)

def init_func():
    flattened_out = _init_primitive.bind()
    return {k: o for k, o in zip(step_outputs_iface['obs'].keys(), flattened_out)}


def step_func(step_inputs):
    flattened_in = []

    flattened_in.append(step_inputs['actions'])
    flattened_in.append(step_inputs['resets'])

    for k in step_inputs_iface['pbt'].keys():
        flattened_in.append(step_inputs['pbt'][k])

    flattened_out = _step_primitive.bind(*flattened_in)

    out = {}

    cur_idx = 0

    def next_out():
        nonlocal cur_idx
        o = flattened_out[cur_idx]
        cur_idx += 1
        return o

    out['obs'] = {}
    for k in step_outputs_iface['obs'].keys():
        out['obs'][k] = next_out()

    out['rewards'] = next_out()
    out['dones'] = next_out()

    out['pbt'] = {}
    for k in step_outputs_iface['pbt'].keys():
        out['pbt'][k] = next_out()

    return out

init_func = jax.jit(init_func)
step_func = jax.jit(step_func)

if ckpt_iface != None:
    load_ckpts_custom_call_name = f"{custom_call_prefix}_load_ckpts"
    get_ckpts_custom_call_name = f"{custom_call_prefix}_get_ckpts"

    def _flatten_load_ckpts_output_shape_dtypes():
        result_shape_dtypes = list(step_outputs_iface['obs'].values())
        return result_shape_dtypes


    def _load_ckpts_lowering(ctx, *flattened_inputs):
        token = ctx.tokens_in.get(_SimEffect)[0]
        inputs = [token, *flattened_inputs]
    
        input_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
        input_layouts = [_row_major_layout(t.shape) for t in input_types]
        input_types, input_layouts = _prepend_token_to_inputs(
            input_types, input_layouts)
    
        result_types = _lower_shape_dtypes(
            _flatten_load_ckpts_output_shape_dtypes())
        result_layouts = [_row_major_layout(t.shape) for t in result_types]
        result_types, result_layouts = _append_token_to_results(
            result_types, result_layouts)
    
        results = custom_call(
            load_custom_call_name,
            backend_config=sim_encode,
            operands=inputs,
            operand_layouts=input_layouts,
            result_types=result_types,
            result_layouts=result_layouts,
            has_side_effect=True,
        ).results
    
        *results, token = results
        ctx.set_tokens_out(mlir.TokenSet({_SimEffect: (token,)}))
    
        return results
    
    
    def _load_ckpts_abstract(*inputs):
        return _shape_dtype_to_abstract_vals(
            _flatten_load_ckpts_output_shape_dtypes()), {_SimEffect}
    
    _load_ckpts_primitive = core.Primitive(load_ckpts_custom_call_name)
    _load_ckpts_primitive.multiple_results = True
    _load_ckpts_primitive.def_impl(partial(xla.apply_primitive, _load_ckpts_primitive))
    _load_ckpts_primitive.def_effectful_abstract_eval(_load_ckpts_abstract)
    
    mlir.register_lowering(
        _load_ckpts_primitive,
        _load_ckpts_lowering,
        platform=custom_call_platform,
    )
    

    def _flatten_get_ckpts_output_shape_dtypes():
        result_shape_dtypes = list(ckpt_iface['data'].values())
        return result_shape_dtypes


    def _get_ckpts_lowering(ctx, *flattened_inputs):
        token = ctx.tokens_in.get(_SimEffect)[0]
        inputs = [token, *flattened_inputs]
    
        input_types = [ir.RankedTensorType(i.type) for i in flattened_inputs]
        input_layouts = [_row_major_layout(t.shape) for t in input_types]
        input_types, input_layouts = _prepend_token_to_inputs(
            input_types, input_layouts)
    
        result_types = _lower_shape_dtypes(
            _flatten_get_ckpts_output_shape_dtypes())
        result_layouts = [_row_major_layout(t.shape) for t in result_types]
        result_types, result_layouts = _append_token_to_results(
            result_types, result_layouts)
    
        results = custom_call(
            get_ckpts_custom_call_name,
            backend_config=sim_encode,
            operands=inputs,
            operand_layouts=input_layouts,
            result_types=result_types,
            result_layouts=result_layouts,
            has_side_effect=True,
        ).results
    
        *results, token = results
        ctx.set_tokens_out(mlir.TokenSet({_SimEffect: (token,)}))
    
        return results
    
    
    def _get_ckpts_abstract():
        return _shape_dtype_to_abstract_vals(
            _flatten_get_ckpts_output_shape_dtypes()), {_SimEffect}
    
    _get_ckpts_primitive = core.Primitive(get_ckpts_custom_call_name)
    _get_ckpts_primitive.multiple_results = True
    _get_ckpts_primitive.def_impl(partial(xla.apply_primitive, _get_ckpts_primitive))
    _get_ckpts_primitive.def_effectful_abstract_eval(_get_ckpts_abstract)
    
    mlir.register_lowering(
        _get_ckpts_primitive,
        _get_ckpts_lowering,
        platform=custom_call_platform,
    )
    
    def load_ckpts_func(trigger_load, ckpts):
        flattened_out = _load_ckpts_primitive.bind(trigger_load, ckpts)
        return {k: o for k, o in zip(step_outputs_iface['obs'].keys(), flattened_out)}

    def get_ckpts_func():
        flattened_out = _get_ckpts_primitive.bind(trigger_load, ckpts)
        return flattened_out
    
    
    load_ckpts_func = jax.jit(load_ckpts_func)
    get_ckpts_func = jax.jit(get_ckpts_func)

#)==="
