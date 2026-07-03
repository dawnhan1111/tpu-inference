# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.kernels.collectives import \
    hierarchical_reduce_scatter as hier_rs
from tpu_inference.kernels.megablox.gmm_v2 import apply_act_fn, gmm_v2
from tpu_inference.kernels.sparse_core.ragged_gather import ragged_gather
from tpu_inference.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

# Target chunk size of 2048 slots was found empirically to be optimal
# for MoE workloads (e.g., Qwen) to hide ICI/DMA latency during AllReduce.
TARGET_SLOT_CHUNK_SIZE = 2048


def _override_token_indices_for_random_routing(
        topk_indices: jax.Array, global_num_experts: int) -> jax.Array:
    logger.warning(
        "Forcing random routing should be used for performance testing only.")
    original_topk_indices = topk_indices
    num_tokens, topk = original_topk_indices.shape
    # Forcing random routing is useful to get rid of the effect
    # of routing imbalance during performance debugging.
    # (original_topk_indices // global_num_experts) is just zero, but we keep it so that
    # the all-gather of topk_indices won't be skipped so that the performance comparison between
    # with and without random routing is fair.
    rng_key = jax.random.PRNGKey(42)
    topk_indices = jax.vmap(lambda key: jax.random.choice(
        key, global_num_experts, shape=(topk, ), replace=False))(
            jax.random.split(rng_key, num_tokens)) + (original_topk_indices //
                                                      global_num_experts)
    return topk_indices


def all_gather_topk_indices_and_weights(
        topk_indices: jax.Array, topk_weights: jax.Array, dtype: jnp.dtype,
        mesh: Mesh) -> tuple[jax.Array, jax.Array]:
    # `topk_indices` and `topk_weights` are relatively small (and last dimension is top-k),
    # directly all-gather them is inefficient. We use reshape, bitcast to convert the data into one array,
    #  all gather, then unpack.
    top_k = topk_indices.shape[-1]
    topk_indices = topk_indices.astype(jnp.int32).reshape(-1)
    topk_weights = topk_weights.astype(jnp.float32).reshape(-1)
    topk_weights = jax.lax.bitcast_convert_type(topk_weights,
                                                topk_indices.dtype)

    blob = jnp.stack([topk_indices, topk_weights])
    # The optimization barrier here is to prevent the compiler from reordering the all-gather the operations above.
    blob = jax.lax.optimization_barrier(blob)
    gathered_blob = jax.lax.with_sharding_constraint(
        blob, NamedSharding(mesh, P(None, ShardingAxisName.MLP_DATA)))

    topk_indices = gathered_blob[0]
    topk_weights = gathered_blob[1]
    topk_indices = topk_indices.reshape(-1, top_k)
    topk_weights = jax.lax.bitcast_convert_type(topk_weights, jnp.float32)
    topk_weights = topk_weights.reshape(-1, top_k).astype(dtype)

    return topk_indices, topk_weights


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case "sqrtsoftplus":
            return jnp.sqrt(jax.nn.softplus(x))
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function")


def gmm_wrapper(lhs,
                rhs,
                rhs_scale,
                rhs_bias,
                group_sizes,
                group_offset,
                fuse_act=None,
                preferred_element_type=None):
    gmm_res = gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        zero_initialize=False,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
    )
    return gmm_res


def valid_rows_mask(batch_size: int, group_sizes: jax.Array,
                    group_start: jax.Array, group_end: jax.Array) -> jax.Array:
    """Mask indicating rows processed by current shard."""

    group_sizes_sum = jnp.cumulative_sum(group_sizes, include_initial=True)

    token_start = group_sizes_sum[group_start]
    token_end = group_sizes_sum[group_end]

    index = jnp.arange(batch_size)
    return jnp.where(jnp.logical_and(token_start <= index, index < token_end),
                     True, False)


def moe_gmm_local(x: jax.Array,
                  w1: jax.Array,
                  w1_scale: jax.Array | None,
                  w1_bias: jax.Array | None,
                  w2: jax.Array,
                  w2_scale: jax.Array | None,
                  w2_bias: jax.Array | None,
                  group_sizes: jax.Array,
                  group_offset: jax.Array,
                  topk_argsort_revert_indices: jax.Array,
                  topk_weights: jax.Array,
                  *,
                  activation: str,
                  topk: int,
                  parallelism: Literal["tp", "ep"],
                  enable_rs_kernel: bool = False,
                  onehot_moe_permute_threshold: int = 0,
                  scatter_results: bool = False,
                  tp_shard: str = "F") -> jax.Array:
    """Main MoE logic on a local shard can run in TP or EP mode.

    Set parallelism for "tp" or "ep". For TP, tp_shard selects which weight dim
    is sharded: "F" (intermediate, standard column->row MLP, reduce after gmm2)
    or "D" (hidden, row->column MLP: reduce after gmm1, output H-sharded).
    """

    assert parallelism in ["tp", "ep"]
    # Both "D" and "DF" split the gmm1 contraction (hidden H), so gmm1 is partial
    # and must be reduced before the activation; "F" (default) does not.
    d_shard = (parallelism == "tp" and tp_shard in ("D", "DF"))

    if parallelism == "tp" and tp_shard == "DF":
        # per-device local shapes: expect x=[tok,H/D], w1=[E,H/D,2I/F],
        # w2=[E,I/F,H/D] -> confirms H sharded over D and I over F.
        logger.info("[MoE DF local] x=%s w1=%s w2=%s w1_scale=%s w2_scale=%s",
                    x.shape, w1.shape, w2.shape,
                    None if w1_scale is None else w1_scale.shape,
                    None if w2_scale is None else w2_scale.shape)

    # GMM1 computes x @ (W_up | W_gate) together and activation, output is [tokens,padded_intermediate_size]
    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        fuse_act=(None if d_shard else activation),
        preferred_element_type=x.dtype,
    )
    if d_shard:
        # D-sharding: hidden (contraction of gmm1) is split, so gmm1 is partial.
        # Reduce across the shard axis BEFORE the activation, then apply swiglu.
        # Pure "D" reduces over the full MLP_TENSOR; the "DF" hybrid only splits H
        # over the model (D) axis, so it reduces over that axis alone (the small,
        # exposed mid-layer all-reduce).
        gmm1_reduce_axis = (_df_shard_axes()[0]
                            if tp_shard == "DF" else ShardingAxisName.MLP_TENSOR)
        gmm1_res = jax.lax.psum(gmm1_res, axis_name=gmm1_reduce_axis)
        gmm1_res = apply_act_fn(gmm1_res, activation)

    # When the parallelism is TP since w2_bias is not sharded, we should only apply bias
    # once, not applying to every shard. So we set w2_bias to 0 to all shards other than
    # shard 0. For EP, it is not needed since bias is sharded on leading expert axis.
    if parallelism == "tp" and w2_bias is not None:
        shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)
    gmm1_res = gmm1_res[:, :w2.shape[1]]  # trim to hidden size if padded
    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    batch_size = gmm2_res.shape[0]
    local_group_size = w1.shape[0]

    if local_group_size < group_sizes.size:
        mask = valid_rows_mask(
            gmm1_res.shape[0],
            group_sizes,
            group_offset,
            group_offset + local_group_size,
        )[topk_argsort_revert_indices].reshape(-1, topk, 1)
    else:
        mask = jnp.full((batch_size, ), True).reshape(-1, topk, 1)

    reduction_axis = (ShardingAxisName.MLP_TENSOR
                      if parallelism == "tp" else ShardingAxisName.EXPERT)

    if local_group_size < group_sizes.size:
        if batch_size <= onehot_moe_permute_threshold:
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            assert batch_size % topk == 0, (
                f"batch_size ({batch_size}) should be a multiple of topk "
                f"({topk})")
            num_tokens = batch_size // topk
            revert_indices = topk_argsort_revert_indices.reshape(
                num_tokens, topk)
            onehot = jax.nn.one_hot(revert_indices,
                                    batch_size,
                                    dtype=gmm2_res.dtype)
            combine = (onehot * topk_weights[..., None] * mask).sum(axis=1)
            out = combine @ gmm2_res
        else:
            out = ragged_gather_reduce(gmm2_res, topk_argsort_revert_indices,
                                       topk_weights.reshape(-1),
                                       mask.reshape(-1), topk)
    else:
        token_hidden_full = gmm2_res[topk_argsort_revert_indices]
        cur_sorted = token_hidden_full.reshape((-1, topk, gmm2_res.shape[-1]))
        cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
        cur_weighted = cur_sorted * cur_topk_weights
        cur_masked = jnp.where(mask, cur_weighted, 0.0)
        out = cur_masked.sum(axis=-2)

    if d_shard and tp_shard == "DF":
        # Hybrid DF: gmm2 contracts the F-sharded intermediate I, so its output is
        # partial over the F axes AND sharded on hidden H over the D
        # (attn_dp_expert) axis. Both D and F are attention data-parallel axes.
        # (1) Reduce the F partials over `attn_dp` -- a reduce-scatter that also
        #     scatters tokens onto attn_dp under DP-attention (scatter_results),
        #     else a plain all-reduce.
        # (2) Over the D axis `attn_dp_expert` (which also carries DP tokens):
        #     all-to-all to scatter tokens onto it AND gather the H shards back to
        #     full width -> residual layout (token-sharded 8-way, H-full). With no
        #     DP-attention there are no tokens to scatter, so a plain all-gather.
        d_axis, f_axis = _df_shard_axes()
        if scatter_results:
            dp_axes = ShardingAxisName.ATTN_DATA
            reduce_axes = tuple(a for a in f_axis if a not in dp_axes)
            scatter_axes = tuple(a for a in f_axis if a in dp_axes)
            if reduce_axes:
                out = jax.lax.psum(out, axis_name=reduce_axes)
            if scatter_axes:
                out = jax.lax.psum_scatter(out,
                                           axis_name=scatter_axes,
                                           scatter_dimension=0,
                                           tiled=True)
            out = jax.lax.all_to_all(out,
                                     d_axis,
                                     split_axis=0,
                                     concat_axis=1,
                                     tiled=True)
        else:
            out = jax.lax.psum(out, axis_name=f_axis)
            out = jax.lax.all_gather(out, d_axis, axis=1, tiled=True)
        return out.astype(x.dtype)

    if d_shard:
        # D-sharding: gmm2 output is H-sharded (no reduce needed). Reassemble to
        # the residual layout. Under DP-attention (scatter_results) the residual is
        # token-sharded + H-full, so all-to-all (split tokens, concat H); otherwise
        # all-gather H to give token-full + H-full.
        if scatter_results:
            out = jax.lax.all_to_all(out,
                                     ShardingAxisName.MLP_TENSOR,
                                     split_axis=0,
                                     concat_axis=1,
                                     tiled=True)
        else:
            out = jax.lax.all_gather(out,
                                     ShardingAxisName.MLP_TENSOR,
                                     axis=1,
                                     tiled=True)
        return out.astype(x.dtype)

    # Then global reduction on all ranks for all tokens and all experts
    if enable_rs_kernel:
        reduction_axes = reduction_axis if isinstance(
            reduction_axis, tuple) else (reduction_axis, )
        num_devices = 1
        for axis in reduction_axes:
            num_devices *= jax.lax.axis_size(axis)

        # Fallback to psum-scatter for small token sizes to avoid Mosaic compilation.
        # The threshold is chosen based on the tile dimension (8) in the
        # hierarchical reduce-scatter kernel.
        if out.shape[0] // num_devices < 8:
            out = jax.lax.psum_scatter(out,
                                       axis_name=reduction_axis,
                                       scatter_dimension=0,
                                       tiled=True).astype(x.dtype)
        else:
            # Determine the number of micro-batches
            # Use 4 for large inputs to improve efficiency by maximizing the number of
            # concurrent reduction streams, and 2 for smaller inputs to fit in ~32MB VMEM
            num_mb = 2
            if out.shape[0] // num_devices > 600:
                num_mb = 4
            rs_out = hier_rs.hierarchical_reduce_scatter_local(
                out,
                num_devices=num_devices,
                num_micro_batches=num_mb,
                axis_name=reduction_axis)
            out = rs_out.astype(x.dtype)
    elif scatter_results:
        dp_axes = ShardingAxisName.ATTN_DATA
        if isinstance(reduction_axis, tuple):
            reduce_axes = tuple(a for a in reduction_axis if a not in dp_axes)
            scatter_axes = tuple(a for a in reduction_axis if a in dp_axes)
        else:
            reduce_axes = () if reduction_axis in dp_axes else (
                reduction_axis, )
            scatter_axes = (
                reduction_axis, ) if reduction_axis in dp_axes else ()

        if reduce_axes:
            out = jax.lax.psum(out, axis_name=reduce_axes)

        if scatter_axes:
            out = jax.lax.psum_scatter(out,
                                       axis_name=scatter_axes,
                                       scatter_dimension=0,
                                       tiled=True).astype(x.dtype)
        else:
            out = out.astype(x.dtype)
    else:
        out = jax.lax.psum(out, axis_name=reduction_axis).astype(x.dtype)
    return out


def _df_shard_axes():
    """Mesh axes for the hybrid "DF" MoE sharding (MOE_TP_SHARD=DF).

    Both axes are attention DATA-PARALLEL sub-axes, so attention stays a plain
    8-way DP with full heads and MLA is NEVER TP-sharded (model stays 1):
      D = 'attn_dp_expert' shards the hidden dim H (gives gmm2 a deeper K than
          pure-F); its collectives are a 2-way all-reduce (mid) + all-to-all (end).
      F = MLP_TENSOR minus attn_dp_expert (i.e. 'attn_dp') shards the intermediate.
    With attn_dp_size=8 + attn_dp_expert_size=2 on TP=8 this is a 2-way D
    (attn_dp_expert) x 4-way F (attn_dp) split. Returns ``(d_axis, f_axis)``.
    """
    d_axis = "attn_dp_expert"
    mt = ShardingAxisName.MLP_TENSOR
    mt = mt if isinstance(mt, tuple) else (mt, )
    f_axis = tuple(a for a in mt if a != d_axis)
    return d_axis, f_axis


def tensor_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    enable_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    scatter_results: bool = False,
) -> jax.Array:
    tp_shard = os.getenv("MOE_TP_SHARD", "F")
    MT = ShardingAxisName.MLP_TENSOR
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    attn_data_p_spec = P(ShardingAxisName.ATTN_DATA)
    group_offset = jnp.array([0])

    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    if tp_shard == "D":
        # Shard hidden H: w1=[E,H,2I] dim1, w2=[E,I,H] dim2; x sharded on H so
        # gmm1 contracts H/N (reduce after gmm1); gmm2 contracts full I (deep K).
        x_in_spec = P(ShardingAxisName.MLP_DATA, MT)
        w1_spec = P(None, MT, None)
        w2_spec = P(None, None, MT)
        w1_scale_spec = (None if w1_scale is None else P(None, MT, None, None))
        w1_bias_spec = (None if w1_bias is None else P(None, None, None))
        w2_scale_spec = (None if w2_scale is None else P(None, None, None, MT))
        w2_bias_spec = None if w2_bias is None else P(None, None, MT)
    elif tp_shard == "DF":
        # Hybrid: shard hidden H over the D axis 'attn_dp_expert' AND intermediate
        # 2I/I over the F axis 'attn_dp' (both are attention DP axes, so model stays
        # 1 and MLA is untouched). w1=[E,H,2I], w2=[E,I,H]. gmm1 contracts H/D
        # (reduce over D after gmm1); gmm2 contracts I/F (reduce over F after gmm2),
        # giving K = I/F -- deeper than pure-F's I/8.
        d_axis, f_axis = _df_shard_axes()
        x_in_spec = P(ShardingAxisName.MLP_DATA, d_axis)
        w1_spec = P(None, d_axis, f_axis)
        w2_spec = P(None, f_axis, d_axis)
        w1_scale_spec = (None if w1_scale is None else
                         P(None, d_axis, None, f_axis))
        w1_bias_spec = (None if w1_bias is None else P(None, None, f_axis))
        if w2_scale is None:
            w2_scale_spec = None
        elif num_blocks == 1:
            # per-channel along I: the I-block dim is size 1, not shardable over F.
            w2_scale_spec = P(None, None, None, d_axis)
        else:
            w2_scale_spec = P(None, f_axis, None, d_axis)
        w2_bias_spec = None if w2_bias is None else P(None, None, d_axis)
        d_deg = get_mesh_shape_product(mesh, d_axis)
        f_deg = get_mesh_shape_product(mesh, f_axis)
        logger.info(
            "[MoE DF] d_axis=%s(x%d) f_axis=%s(x%d) | global shapes: "
            "x=%s w1=%s(H,2I) w2=%s(I,H) w1_scale=%s w2_scale=%s | "
            "x_spec=%s w1_spec=%s w2_spec=%s", d_axis, d_deg, f_axis, f_deg,
            x.shape, w1.shape, w2.shape,
            None if w1_scale is None else w1_scale.shape,
            None if w2_scale is None else w2_scale.shape, x_in_spec, w1_spec,
            w2_spec)
    else:
        # F (default): shard intermediate 2I/I (standard column->row MLP).
        x_in_spec = data_p_spec
        w1_spec = P(None, None, MT)
        w2_spec = P(None, MT, None)
        w1_scale_spec = (None if w1_scale is None else P(None, None, None, MT))
        w1_bias_spec = (None if w1_bias is None else P(None, None, MT))
        w2_scale_spec = (None if num_blocks == 1 else P(None, MT, None, None))
        w2_bias_spec = None if w2_bias is None else P(None, None, None)

    if tp_shard in ("D", "DF") and x.shape[-1] != w1.shape[1]:
        # w13's contraction (H) was padded for quant-block alignment; pad x to
        # match so it shards to the same per-chip size as the weight. gmm2's
        # output H (w2) is unpadded, so the result needs no un-padding.
        x = jnp.pad(x, [(0, 0), (0, w1.shape[1] - x.shape[-1])])

    if scatter_results:
        final_out_specs = attn_data_p_spec
    else:
        final_out_specs = data_p_spec

    return jax.shard_map(
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="tp",
            enable_rs_kernel=False,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
            tp_shard=tp_shard,
        ),
        mesh=mesh,
        in_specs=(
            x_in_spec,
            w1_spec,
            w1_scale_spec,
            w1_bias_spec,
            w2_spec,
            w2_scale_spec,
            w2_bias_spec,
            data_p_spec,
            P(),
            data_p_spec,
            data_p_spec,
        ),
        out_specs=(final_out_specs),
        check_vma=False,
    )(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        group_offset,
        topk_argsort_revert_indices,
        topk_weights,
    )


def expert_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    enable_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    scatter_results: bool = False,
) -> jax.Array:
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
    ep_p_spec = P(ShardingAxisName.EXPERT)
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    ep_data_p_spec = P(ShardingAxisName.EXPERT_DATA)
    attn_data_p_spec = P(ShardingAxisName.ATTN_DATA)
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec

    if scatter_results:
        final_out_specs = attn_data_p_spec
    elif enable_rs_kernel:
        final_out_specs = ep_data_p_spec
    else:
        final_out_specs = data_p_spec

    return jax.shard_map(
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="ep",
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            enable_rs_kernel=enable_rs_kernel,
            scatter_results=scatter_results,
        ),
        mesh=mesh,
        in_specs=(
            data_p_spec,
            ep_p_spec,
            w1_scale_spec,
            w1_bias_spec,
            ep_p_spec,
            w2_scale_spec,
            w2_bias_spec,
            data_p_spec,
            ep_p_spec,
            data_p_spec,
            data_p_spec,
        ),
        out_specs=(final_out_specs),
        check_vma=False,
    )(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        group_offset,
        topk_argsort_revert_indices,
        topk_weights,
    )


def _apply_all_gather_fp8(hidden_states: jax.Array, mesh: Mesh,
                          dtype: jnp.dtype) -> jax.Array:
    logger.info("Apply FP8 all-gather on input of MOE")
    hidden_states_q, scale = quantize_tensor(
        jnp.float8_e4m3fn,
        hidden_states,
        axis=-1,
    )
    # quantize_tensor squeezes the scale if axis is int. We need to expand it back.
    scale = jnp.expand_dims(scale, -1)

    # Dequantize if needed
    return jax.shard_map(
        lambda x, s: (x.astype(jnp.float32) * s).astype(dtype),
        mesh=mesh,
        in_specs=(
            P(ShardingAxisName.MLP_DATA, None),
            P(ShardingAxisName.MLP_DATA, None),
        ),
        out_specs=P(ShardingAxisName.MLP_DATA, None),
    )(hidden_states_q, scale)


@jax.jit(static_argnames=(
    "topk",
    "renormalize",
    "mesh",
    "use_ep",
    "activation",
    "scoring_fn",
    "all_gather_fp8",
    "enable_rs_kernel",
    "onehot_moe_permute_threshold",
    "scatter_results",
))
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
    scoring_fn: str,
    all_gather_fp8: bool = False,
    enable_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    scatter_results: bool = False,
    hash_based_topk_indices: jax.Array | None = None,
    expert_score_correction_bias: jax.Array | None = None,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: first moe weights [num_experts, hidden_size, intermediate_size * 2]
        w2: second moe weights [num_experts, intermediate_size, hidden_size]
        w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
        w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
        w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
        w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
        gating_output: routing information of tokens [num_tokens, num_experts]
        topk: number of experts to choose per token.
        renormalize: normalize gating_output.
        mesh: mesh to perform moe.
        use_ep: use expert parallelism.
        activation: activation function to perform on the output of w1.
        scoring_fn: scoring function to apply on gating_output.
        enable_rs_kernel: enable custom Hierarchical Reduce-Scatter kernel.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    assert gating_output.shape == (num_tokens, global_num_experts)

    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    if hash_based_topk_indices is not None:
        topk_indices = hash_based_topk_indices
        topk_weights = jnp.take_along_axis(topk_weights, topk_indices, axis=-1)
    elif envs.MOE_APPROX_TOPK:
        topk_weights, topk_indices = jax.lax.approx_max_k(
            topk_weights,
            k=topk,
            recall_target=envs.MOE_APPROX_TOPK_RECALL_TARGET)
    else:
        if expert_score_correction_bias is not None:
            _, topk_indices = jax.lax.top_k(
                topk_weights + expert_score_correction_bias[None, :], k=topk)
            topk_weights = jnp.take_along_axis(topk_weights,
                                               topk_indices,
                                               axis=-1)
        else:
            topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    # All gathering topk_indices and topk_weights if attention dp is used.
    if get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA) > 1:
        topk_indices, topk_weights = all_gather_topk_indices_and_weights(
            topk_indices, topk_weights, dtype, mesh)
    topk_weights = topk_weights.astype(dtype)
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))

    # Only enable Reduce-Scatter if flag is on and Attention is pure DP
    total_num_devices = mesh.devices.size
    is_attn_dp = get_mesh_shape_product(
        mesh, ShardingAxisName.ATTN_DATA) == total_num_devices
    actual_enable_rs_kernel = enable_rs_kernel and is_attn_dp

    if envs.FORCE_MOE_RANDOM_ROUTING:
        logger.warning(
            "Forcing random routing should be used for performance testing only."
        )
        # Forcing random routing is useful to get rid of the effect
        # of routing imbalance during performance debugging.
        topk_indices = _override_token_indices_for_random_routing(
            topk_indices, global_num_experts)

    def _process_tokens_locally(hidden_states_local, topk_indices_local):
        num_tokens_local = hidden_states_local.shape[0]
        topk_indices_flat = topk_indices_local.flatten()
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        token_indices = jnp.arange(num_tokens_local,
                                   dtype=jnp.int32).repeat(topk)
        token_indices_sorted = token_indices[topk_argsort_indices]
        # Below one_hot is equivalent to jnp.bincount(topk_indices_flat,
        # length=global_num_experts) but is more performant.
        group_sizes_local = jax.nn.one_hot(topk_indices_flat,
                                           global_num_experts,
                                           dtype=jnp.int32).sum(axis=0)
        topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)

        if use_ep:
            num_ep_shard = get_mesh_shape_product(mesh,
                                                  ShardingAxisName.EXPERT)
            local_num_experts = global_num_experts // num_ep_shard
            shard_idx = jax.lax.axis_index(ShardingAxisName.EXPERT)

            experts_start = shard_idx * local_num_experts
            experts_end = experts_start + local_num_experts
            group_offsets = jnp.cumulative_sum(group_sizes_local,
                                               include_initial=True)
            shard_output_start = group_offsets[experts_start]
            shard_output_end = group_offsets[experts_end]
            num_tokens = token_indices_sorted.shape[0]
            if num_tokens <= onehot_moe_permute_threshold:
                # Use one-hot matmul for permutation, which can be faster
                # for small batch size
                onehot = jax.nn.one_hot(token_indices_sorted,
                                        hidden_states_local.shape[0],
                                        dtype=hidden_states_local.dtype)
                x = onehot @ hidden_states_local
            else:
                x = ragged_gather(
                    hidden_states_local,
                    token_indices_sorted,
                    shard_output_start,
                    shard_output_end,
                )
        else:
            x = hidden_states_local[token_indices_sorted]

        return x, group_sizes_local, topk_argsort_revert_indices

    if all_gather_fp8:
        hidden_states = _apply_all_gather_fp8(hidden_states, mesh, dtype)

    x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(
            P(ShardingAxisName.MLP_DATA, None),
            P(ShardingAxisName.MLP_DATA, None),
        ),
        out_specs=(
            P(ShardingAxisName.MLP_DATA),
            P(ShardingAxisName.MLP_DATA),
            P(ShardingAxisName.MLP_DATA),
        ),
        check_vma=False,
    )(hidden_states, topk_indices)

    try:
        x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    except Exception as e:
        raise ValueError(
            f"Error when padding input hidden states from {hidden_size} to {padded_hidden_size}."
        ) from e

    if use_ep:
        x = expert_parallel_gmm(
            x,
            w1,
            w1_scale,
            w1_bias,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            topk_argsort_revert_indices,
            topk_weights,
            activation=activation,
            topk=topk,
            mesh=mesh,
            enable_rs_kernel=actual_enable_rs_kernel,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
        )
    else:
        x = tensor_parallel_gmm(
            x,
            w1,
            w1_scale,
            w1_bias,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            topk_argsort_revert_indices,
            topk_weights,
            activation=activation,
            topk=topk,
            mesh=mesh,
            enable_rs_kernel=actual_enable_rs_kernel,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
        )

    return x[:num_tokens, :hidden_size]
