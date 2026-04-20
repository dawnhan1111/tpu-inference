# Copyright 2025 Google LLC
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

"""Hierarchical Recursive Doubling Reduce-Scatter Implementation.

This module provides a prototype implementation of Hierarchical
Reduce-Scatter using Pallas on TPUs.
"""

import math

import jax
from jax import sharding
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _next_multiple_of(val, multiple):
  return ((val + multiple - 1) // multiple) * multiple


def _accumulate(
    hbm_recv,
    hbm_run,
    vmem_recv,
    vmem_run,
    sync_sems,
    vmem_idx=0,
    hbm_out=None,
):
  """Loads received data and local running sum into VMEM, adds, and stores back or to out_hbm."""
  load_recv_sem, load_run_sem, store_run_sem = sync_sems

  load_recv_op = pltpu.make_async_copy(
      hbm_recv, vmem_recv.at[vmem_idx], load_recv_sem
  )
  load_run_op = pltpu.make_async_copy(
      hbm_run, vmem_run.at[vmem_idx], load_run_sem
  )
  load_recv_op.start()
  load_run_op.start()
  load_recv_op.wait()
  load_run_op.wait()

  vmem_run_bf16 = vmem_run[vmem_idx, ...].astype(jnp.bfloat16)
  vmem_recv_bf16 = vmem_recv[vmem_idx, ...].astype(jnp.bfloat16)
  res_bf16 = vmem_run_bf16 + vmem_recv_bf16
  vmem_run[vmem_idx, ...] = res_bf16.astype(vmem_run.dtype)

  if hbm_out is None:
    hbm_out = hbm_run

  store_run_op = pltpu.make_async_copy(
      vmem_run.at[vmem_idx], hbm_out, store_run_sem
  )
  store_run_op.start()
  return store_run_op


def _get_hypercube_chunk_idx(
    loop_idx, future_dims, prev_dims, my_chip_id, target_dim, dim_val
):
  """Calculates the specific chunk index representing a hypercube node's data.

  This function determines which chunk of the global tensor this device should
  operate on during a specific step of the hypercube algorithm. It constructs
  the chunk index by combining bits from:
  1. The device's own ID for dimensions already processed (prev_dims).
  2. The loop index for dimensions yet to be processed (future_dims).
  3. The target dimension's value (dim_val).

  Returns:
    The calculated chunk index (integer).
  """
  base = 0
  for d in prev_dims:
    bit = (my_chip_id >> d) & 1
    base = base | (bit << d)
  for bit_pos, d in enumerate(future_dims):
    bit = (loop_idx >> bit_pos) & 1
    base = base | (bit << d)
  base = base | (dim_val << target_dim)
  return base


def _intra_chip_exchange(
    input_ref,
    recv_buf_ref,
    intra_chip_send_sems,
    intra_chip_recv_sems,
    cur_twin_id,
    cur_twin_bit,
    num_chips,
    final_chunk_size,
    mb_start,
    mb_size_actual,
    micro_batch_idx,
):
  """Initiates data exchange between twin cores on the same chip.

  This function sets up asynchronous remote copy operations to exchange data
  chunks between the current device and its twin on the same chip. This is the
  first phase of the hierarchical reduce-scatter.

  Args:
    input_ref: The input HBM buffer reference.
    recv_buf_ref: The HBM buffer reference to store received data.
    intra_chip_send_sems: Semaphores for sending data.
    intra_chip_recv_sems: Semaphores for receiving data.
    cur_twin_id: The logical device ID of the twin core.
    cur_twin_bit: 0 if the current device is the even twin, 1 otherwise.
    num_chips: Total number of chips in the system.
    final_chunk_size: The size of the final reduced chunk per device.
    mb_start: The starting index of the current micro-batch in the hidden dim.
    mb_size_actual: The actual size of the current micro-batch.
    micro_batch_idx: The index of the current micro-batch.

  Returns:
    A list of asynchronous copy operations.
  """
  intra_chip_ops = []
  for pair_idx in range(num_chips):
    neighbor_chunk = pair_idx * 2 + (1 - cur_twin_bit)
    op = pltpu.make_async_remote_copy(
        src_ref=input_ref.at[
            pl.ds(neighbor_chunk * final_chunk_size, final_chunk_size),
            pl.ds(mb_start, mb_size_actual),
        ],
        dst_ref=recv_buf_ref.at[
            pl.ds(neighbor_chunk * final_chunk_size, final_chunk_size),
            pl.ds(mb_start, mb_size_actual),
        ],
        send_sem=intra_chip_send_sems.at[pair_idx, micro_batch_idx],
        recv_sem=intra_chip_recv_sems.at[pair_idx, micro_batch_idx],
        device_id=cur_twin_id,
        device_id_type=pltpu.DeviceIdType.LOGICAL,
    )
    op.start()
    intra_chip_ops.append(op)
  return intra_chip_ops


def _intra_chip_reduce(
    intra_chip_ops,
    recv_buf_ref,
    input_ref,
    running_sum_ref,
    vmem_recv_ref,
    vmem_run_ref,
    sync_sems,
    cur_twin_bit,
    num_chips,
    final_chunk_size,
    mb_start,
    mb_size_actual,
    vmem_pipeline_depth,
):
  """Reduces data within the same chip by accumulating received and local data.

  This function waits for the intra-chip exchange operations to complete,
  loads the received data and the local running sum into VMEM, performs
  the accumulation, and stores the result back to the running sum buffer in HBM.

  Args:
    intra_chip_ops: A list of asynchronous copy operations from
      _intra_chip_exchange.
    recv_buf_ref: The HBM buffer reference where received data is stored.
    input_ref: The input HBM buffer reference containing the local data.
    running_sum_ref: The HBM buffer reference for storing the running sum.
    vmem_recv_ref: VMEM buffer for temporarily storing received data.
    vmem_run_ref: VMEM buffer for temporarily storing the running sum.
    sync_sems: Tuple of semaphores for VMEM load/store operations.
    cur_twin_bit: 0 if the current device is the even twin, 1 otherwise.
    num_chips: Total number of chips in the system.
    final_chunk_size: The size of the final reduced chunk per device.
    mb_start: The starting index of the current micro-batch in the hidden dim.
    mb_size_actual: The actual size of the current micro-batch.
    vmem_pipeline_depth: The depth of the VMEM pipeline.
  """
  store_ops = []
  for pair_idx in range(num_chips):
    cur_chunk = pair_idx * 2 + cur_twin_bit
    intra_chip_ops[pair_idx].wait()
    if pair_idx >= vmem_pipeline_depth:
      store_ops[pair_idx - vmem_pipeline_depth].wait()
    s_op = _accumulate(
        recv_buf_ref.at[
            pl.ds(cur_chunk * final_chunk_size, final_chunk_size),
            pl.ds(mb_start, mb_size_actual),
        ],
        input_ref.at[
            pl.ds(cur_chunk * final_chunk_size, final_chunk_size),
            pl.ds(mb_start, mb_size_actual),
        ],
        vmem_recv_ref.at[
            :, pl.ds(0, final_chunk_size), pl.ds(0, mb_size_actual)
        ],
        vmem_run_ref.at[
            :, pl.ds(0, final_chunk_size), pl.ds(0, mb_size_actual)
        ],
        sync_sems,
        vmem_idx=pair_idx % vmem_pipeline_depth,
        hbm_out=running_sum_ref.at[
            pl.ds(cur_chunk * final_chunk_size, final_chunk_size),
            pl.ds(mb_start, mb_size_actual),
        ],
    )
    store_ops.append(s_op)
  for s_op in store_ops:
    s_op.wait()


def _inter_chip_exchange(
    running_sum_ref,
    recv_buf_ref,
    sync_sems_c2c_send,
    sync_sems_c2c_recv,
    cur_chip_id,
    cur_twin_bit,
    num_hypercube_dims,
    phase_step,
    micro_batch_idx,
    hidden_size_dim,
    final_chunk_size,
    full_chunk_size,
    mb_size,
):
  """Initiates data exchange between chips using a hypercube algorithm.

  This function sets up asynchronous remote copy operations to exchange data
  chunks between the current chip and its neighbor in the current dimension
  of the hypercube. This is part of the inter-chip reduce-scatter phase.

  Args:
    running_sum_ref: The HBM buffer reference for the running sum.
    recv_buf_ref: The HBM buffer reference to store received data.
    sync_sems_c2c_send: Semaphores for sending data between chips.
    sync_sems_c2c_recv: Semaphores for receiving data between chips.
    cur_chip_id: The logical ID of the current chip.
    cur_twin_bit: 0 if the current device is the even twin, 1 otherwise.
    num_hypercube_dims: The number of dimensions in the hypercube.
    phase_step: The current step in the hypercube algorithm (0 to
      num_hypercube_dims - 1).
    micro_batch_idx: The index of the current micro-batch.
    hidden_size_dim: The size of the hidden dimension.
    final_chunk_size: The size of the final reduced chunk per device.
    full_chunk_size: The size of a full chunk for one hypercube dimension.
    mb_size: The size of each micro-batch.

  Returns:
    A list of tuples, each containing an asynchronous copy operation,
    the chunk index, the micro-batch start index, and the micro-batch size.
  """
  mb_ops = []
  # Number of independent reduction operations in this step.
  num_ops_in_step = 2 ** (num_hypercube_dims - 1 - phase_step)
  for op_idx in range(num_ops_in_step):
    for hypercube_dim_idx in range(num_hypercube_dims):
      # Rotate the dimension we operate on based on the phase step.
      # This helps in structuring the recursive doubling.
      dim = (hypercube_dim_idx + phase_step) % num_hypercube_dims
      # Find the neighbor chip that differs only in the 'dim' bit.
      neighbor_chip_id = cur_chip_id ^ (1 << dim)
      my_dim_bit = (cur_chip_id >> dim) & 1
      neigh_dim_bit = 1 - my_dim_bit

      # Keep track of resolved (prev) and unresolved (future) dimensions
      # to calculate which chunk of data to operate on.
      prev_dims = [
          (hypercube_dim_idx + j) % num_hypercube_dims
          for j in range(phase_step)
      ]
      future_dims = [
          (hypercube_dim_idx + j) % num_hypercube_dims
          for j in range(phase_step + 1, num_hypercube_dims)
      ]

      chunk_start = hypercube_dim_idx * full_chunk_size
      chunk_end = min(chunk_start + full_chunk_size, hidden_size_dim)

      mb_start_idx = min(
          chunk_start + (micro_batch_idx * mb_size), chunk_end
      )
      mb_end_idx = min(mb_start_idx + mb_size, chunk_end)
      k_size = mb_end_idx - mb_start_idx

      base_chunk_idx = _get_hypercube_chunk_idx(
          op_idx,
          future_dims,
          prev_dims,
          cur_chip_id,
          target_dim=dim,
          dim_val=my_dim_bit,
      )
      neighbor_base_chunk_idx = _get_hypercube_chunk_idx(
          op_idx,
          future_dims,
          prev_dims,
          cur_chip_id,
          target_dim=dim,
          dim_val=neigh_dim_bit,
      )

      chunk_idx = base_chunk_idx * 2 + cur_twin_bit
      neighbor_chunk_idx = neighbor_base_chunk_idx * 2 + cur_twin_bit

      if k_size > 0:
        op = pltpu.make_async_remote_copy(
            src_ref=running_sum_ref.at[
                pl.ds(
                    neighbor_chunk_idx * final_chunk_size,
                    final_chunk_size,
                ),
                pl.ds(mb_start_idx, k_size),
            ],
            dst_ref=recv_buf_ref.at[
                pl.ds(
                    neighbor_chunk_idx * final_chunk_size,
                    final_chunk_size,
                ),
                pl.ds(mb_start_idx, k_size),
            ],
            # Use separate send and recv semaphores to prevent deadlocks
            send_sem=sync_sems_c2c_send.at[
                phase_step, micro_batch_idx, hypercube_dim_idx, op_idx
            ],
            recv_sem=sync_sems_c2c_recv.at[
                phase_step, micro_batch_idx, hypercube_dim_idx, op_idx
            ],
            device_id=neighbor_chip_id * 2 + cur_twin_bit,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        op.start()
        mb_ops.append((op, chunk_idx, mb_start_idx, k_size))
  return mb_ops


def _inter_chip_reduce(
    mb_ops,
    recv_buf_ref,
    running_sum_ref,
    output_ref,
    vmem_recv_ref,
    vmem_run_ref,
    sync_sems,
    phase_step,
    num_hypercube_dims,
    cur_id,
    final_chunk_size,
    vmem_pipeline_depth,
):
  """Reduces data across chips by accumulating received and running sum data.

  This function waits for the inter-chip exchange operations to complete,
  loads the received data and the local running sum into VMEM, performs
  the accumulation, and stores the result either to the final output buffer
  (if it's the last phase) or back to the running sum buffer in HBM.

  Args:
    mb_ops: A list of tuples containing asynchronous copy operations and
      metadata from _inter_chip_exchange.
    recv_buf_ref: The HBM buffer reference where received data is stored.
    running_sum_ref: The HBM buffer reference for the running sum.
    output_ref: The HBM buffer reference for the final output.
    vmem_recv_ref: VMEM buffer for temporarily storing received data.
    vmem_run_ref: VMEM buffer for temporarily storing the running sum.
    sync_sems: Tuple of semaphores for VMEM load/store operations.
    phase_step: The current step in the hypercube algorithm.
    num_hypercube_dims: The number of dimensions in the hypercube.
    cur_id: The logical device ID of the current core.
    final_chunk_size: The size of the final reduced chunk per device.
    vmem_pipeline_depth: The depth of the VMEM pipeline.
  """
  def make_write_to_output(hidden_start, hidden_size, op_id_arg):
    def write_to_output(operand):
      _, chunk_idx = operand
      s_op = _accumulate(
          recv_buf_ref.at[
              pl.ds(chunk_idx * final_chunk_size, final_chunk_size),
              pl.ds(hidden_start, hidden_size),
          ],
          running_sum_ref.at[
              pl.ds(chunk_idx * final_chunk_size, final_chunk_size),
              pl.ds(hidden_start, hidden_size),
          ],
          vmem_recv_ref.at[
              :, pl.ds(0, final_chunk_size), pl.ds(0, hidden_size)
          ],
          vmem_run_ref.at[:, pl.ds(0, final_chunk_size), pl.ds(0, hidden_size)],
          sync_sems,
          vmem_idx=op_id_arg % vmem_pipeline_depth,
          hbm_out=output_ref.at[:, pl.ds(hidden_start, hidden_size)],
      )
      s_op.wait()
      return None
    return write_to_output

  def make_write_to_running_sum(hidden_start, hidden_size, op_id_arg):
    def write_to_running_sum(operand):
      _, chunk_idx = operand
      s_op = _accumulate(
          recv_buf_ref.at[
              pl.ds(chunk_idx * final_chunk_size, final_chunk_size),
              pl.ds(hidden_start, hidden_size),
          ],
          running_sum_ref.at[
              pl.ds(chunk_idx * final_chunk_size, final_chunk_size),
              pl.ds(hidden_start, hidden_size),
          ],
          vmem_recv_ref.at[
              :, pl.ds(0, final_chunk_size), pl.ds(0, hidden_size)
          ],
          vmem_run_ref.at[:, pl.ds(0, final_chunk_size), pl.ds(0, hidden_size)],
          sync_sems,
          vmem_idx=op_id_arg % vmem_pipeline_depth,
      )
      s_op.wait()
      return None
    return write_to_running_sum

  for op_id, (op, chunk_idx, hidden_start, hidden_size) in enumerate(mb_ops):
    op.wait()
    condition = (phase_step == num_hypercube_dims - 1) & (chunk_idx == cur_id)
    operand = (None, chunk_idx)
    jax.lax.cond(
        condition,
        make_write_to_output(hidden_start, hidden_size, op_id),
        make_write_to_running_sum(hidden_start, hidden_size, op_id),
        operand,
    )


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    vmem_recv_ref,
    vmem_run_ref,
    intra_chip_send_sems,
    intra_chip_recv_sems,
    load_recv_sem,
    load_run_sem,
    store_run_sem,
    *inter_chip_sems,
    num_devices: int,
    num_hypercube_dims: int,
    num_micro_batches: int,
    hidden_size_dim: int,
    final_chunk_size: int,
    full_chunk_size: int,
    mb_size: int,
):
  """Pallas kernel for hierarchical reduce-scatter."""
  sync_sems = (load_recv_sem, load_run_sem, store_run_sem)

  if inter_chip_sems:
    sync_sems_c2c_send = inter_chip_sems[0]
    sync_sems_c2c_recv = inter_chip_sems[1]
  else:
    sync_sems_c2c_send = None
    sync_sems_c2c_recv = None

  cur_id = jax.lax.axis_index("x")
  num_chips = num_devices // 2
  cur_chip_id = cur_id // 2
  cur_twin_bit = cur_id % 2
  is_even = cur_twin_bit == 0
  cur_twin_id = jax.lax.select(is_even, cur_id + 1, cur_id - 1)

  vmem_pipeline_depth = 4  # Outstanding VMEM accumulation stages

  # Intra-chip Reduce-Scatter
  # Sum data between two chiplets on the same chip (fastest link).
  # Each chip has 2 devices. We want to exchange data between them so each
  # device gets a running sum of the local pair.

  intra_chip_mb_size = _next_multiple_of(hidden_size_dim // num_micro_batches, 128)

  for micro_batch_idx in range(num_micro_batches):
    mb_start = micro_batch_idx * intra_chip_mb_size
    mb_size_actual = min(intra_chip_mb_size, hidden_size_dim - mb_start)

    if mb_size_actual > 0:
      with jax.named_scope(f"intra_chip_mb_{micro_batch_idx}"):
        intra_chip_ops = _intra_chip_exchange(
            input_ref,
            recv_buf_ref,
            intra_chip_send_sems,
            intra_chip_recv_sems,
            cur_twin_id,
            cur_twin_bit,
            num_chips,
            final_chunk_size,
            mb_start,
            mb_size_actual,
            micro_batch_idx,
        )

        _intra_chip_reduce(
            intra_chip_ops,
            recv_buf_ref,
            input_ref,
            running_sum_ref,
            vmem_recv_ref,
            vmem_run_ref,
            sync_sems,
            cur_twin_bit,
            num_chips,
            final_chunk_size,
            mb_start,
            mb_size_actual,
            vmem_pipeline_depth,
        )

  # Inter-chip Hypercube
  # Perform reduce-scatter across chips using a hypercube algorithm.
  # In each step, chips exchange data along one dimension of the hypercube.
  # By the end of all steps, each chip holds a fully reduced chunk of the
  # data.
  #
  # Example for a 4-chip setup (2 hypercube dimensions):
  # Chip IDs in binary: 00, 01, 10, 11
  # Step 0 (dim=0):
  #   - Chip 00 exchanges with 01 (differs in bit 0)
  #   - Chip 10 exchanges with 11
  # Step 1 (dim=1):
  #   - Chip 00 exchanges with 10 (differs in bit 1)
  #   - Chip 01 exchanges with 11
  for micro_batch_idx in range(num_micro_batches):
    mb_start_idx_all = micro_batch_idx * mb_size
    if mb_start_idx_all < hidden_size_dim:
      for phase_step in range(num_hypercube_dims):
        assert (
            sync_sems_c2c_send is not None and sync_sems_c2c_recv is not None
        ), "sync_sems_c2c must be provided for hypercube dims > 0"
        with jax.named_scope(
            f"inter_chip_step_{phase_step}_mb_{micro_batch_idx}"
        ):
          mb_ops = _inter_chip_exchange(
              running_sum_ref,
              recv_buf_ref,
              sync_sems_c2c_send,
              sync_sems_c2c_recv,
              cur_chip_id,
              cur_twin_bit,
              num_hypercube_dims,
              phase_step,
              micro_batch_idx,
              hidden_size_dim,
              final_chunk_size,
              full_chunk_size,
              mb_size,
          )

          _inter_chip_reduce(
              mb_ops,
              recv_buf_ref,
              running_sum_ref,
              output_ref,
              vmem_recv_ref,
              vmem_run_ref,
              sync_sems,
              phase_step,
              num_hypercube_dims,
              cur_id,
              final_chunk_size,
              vmem_pipeline_depth,
          )


def hierarchical_reduce_scatter(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
        "x", None
    ),
    num_micro_batches: int = 1,
) -> jax.Array:
  """Performs hierarchical reduce-scatter using Pallas.

  Args:
    x: Sharded input array, sharding should be PartitionSpec('x', None).
    mesh: JAX Mesh object.
    in_specs: PartitionSpec for input array.
    num_micro_batches: Number of micro-batches for pipelining.

  Returns:
    Reduced and scattered array.
  """
  num_devices = mesh.devices.size
  num_chips = num_devices // 2
  num_hypercube_dims = int(math.log2(num_chips)) if num_chips > 0 else 0

  # Assertions for production robustness
  assert (
      num_devices % 2 == 0
  ), "Number of devices must be even (twin assumption)"
  assert (
      2**num_hypercube_dims == num_chips
  ), "Number of chips must be a power of 2 for hypercube algorithm"

  global_tokens = x.shape[0]
  hidden_size_dim = math.prod(x.shape[1:])

  # Check if Tokens dimension is sharded in in_specs.
  # Default in_specs is P('x', None), meaning tokens are sharded.
  tokens_sharded = True
  if in_specs is not None and len(in_specs) > 0 and in_specs[0] is None:
    tokens_sharded = False

  seq_len_dim = (
      global_tokens // num_devices if tokens_sharded else global_tokens
  )

  chunk_size_raw = (
      hidden_size_dim // num_hypercube_dims
      if num_hypercube_dims > 0
      else hidden_size_dim
  )
  full_chunk_size = _next_multiple_of(chunk_size_raw, 128)

  mb_size_raw = full_chunk_size // num_micro_batches
  mb_size = _next_multiple_of(mb_size_raw, 128)

  final_chunk_size = seq_len_dim // num_devices

  max_vmem_mb_size = _next_multiple_of(
      hidden_size_dim // num_micro_batches, 128
  )

  out_shape = jax.ShapeDtypeStruct(
      (seq_len_dim // num_devices, hidden_size_dim), x.dtype
  )
  running_sum_shape = jax.ShapeDtypeStruct(
      (seq_len_dim, hidden_size_dim), x.dtype
  )
  recv_buf_shape = jax.ShapeDtypeStruct((seq_len_dim, hidden_size_dim), x.dtype)

  vmem_pipeline_depth = 4
  scratch_shapes = [
      pltpu.VMEM(
          (vmem_pipeline_depth, final_chunk_size, max_vmem_mb_size), x.dtype
      ),
      pltpu.VMEM(
          (vmem_pipeline_depth, final_chunk_size, max_vmem_mb_size), x.dtype
      ),
      pltpu.SemaphoreType.DMA(
          (num_chips, num_micro_batches)
      ),  # intra_chip_send_sems
      pltpu.SemaphoreType.DMA(
          (num_chips, num_micro_batches)
      ),  # intra_chip_recv_sems
      pltpu.SemaphoreType.DMA,  # load_recv_sem
      pltpu.SemaphoreType.DMA,  # load_run_sem
      pltpu.SemaphoreType.DMA,  # store_run_sem
  ]

  if num_hypercube_dims > 0:
    # Separate semaphores for send and receive to prevent deadlocks
    scratch_shapes.append(
        pltpu.SemaphoreType.DMA((
            num_hypercube_dims,
            num_micro_batches,
            num_hypercube_dims,
            2 ** (num_hypercube_dims - 1),
        ))
    )
    scratch_shapes.append(
        pltpu.SemaphoreType.DMA((
            num_hypercube_dims,
            num_micro_batches,
            num_hypercube_dims,
            2 ** (num_hypercube_dims - 1),
        ))
    )

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
      out_specs=(
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
      ),
      scratch_shapes=tuple(scratch_shapes),
      grid=(1,),
  )

  # Bind static arguments to the kernel
  kernel_fn = jax.tree_util.Partial(
      hier_rs_kernel,
      num_devices=num_devices,
      num_hypercube_dims=num_hypercube_dims,
      num_micro_batches=num_micro_batches,
      hidden_size_dim=hidden_size_dim,
      final_chunk_size=final_chunk_size,
      full_chunk_size=full_chunk_size,
      mb_size=mb_size,
  )

  hier_rs = pl.pallas_call(
      kernel_fn,
      out_shape=(out_shape, running_sum_shape, recv_buf_shape),
      grid_spec=grid_spec,
  )

  def inner(local_x):
    original_shape = local_x.shape
    flat_local_x = local_x.reshape((original_shape[0], -1))
    final_out, _, _ = hier_rs(flat_local_x)
    return final_out.reshape((final_out.shape[0],) + original_shape[1:])

  return shard_map.shard_map(
      inner,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=sharding.PartitionSpec("x", None),
      check_rep=False,
  )(x)
