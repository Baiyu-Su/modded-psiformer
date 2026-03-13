"""Runtime sanity checks for distributed training."""

import os

from absl import logging
import jax
from jax.experimental import multihost_utils
import numpy as np


def _expected_process_count() -> int:
  for key in ('JAX_PROCESS_COUNT', 'JAX_NUM_PROCESSES', 'SLURM_NNODES'):
    raw = os.environ.get(key)
    if raw:
      return int(raw)
  return jax.process_count()


def _expected_local_device_count() -> int:
  visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
  if visible:
    devices = [dev for dev in visible.split(',') if dev.strip() and dev != '-1']
    if devices:
      return len(devices)
  return jax.local_device_count()


def assert_global_device_topology(stage: str) -> None:
  """Fails if JAX is not using the expected global multi-host topology."""
  actual_processes = jax.process_count()
  actual_local_devices = jax.local_device_count()
  actual_global_devices = jax.device_count()

  expected_processes = _expected_process_count()
  expected_local_devices = _expected_local_device_count()
  expected_global_devices = expected_processes * expected_local_devices

  if (
      actual_processes != expected_processes
      or actual_local_devices != expected_local_devices
      or actual_global_devices != expected_global_devices
  ):
    raise ValueError(
        f'{stage} sanity check failed: expected '
        f'{expected_processes} hosts x {expected_local_devices} local devices '
        f'= {expected_global_devices} total devices, but JAX sees '
        f'{actual_processes} hosts x {actual_local_devices} local devices '
        f'= {actual_global_devices} total devices.'
    )

  if jax.process_index() == 0:
    logging.info(
        '%s sanity check passed: JAX sees %d hosts x %d local devices = '
        '%d total devices.',
        stage,
        actual_processes,
        actual_local_devices,
        actual_global_devices,
    )


def assert_distinct_host_positions(
    stage: str,
    positions,
    *,
    sample_size: int = 128,
) -> None:
  """Fails if sampled host-local walker positions are identical across hosts."""
  process_count = jax.process_count()
  if process_count <= 1:
    return

  flat_positions = np.asarray(positions).reshape(-1)
  sample_size = min(sample_size, flat_positions.size)
  sample = flat_positions[:sample_size]
  gathered = np.asarray(
      multihost_utils.process_allgather(sample, tiled=False)
  )

  duplicate_pairs = []
  for i in range(process_count):
    for j in range(i + 1, process_count):
      if np.array_equal(gathered[i], gathered[j]):
        duplicate_pairs.append((i, j))

  if duplicate_pairs:
    raise ValueError(
        f'{stage} sanity check failed: host-local walker position samples are '
        f'identical across hosts {duplicate_pairs}.'
    )

  if jax.process_index() == 0:
    logging.info(
        '%s sanity check passed: host-local walker position samples differ '
        'across %d hosts.',
        stage,
        process_count,
    )
