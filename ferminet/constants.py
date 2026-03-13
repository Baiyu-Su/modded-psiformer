# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for FermiNet."""

import functools
import jax
from jax.experimental import multihost_utils
import numpy as np
import kfac_jax


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Canonical broadcast: replicates leading-dim-1 arrays across all devices
# with the correct axis_name. Use instead of
# kfac_jax.utils.broadcast_all_local_devices to avoid PmapSharding mismatch.
broadcast = pmap(lambda x: x)

# Canonical key split with matching axis_name.
# Use instead of kfac_jax.utils.p_split.
p_split = pmap(lambda key: tuple(jax.random.split(key)))

# Shortcut for kfac utils
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(
    kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather),
                               axis_name=PMAP_AXIS_NAME)


GLOBAL_BATCH_AXIS = 'qmc_data'


def make_global_mesh(axis_name: str = GLOBAL_BATCH_AXIS) -> jax.sharding.Mesh:
  """Creates a single-axis mesh over all visible devices."""
  devices = np.asarray(jax.devices())
  return jax.sharding.Mesh(devices, (axis_name,))


def replicated_sharding(
    mesh: jax.sharding.Mesh,
) -> jax.sharding.NamedSharding:
  """Returns replicated sharding over `mesh`."""
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def batch_sharding(
    mesh: jax.sharding.Mesh,
    axis_name: str = GLOBAL_BATCH_AXIS,
) -> jax.sharding.NamedSharding:
  """Returns leading-axis sharding over `mesh`."""
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(axis_name)
  )


def host_local_to_global_array(
    local_inputs,
    mesh: jax.sharding.Mesh,
    pspecs,
):
  """Converts process-local inputs into global arrays on `mesh`."""
  return multihost_utils.host_local_array_to_global_array(
      local_inputs, mesh, pspecs
  )


def global_array_to_host_local_array(
    global_inputs,
    mesh: jax.sharding.Mesh,
    pspecs,
):
  """Converts global arrays on `mesh` back into process-local arrays."""
  return multihost_utils.global_array_to_host_local_array(
      global_inputs, mesh, pspecs
  )
