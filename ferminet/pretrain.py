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

"""Utilities for pretraining and importing PySCF models."""

import functools
from typing import Callable, Sequence, Tuple

from absl import logging
import chex
from ferminet import constants
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
from ferminet.utils import system
import jax
from jax import numpy as jnp
import numpy as np
import optax
import pyscf


def get_hf(molecule: Sequence[system.Atom] | None = None,
           nspins: Tuple[int, int] | None = None,
           basis: str | None = 'sto-3g',
           pyscf_mol: pyscf.gto.Mole | None = None,
           restricted: bool | None = False) -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol,
                         restricted=restricted)
  else:
    scf_approx = scf.Scf(molecule,
                         nelectrons=nspins,
                         basis=basis,
                         restricted=restricted)
  scf_approx.run()
  return scf_approx



def make_pretrain_step(
    batch_orbitals: networks.OrbitalFnLike,
    batch_network: networks.LogFermiNetLike,
    optimizer_update: optax.TransformUpdateFn,
    electrons: Tuple[int, int],
    batch_size: int = 0,
    full_det: bool = False,
    scf_fraction: float = 0.0,
):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in the
      network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    electrons: number of spin-up and spin-down electrons.
    batch_size: number of walkers per device, used to make MCMC step.
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.
    scf_fraction: What fraction of the wavefunction sampled from is the SCF
      wavefunction and what fraction is the neural network wavefunction?

  Returns:
    Callable for performing a single pretraining optimisation step.
  """

  # Create a function which gives either the SCF ansatz, the neural network
  # ansatz, or a weighted mixture of the two.
  if scf_fraction > 1 or scf_fraction < 0:
    raise ValueError('scf_fraction must be in between 0 and 1, inclusive.')

  scf_network = lambda fn, x: fn(x, electrons)[1]

  if scf_fraction < 1e-6:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      return batch_network(full_params['ferminet'], pos, spins, atoms, charges)
  elif scf_fraction > 0.999999:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      del spins, atoms, charges
      return scf_network(full_params['scf'].eval_slater, pos)
  else:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      log_ferminet = batch_network(full_params['ferminet'], pos, spins, atoms,
                                   charges)
      log_scf = scf_network(full_params['scf'].eval_slater, pos)
      return (1 - scf_fraction) * log_ferminet + scf_fraction * log_scf

  mcmc_step = mcmc.make_mcmc_step(
      mcmc_network, batch_per_device=batch_size, steps=1)

  def loss_fn(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      scf_approx: scf.Scf,
  ):
    pos = data.positions
    spins = data.spins
    scf_orbitals = scf_approx.eval_orbitals
    net_orbitals = batch_orbitals
    target = scf_orbitals(pos, electrons)
    orbitals = net_orbitals(params, pos, spins, data.atoms, data.charges)
    cnorm = lambda x, y: (x - y) ** 2
    if full_det:
      dims = target[0].shape[:-2]
      na = target[0].shape[-2]
      nb = target[1].shape[-2]
      target = jnp.concatenate(
          (
              jnp.concatenate(
                  (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
              jnp.concatenate(
                  (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
          ),
          axis=-2,
      )
      result = jnp.mean(cnorm(target[:, None, ...], orbitals[0]))
    else:
      result = jnp.array([
          jnp.mean(cnorm(t[:, None, ...], o))
          for t, o in zip(target, orbitals)
      ]).sum()
    return constants.pmean(result)

  def pretrain_step(data, params, state, key, scf_approx):
    """One iteration of pretraining to match HF."""
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, scf_approx)
    search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    full_params = {'ferminet': params, 'scf': scf_approx}
    data, pmove = mcmc_step(full_params, data, key, width=0.02)
    return data, params, state, loss_val, pmove

  return pretrain_step


def pretrain_hartree_fock(
    *,
    params: networks.ParamTree,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network: networks.FermiNetLike,
    batch_orbitals: networks.OrbitalFnLike,
    network_options: networks.BaseNetworkOptions,
    sharded_key: chex.PRNGKey,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Callable[[int, float], None] | None = None,
    scf_fraction: float = 0.0,
):
  """Performs training to match initialization as closely as possible to HF.

  Args:
    params: Network parameters.
    positions: Electron position configurations.
    spins: Electron spin configuration (1 for alpha electrons, -1 for beta), as
      a 1D array. Note we always use the same spin configuration for the entire
      batch in pretraining.
    atoms: atom positions (batched).
    charges: atomic charges (batched).
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in the
      network evaluated at those positions.
    network_options: FermiNet network options.
    sharded_key: JAX RNG state (sharded) per device.
    electrons: tuple of number of electrons of each spin.
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    iterations: number of pretraining iterations to perform.
    batch_size: number of walkers per device, used to make MCMC step.
    logger: Callable with signature (step, value) which externally logs the
      pretraining loss.
    scf_fraction: What fraction of the wavefunction sampled from is the SCF
      wavefunction and what fraction is the neural network wavefunction?

  Returns:
    params, positions: Updated network parameters and MCMC configurations such
    that the orbitals in the network closely match Hartree-Fock and the MCMC
    configurations are drawn from the log probability of the network.
  """
  is_host_0 = jax.process_index() == 0
  local_device_count = jax.local_device_count()
  # Pretraining is slow on larger systems (very low GPU utilization) because the
  # Hartree-Fock orbitals are evaluated on CPU and only on a single host.
  # Implementing the basis set in JAX would enable using GPUs and allow
  # eval_orbitals to be pmapped.

  if is_host_0:
    logging.info('Initializing pretrain optimizer state on %d local devices.',
                 local_device_count)
  optimizer = optax.adam(3.e-4)
  opt_state_pt = constants.pmap(optimizer.init)(params)
  if is_host_0:
    logging.info('Pretrain optimizer state initialized.')

  pretrain_step = make_pretrain_step(
      batch_orbitals,
      batch_network,
      optimizer.update,
      electrons=electrons,
      batch_size=batch_size,
      full_det=network_options.full_det,
      scf_fraction=scf_fraction,
  )
  pretrain_step = constants.pmap(pretrain_step)

  # All arrays must use PmapSharding (from broadcast_all_local_devices) to
  # avoid CopyArrays errors when mixed with other pmap outputs in multi-host.
  batch_spins = jnp.tile(spins[None], [positions.shape[1], 1])
  if is_host_0:
    logging.info('Broadcasting pretrain spin configurations to local devices.')
  pmap_spins = constants.broadcast(
      jnp.tile(
          batch_spins[None],
          [local_device_count] + [1] * batch_spins.ndim,
      )
  )
  data = networks.FermiNetData(
      positions=positions, spins=pmap_spins, atoms=atoms,
      charges=charges
  )
  if is_host_0:
    logging.info('Starting Hartree-Fock pretraining for %d iterations.',
                 iterations)
  for t in range(iterations):
    if t == 0 and is_host_0:
      logging.info('Compiling/executing first pretrain step.')
    sharded_key, subkeys = constants.p_split(sharded_key)
    data, params, opt_state_pt, loss, pmove = pretrain_step(
        data, params, opt_state_pt, subkeys, scf_approx)
    if is_host_0:
      logging.info('Pretrain iter %05d: %g %g', t, loss[0], pmove[0])
    if logger and is_host_0:
      logger(t, loss[0])
  return params, data.positions


def pretrain_hartree_fock_global(
    *,
    params: networks.ParamTree,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network: networks.FermiNetLike,
    batch_orbitals: networks.OrbitalFnLike,
    network_options: networks.BaseNetworkOptions,
    key: chex.PRNGKey,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Callable[[int, float], None] | None = None,
    scf_fraction: float = 0.0,
):
  """Runs HF pretraining on a mesh-sharded global walker batch."""
  is_host_0 = jax.process_index() == 0
  mesh = constants.make_global_mesh()
  batch_pspec = jax.sharding.PartitionSpec(constants.GLOBAL_BATCH_AXIS)
  replicated = constants.replicated_sharding(mesh)
  batch_sharding = constants.batch_sharding(mesh)
  data_sharding = networks.FermiNetData(
      positions=batch_sharding,
      spins=batch_sharding,
      atoms=batch_sharding,
      charges=batch_sharding,
  )

  if is_host_0:
    logging.info(
        'Initializing mesh pretraining on %d global devices.',
        jax.device_count(),
    )
    logging.info('Materializing host-local walker state as global arrays.')

  global_positions = constants.host_local_to_global_array(
      positions, mesh, batch_pspec
  )
  if is_host_0:
    logging.info('Global walker positions ready.')
  global_spins = constants.host_local_to_global_array(spins, mesh, batch_pspec)
  if is_host_0:
    logging.info('Global walker spins ready.')
  global_atoms = constants.host_local_to_global_array(atoms, mesh, batch_pspec)
  if is_host_0:
    logging.info('Global atom batches ready.')
  global_charges = constants.host_local_to_global_array(
      charges, mesh, batch_pspec
  )
  if is_host_0:
    logging.info('Global charge batches ready.')

  optimizer = optax.adam(3.e-4)
  opt_state = optimizer.init(params)
  global_params = params
  global_key = key
  if is_host_0:
    logging.info('Host-local optimizer state ready.')
    logging.info('Global pretraining arrays/materialized state ready.')

  full_det = network_options.full_det
  if scf_fraction > 1 or scf_fraction < 0:
    raise ValueError('scf_fraction must be in between 0 and 1, inclusive.')

  scf_network = lambda fn, x: fn(x, electrons)[1]

  if scf_fraction < 1e-6:
    def mcmc_network(full_params, pos, walker_spins, walker_atoms, walker_charges):
      return batch_network(
          full_params['ferminet'],
          pos,
          walker_spins,
          walker_atoms,
          walker_charges,
      )
  elif scf_fraction > 0.999999:
    def mcmc_network(full_params, pos, walker_spins, walker_atoms, walker_charges):
      del walker_spins, walker_atoms, walker_charges
      return scf_network(full_params['scf'].eval_slater, pos)
  else:
    def mcmc_network(full_params, pos, walker_spins, walker_atoms, walker_charges):
      log_ferminet = batch_network(
          full_params['ferminet'],
          pos,
          walker_spins,
          walker_atoms,
          walker_charges,
      )
      log_scf = scf_network(full_params['scf'].eval_slater, pos)
      return (1 - scf_fraction) * log_ferminet + scf_fraction * log_scf

  global_mcmc_step = mcmc.make_mcmc_step(
      mcmc_network, batch_per_device=batch_size, steps=1
  )

  def loss_fn(
      current_params: networks.ParamTree,
      data: networks.FermiNetData,
  ) -> jnp.ndarray:
    target = scf_approx.eval_orbitals(data.positions, electrons)
    orbitals = batch_orbitals(
        current_params, data.positions, data.spins, data.atoms, data.charges
    )
    cnorm = lambda x, y: (x - y) ** 2
    if full_det:
      dims = target[0].shape[:-2]
      na = target[0].shape[-2]
      nb = target[1].shape[-2]
      target_full = jnp.concatenate(
          (
              jnp.concatenate(
                  (target[0], jnp.zeros(dims + (na, nb))), axis=-1
              ),
              jnp.concatenate(
                  (jnp.zeros(dims + (nb, na)), target[1]), axis=-1
              ),
          ),
          axis=-2,
      )
      return jnp.mean(cnorm(target_full[:, None, ...], orbitals[0]))
    return jnp.array([
        jnp.mean(cnorm(t[:, None, ...], o))
        for t, o in zip(target, orbitals)
    ]).sum()

  @functools.partial(
      jax.jit,
      in_shardings=(replicated, replicated, data_sharding, replicated),
      out_shardings=(replicated, replicated, data_sharding, replicated, replicated),
  )
  def pretrain_step(
      current_params,
      current_opt_state,
      data,
      step_key,
  ):
    loss_val, grads = jax.value_and_grad(loss_fn)(current_params, data)
    updates, next_opt_state = optimizer.update(
        grads, current_opt_state, current_params
    )
    next_params = optax.apply_updates(current_params, updates)
    full_params = {'ferminet': next_params, 'scf': scf_approx}
    next_data, pmove = global_mcmc_step(
        full_params, data, step_key, width=0.02
    )
    return next_params, next_opt_state, next_data, loss_val, pmove

  data = networks.FermiNetData(
      positions=global_positions,
      spins=global_spins,
      atoms=global_atoms,
      charges=global_charges,
  )
  if is_host_0:
    logging.info(
        'Starting mesh Hartree-Fock pretraining for %d iterations.', iterations
    )
  for t in range(iterations):
    if t == 0 and is_host_0:
      logging.info('Compiling/executing first mesh pretrain step.')
    global_key, step_key = jax.random.split(global_key)
    global_params, opt_state, data, loss, pmove = pretrain_step(
        global_params, opt_state, data, step_key
    )
    if is_host_0:
      logging.info('Pretrain iter %05d: %g %g', t, loss, pmove)
    if logger and is_host_0:
      logger(t, float(loss))

  host_positions = constants.global_array_to_host_local_array(
      data.positions, mesh, batch_pspec
  )
  host_params = jax.tree_util.tree_map(np.asarray, global_params)
  return host_params, host_positions
