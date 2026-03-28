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

"""Core training loop for neural QMC in JAX."""

from __future__ import annotations

import importlib
import os
import time
from typing import Sequence, Tuple
import zlib

from absl import logging
import wandb
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet.steps import (
    make_eval_step,
    make_kfac_training_step,
    make_loss_step,
    make_opt_update_step,
    make_training_step,
    null_update,
)
from ferminet.utils import precision as precision_utils
from ferminet.utils import sanity_checks
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import writers
from ferminet.utils import optim_logging
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax


def _make_sharded_key(rng):
  rng = jax.random.fold_in(rng, jax.process_index())
  rng = jax.random.split(rng, jax.local_device_count())
  return constants.broadcast(rng)


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    given_atomic_spin_configs: Sequence[Tuple[int, int]] | None = None,
    max_iter: int = 10_000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.
    given_atomic_spin_configs: optional per-atom spin assignments to use for
      initialization when the neutral atomic defaults do not realize the
      requested global spin partition.
    max_iter: maximum number of iterations to try to find a valid initial
        electron configuration for each atom. If reached, all electrons are
        initialised from a Gaussian distribution centred on the origin.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
  niter = 0
  total_electrons = sum(atom.charge for atom in molecule)
  if given_atomic_spin_configs is not None:
    atomic_spin_configs = list(given_atomic_spin_configs)
  elif total_electrons != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta)
        for atom in molecule
    ]

  assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
  while (
      tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons
      and niter < max_iter
  ):
    i = np.random.randint(len(atomic_spin_configs))
    nalpha, nbeta = atomic_spin_configs[i]
    atomic_spin_configs[i] = nbeta, nalpha
    niter += 1

  if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
    electron_positions = []
    for i in range(2):
      for j in range(len(molecule)):
        atom_position = jnp.asarray(molecule[j].coords)
        electron_positions.append(
            jnp.tile(atom_position, atomic_spin_configs[j][i]))
    electron_positions = jnp.concatenate(electron_positions)
  else:
    logging.warning(
        'Failed to find a valid initial electron configuration after %i'
        ' iterations. Initializing all electrons from a Gaussian distribution'
        ' centred on the origin. This might require increasing the number of'
        ' iterations used for pretraining and MCMC burn-in. Consider'
        ' implementing a custom initialisation.',
        niter,
    )
    electron_positions = jnp.zeros(shape=(3*sum(electrons),))

  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins


def _run_save_path(save_path: str) -> str:
  """Returns the run output directory, creating it if needed."""
  return checkpoint.create_save_path(save_path)


def _shared_run_seed(
    save_path: str,
    *,
    deterministic: bool,
    num_hosts: int,
) -> int:
  """Returns a host-consistent seed without cross-host device collectives."""
  if deterministic:
    return 23
  if num_hosts == 1:
    return int(1e6 * time.time())
  seed_source = ':'.join((
      os.environ.get('SLURM_JOB_ID', '0'),
      os.environ.get('SLURM_STEP_ID', '0'),
      save_path,
  ))
  return zlib.crc32(seed_source.encode('utf-8')) & 0x7FFFFFFF

def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Verify precision configuration
  precision_utils.verify_precision_config()
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  is_host_0 = jax.process_index() == 0
  if is_host_0:
    logging.info('Starting QMC with %i XLA devices per host '
                 'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  total_host_batch_size = host_batch_size
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)
  # Index into pmap output arrays to get the local device's shard.
  # In multi-host JAX, pmap outputs are globally-sharded; indexing with [0]
  # on a non-primary host accesses a remote shard and crashes.
  # Each host must use its own local shard index instead.
  local_pmap_idx = jax.process_index() * num_devices
  run_save_path = _run_save_path(cfg.log.save_path)
  with cfg.ignore_type():
    cfg.log.save_path = run_save_path

  # Initialize wandb if enabled — only on host 0 to avoid duplicate runs.
  if is_host_0 and cfg.log.get('use_wandb', False):
    wandb_config = cfg.log.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'ferminet'),
        name=wandb_config.get('name', None),
        config=dict(cfg),
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        dir=run_save_path,
    )

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  # Generate atomic configurations for each walker.
  # Use broadcast_all_local_devices (pmap-based) for ALL arrays to ensure
  # consistent PmapSharding. Mixing device_put_replicated (from
  # replicate_all_local_devices) with PmapSharding arrays in the same pmap
  # call causes CopyArrays errors in multi-host JAX.
  batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
  batch_atoms = constants.broadcast(
      jnp.tile(batch_atoms[None], [num_devices] + [1] * batch_atoms.ndim))
  batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
  batch_charges = constants.broadcast(
      jnp.tile(batch_charges[None], [num_devices] + [1] * batch_charges.ndim))

  seed = _shared_run_seed(
      cfg.log.save_path,
      deterministic=cfg.debug.deterministic,
      num_hosts=num_hosts,
  )
  if is_host_0:
    logging.info('Using shared run seed %d.', seed)
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0:
    if is_host_0:
      logging.info('Running Hartree-Fock setup on each host.')
    # All hosts need the PySCF object for orbital evaluation.
    # Suppress PySCF stdout on non-primary hosts to avoid duplicate logs.
    if not is_host_0:
      import contextlib, io
      with contextlib.redirect_stdout(io.StringIO()):
        hartree_fock = pretrain.get_hf(
            pyscf_mol=cfg.system.get('pyscf_mol'),
            molecule=cfg.system.molecule,
            nspins=nspins,
            restricted=False,
            basis=cfg.pretrain.basis,
            )
    else:
      hartree_fock = pretrain.get_hf(
          pyscf_mol=cfg.system.get('pyscf_mol'),
          molecule=cfg.system.molecule,
          nspins=nspins,
          restricted=False,
          basis=cfg.pretrain.basis,
          )
    if is_host_0:
      logging.info('Hartree-Fock setup complete on host 0.')
  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()

  network = networks.make_psiformer(
      nspins,
      charges,
      ndim=cfg.system.ndim,
      determinants=cfg.network.determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=cfg.network.get('jastrow', 'default'),
      bias_orbitals=cfg.network.bias_orbitals,
      rescale_inputs=cfg.network.get('rescale_inputs', False),
      **cfg.network.psiformer,
  )
  key, subkey = jax.random.split(key)
  params = network.init(subkey)

  signed_network = network.apply
  # Often just need log|psi(x)|.
  logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(
      logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )  # batched network

  key, subkey = jax.random.split(key)
  # Make sure data on each host is initialized differently.
  subkey = jax.random.fold_in(subkey, jax.process_index())
  host_positions, host_spins = init_electrons(
      subkey,
      cfg.system.molecule,
      cfg.system.electrons,
      batch_size=total_host_batch_size,
      init_width=cfg.mcmc.init_width,
      given_atomic_spin_configs=(
          cfg.system.atom_spin_configs
          if hasattr(cfg.system, 'atom_spin_configs')
          else None
      ),
  )
  host_batch_atoms = jnp.tile(atoms[None, ...], [host_batch_size, 1, 1])
  host_batch_charges = jnp.tile(charges[None, ...], [host_batch_size, 1])
  pretrain_spins_local = host_spins[0]
  t_init = 0

  # Restart: load params, optimizer state, mcmc_width from checkpoint
  restart_opt_state = None
  restart_mcmc_width = None
  if cfg.get('restart', {}).get('path', ''):
    restart_ckpt_path = cfg.restart.path
    logging.info('Restart: loading checkpoint %s', restart_ckpt_path)
    (t_init,
     _restart_data,
     params,
     restart_opt_state,
     restart_mcmc_width) = checkpoint.restore(restart_ckpt_path, host_batch_size)
    logging.info('Restart: loaded params from checkpoint, resuming at step %d',
                 t_init)

  # Set up logging
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']

  # Pretraining to match Hartree-Fock

  if (
      t_init == 0
      and cfg.pretrain.method == 'hf'
      and cfg.pretrain.iterations > 0
  ):
    sanity_checks.assert_global_device_topology('Pretraining startup')
    sanity_checks.assert_distinct_host_positions(
        'Pretraining startup', host_positions
    )
    if is_host_0:
      logging.info('Starting Hartree-Fock pretraining on %d hosts x %d local '
                   'devices (device_batch_size=%d).',
                   num_hosts, num_devices, device_batch_size)
    batch_orbitals = jax.vmap(
        network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    if jax.device_count() > 1 and num_devices > 1:
      params, host_positions = pretrain.pretrain_hartree_fock_global(
          params=params,
          positions=host_positions,
          spins=host_spins,
          atoms=host_batch_atoms,
          charges=host_batch_charges,
          batch_network=batch_network,
          batch_orbitals=batch_orbitals,
          network_options=network.options,
          key=key,
          electrons=cfg.system.electrons,
          scf_approx=hartree_fock,
          iterations=cfg.pretrain.iterations,
          batch_size=cfg.batch_size,
          scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
          optimizer_name=cfg.pretrain.get('optimizer', 'lamb'),
          pretrain_lr=cfg.pretrain.get('lr', 3.e-4),
      )
    else:
      sharded_key = _make_sharded_key(key)
      params = constants.broadcast(
          jax.tree_util.tree_map(
              lambda x: jnp.broadcast_to(x[None], (num_devices,) + x.shape),
              params))
      pos = jnp.reshape(host_positions, data_shape + (-1,))
      pos = constants.broadcast(pos)
      spins = jnp.reshape(host_spins, data_shape + (-1,))
      spins = constants.broadcast(spins)
      data = networks.FermiNetData(
          positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
      )
      sharded_key, subkeys = constants.p_split(sharded_key)
      params, data.positions = pretrain.pretrain_hartree_fock(
          params=params,
          positions=data.positions,
          spins=pretrain_spins_local,
          atoms=data.atoms,
          charges=data.charges,
          batch_network=batch_network,
          batch_orbitals=batch_orbitals,
          network_options=network.options,
          sharded_key=subkeys,
          electrons=cfg.system.electrons,
          scf_approx=hartree_fock,
          iterations=cfg.pretrain.iterations,
          batch_size=device_batch_size,
          scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
          optimizer_name=cfg.pretrain.get('optimizer', 'lamb'),
          pretrain_lr=cfg.pretrain.get('lr', 3.e-4),
      )
    if is_host_0:
      logging.info('Hartree-Fock pretraining complete.')

  params = constants.broadcast(
      jax.tree_util.tree_map(
          lambda x: jnp.broadcast_to(x[None], (num_devices,) + x.shape),
          params))
  pos = jnp.reshape(host_positions, data_shape + (-1,))
  pos = constants.broadcast(pos)
  spins = jnp.reshape(host_spins, data_shape + (-1,))
  spins = constants.broadcast(spins)
  data = networks.FermiNetData(
      positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
  )

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = _make_sharded_key(key)

  sanity_checks.assert_global_device_topology('Training startup')
  sanity_checks.assert_distinct_host_positions('Training startup', host_positions)

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      device_batch_size,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      blocks=cfg.mcmc.blocks,
      algorithm=cfg.mcmc.algorithm,
  )
  # Construct loss and optimizer
  laplacian_method = cfg.optim.get('laplacian', 'default')
  if cfg.system.get('make_local_energy_fn', ''):
    raise NotImplementedError(
        'Custom local energy functions are no longer supported. '
        'Remove `system.make_local_energy_fn` from your config.'
    )
  local_energy_fn = hamiltonian.local_energy(
      f=signed_network,
      charges=charges,
      nspins=nspins,
      use_scan=False,
      laplacian_method=laplacian_method)

  local_energy = local_energy_fn

  evaluate_loss = qmc_loss_functions.make_loss(
      logabs_network,
      local_energy,
      clip_local_energy=cfg.optim.clip_local_energy,
      clip_from_median=cfg.optim.clip_median,
      center_at_clipped_energy=cfg.optim.center_at_clip,
      max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
  )

  if cfg.optim.kfac.momentum > 0.0:
    # Build Nesterov multi-stage schedules
    # learning_rate_schedule, momentum_schedule = _make_nesterov_multistage_schedules(
    #     L=5.0, mu=5e-4, p=3, max_stage=8)
    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      return cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay) 
    def momentum_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      # Linear warmup from 0 -> final momentum over warmup_steps, then constant
      final_mu = jnp.asarray(cfg.optim.kfac.momentum, dtype=jnp.asarray(1.0).dtype)
      t_float = t_.astype(jnp.asarray(1.0).dtype)
      warmup_steps = jnp.asarray(cfg.optim.kfac.momentum_warmup_steps, dtype=t_float.dtype)
      frac = jnp.clip(t_float / warmup_steps, 0.0, 1.0)
      return frac * final_mu
  else:
    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      return cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)
    def momentum_schedule(t_: jnp.ndarray) -> jnp.ndarray:
      return jnp.asarray(0.0)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        momentum_schedule=momentum_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        curvature_update_period=cfg.optim.kfac.cov_update_every,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        use_nesterov=cfg.optim.kfac.nesterov,
        include_norms_in_stats=True,
        constrain_l2=cfg.optim.kfac.constrain_l2,
        # debug=True
    )
    if is_host_0:
      logging.info('Initializing KFAC optimizer state.')
    sharded_key, subkeys = constants.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    if restart_opt_state is not None:
      opt_state = restart_opt_state
    if is_host_0:
      logging.info('KFAC optimizer state initialized.')
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = constants.pmap(optimizer.init)(params)
    if restart_opt_state is not None:
      opt_state = tuple(restart_opt_state)
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
        reset_if_nan=cfg.optim.reset_if_nan)
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer,
        reset_if_nan=cfg.optim.reset_if_nan)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  # Build evaluation step (independent of optimizer type)
  eval_step = make_eval_step(mcmc_step=mcmc_step, evaluate_loss=evaluate_loss)

  mcmc_width_value = cfg.mcmc.move_width
  mcmc_width = constants.broadcast(
      jnp.full((num_devices,), mcmc_width_value)
  )
  if restart_mcmc_width is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        restart_mcmc_width[0])
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    if is_host_0:
      logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = constants.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    if is_host_0:
      logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = constants.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    if is_host_0:
      logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  if cfg.get('restart', {}).get('path', ''):
    restart_burn_in_steps = cfg.restart.get('burn_in', 1000)
    logging.info('Restart: burning in MCMC chain for %d steps',
                 restart_burn_in_steps)
    burn_in_step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=null_update)
    for t in range(restart_burn_in_steps):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data, params, state=None, key=subkeys, mcmc_width=mcmc_width)
    logging.info('Restart: completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Restart: initial energy after burn-in: %03.4f E_h',
                 initial_energy[0])

  weighted_stats = None

  if writer_manager is None:
    if is_host_0:
      writer_manager = writers.Writer(
          name='train_stats',
          schema=train_schema,
          directory=run_save_path,
          iteration_key=None,
          log=False)
    else:
      writer_manager = writers.NoOpWriter()
  with writer_manager as writer:
    # Log dtypes of key data structures — only on host 0.
    if is_host_0:
      precision_utils.log_tree_dtypes(params, "params")
      precision_utils.log_tree_dtypes(data, "data")

    # Main training loop
    num_resets = 0  # used if reset_if_nan is true
    completed_step = t_init
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = constants.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove, optimizer_stats = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width)

      # due to pmean, loss, and pmove should be the same across
      # devices. Use local shard index (not [0]) to avoid accessing
      # a remote shard in multi-host JAX.
      loss = loss[local_pmap_idx]
      valid_samples = (
          aux_data.valid_samples[local_pmap_idx]
          if aux_data.valid_samples is not None
          else None
      )
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)
      pmove = pmove[local_pmap_idx]

      # Update MCMC move width
      if cfg.mcmc.get('algorithm', 'mh') == 'mala':
        mcmc_width, pmoves = mcmc.update_mala_width(
            t,
            mcmc_width,
            cfg.mcmc.adapt_frequency,
            pmove,
            pmoves,
            target_acceptance=cfg.mcmc.get('mala_target_acceptance', 0.57),
        )
      else:
        mcmc_width, pmoves = mcmc.update_mcmc_width(
            t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        try:
          chex.assert_tree_all_finite(tree)
          num_resets = 0  # Reset counter if check passes
        except AssertionError as e:
          if cfg.optim.reset_if_nan:  # Allow a certain number of NaNs
            num_resets += 1
            if num_resets > 100:
              raise e
          else:
            raise e

      completed_step = t + 1

      # Logging — only on host 0.
      if is_host_0 and t % cfg.log.stats_frequency == 0:
        logging_str = ('Step %05d: '
                       '%03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f')
        logging_args = t, loss, weighted_stats.variance, pmove
        writer_kwargs = {
            'step': t,
            'energy': np.asarray(loss),
            'ewmean': np.asarray(weighted_stats.mean),
            'ewvar': np.asarray(weighted_stats.variance),
            'pmove': np.asarray(pmove),
        }
        logging.info(logging_str, *logging_args)
        writer.write(t, **writer_kwargs)
        
        # Log to wandb if enabled — only on host 0.
        if is_host_0 and cfg.log.get('use_wandb', False):
          current_lr = learning_rate_schedule(jnp.asarray(t))
          current_mom = momentum_schedule(jnp.asarray(t))

          wandb_log = {
              'step': t,
              'train/energy': float(loss),
              'train/ewmean': float(weighted_stats.mean),
              'train/ewvar': float(weighted_stats.variance),
              'train/pmove': float(pmove),
              'train/learning_rate': float(current_lr),
              'train/momentum': float(current_mom),
          }
          if valid_samples is not None:
            wandb_log['train/valid_samples'] = float(valid_samples)

          # Add optimizer norm statistics if available
          if optimizer_stats is not None:
            wandb_log['train/param_norm'] = float(optimizer_stats.param_norm[0])
            wandb_log['train/grad_norm'] = float(optimizer_stats.grad_norm[0])
            wandb_log['train/precon_grad_norm'] = float(optimizer_stats.precon_grad_norm[0])
            wandb_log['train/fisher_norm'] = float(optimizer_stats.fisher_norm[0])
            wandb_log['train/update_norm'] = float(optimizer_stats.update_norm[0])

          wandb.log(wandb_log)

          if optimizer_stats is not None and optimizer_stats.precon_grad_tree is not None:
            optim_logging.log_optim_rms_to_wandb(
                optimizer_stats.precon_grad_tree, t)
          optim_logging.log_param_stats_to_wandb(params, t)

      # Checkpoint saving (step-based) — only on host 0.
      if is_host_0 and (
          cfg.log.save_frequency > 0
          and completed_step % cfg.log.save_frequency == 0):
        checkpoint.save(
            run_save_path, completed_step, data, params, opt_state, mcmc_width)

      # Evaluation (checked after each training step)
      if cfg.get('eval', {}).get('interval', 0) and ((t + 1) % cfg.eval.interval == 0):
        num_eval_iters = int(cfg.eval.get('iterations', 0) or 0)
        if num_eval_iters > 0:
          eval_losses = []
          eval_pmoves = []
          eval_data = data
          # Run evaluation iterations (no parameter updates)
          # eval_step is pmapped so must run on ALL hosts.
          for _ in range(num_eval_iters):
            sharded_key, subkeys = constants.p_split(sharded_key)
            eval_data, _, _, eval_loss, _, eval_pmove, _ = eval_step(
                eval_data,
                params,
                state=None,
                key=subkeys,
                mcmc_width=mcmc_width)
            if is_host_0:
              eval_losses.append(np.asarray(eval_loss[0]))
              eval_pmoves.append(np.asarray(eval_pmove[0]))

          if is_host_0:
            eval_losses = np.asarray(eval_losses)
            eval_pmoves = np.asarray(eval_pmoves)
            eval_mean = float(np.mean(eval_losses))
            eval_var = float(np.var(eval_losses))
            eval_pmove_mean = float(np.mean(eval_pmoves))

            # Log to wandb if enabled.
            if cfg.log.get('use_wandb', False):
              wandb.log({
                  'step': t + 1,
                  'eval/energy': eval_mean,
                  'eval/ewvar': eval_var,
                  'eval/pmove': eval_pmove_mean,
              })

    # Save final checkpoint — only on host 0.
    if is_host_0:
      checkpoint.save(
          run_save_path, completed_step, data, params, opt_state, mcmc_width)

    # Finish wandb run if enabled — only on host 0.
    if is_host_0 and cfg.log.get('use_wandb', False):
      wandb.finish()
