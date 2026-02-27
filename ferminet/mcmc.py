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

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from ferminet import constants
from ferminet import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom


def _diffs_to_nearest_nuc(elec_pos, atoms_pos):
  """Vector to nearest nucleus for each electron.

  Args:
    elec_pos: (batch, nelec, ndim)
    atoms_pos: (batch, natoms, ndim) or (natoms, ndim)

  Returns:
    z_vec: (batch, nelec, ndim) vector from electron to nearest nucleus
    z2_min: (batch, nelec) squared distance to nearest nucleus
    idx: (batch, nelec) index of nearest nucleus
  """
  if atoms_pos.ndim == 2:
    atoms_pos = jnp.broadcast_to(atoms_pos[None, ...],
                                 (elec_pos.shape[0],) + atoms_pos.shape)
  # diffs: (b, ne, na, ndim)
  diffs = elec_pos[:, :, None, :] - atoms_pos[:, None, :, :]
  z2 = jnp.sum(diffs**2, axis=-1)  # (b, ne, na)
  idx = jnp.argmin(z2, axis=-1)    # (b, ne)
  # Gather nearest vectors
  idx_exp = idx[..., None, None]   # (b, ne, 1, 1)
  z_vec = jnp.take_along_axis(diffs, idx_exp, axis=2)[..., 0, :]  # (b, ne, ndim)
  z2_min = jnp.take_along_axis(z2, idx[..., None], axis=-1)[..., 0]  # (b, ne)
  return z_vec, z2_min, idx


def _crossover_parameter(z_vec, force, charge_nn):
  """Compute crossover parameter used to smooth the force near cusps.

  Args:
    z_vec: (batch, nelec, ndim) vector to nearest nucleus
    force: (batch, nelec, ndim) raw score/force
    charge_nn: (batch, nelec) charge of nearest nucleus

  Returns:
    a: (batch, nelec) crossover parameter in [0, 2]
  """
  eps = jnp.finfo(force.dtype).eps
  z_norm = jnp.clip(jnp.linalg.norm(z_vec, axis=-1, keepdims=True), eps, None)
  f_norm = jnp.clip(jnp.linalg.norm(force, axis=-1, keepdims=True), eps, None)
  z_unit = z_vec / z_norm
  f_unit = force / f_norm
  z2 = jnp.sum(z_vec**2, axis=-1)  # (b, ne)
  z2_scaled = (charge_nn**2) * z2
  # Blend alignment with distance-based term
  align = (1.0 + jnp.sum(f_unit * z_unit, axis=-1)) / 2.0
  return align + z2_scaled / (10.0 * (4.0 + z2_scaled))


def _clean_force(force, elec_pos, atoms_pos, charges, *, tau, max_force_norm=jnp.inf):
  """Clean the score/force to avoid electron-nucleus crossing.

  Args:
    force: (batch, nelec, ndim) gradient of log |psi|^2 (score)
    elec_pos: (batch, nelec, ndim)
    atoms_pos: (batch, natoms, ndim) or (natoms, ndim)
    charges: (batch, natoms) or (natoms,)
    tau: scalar step-size used for deterministic drift in proposal
    max_force_norm: optional cap per electron (float)

  Returns:
    cleaned force with same shape as `force`.
  """
  bsize = elec_pos.shape[0]
  if atoms_pos.ndim == 2:
    atoms_pos = jnp.broadcast_to(atoms_pos[None, ...], (bsize,) + atoms_pos.shape)
  if charges.ndim == 1:
    charges = jnp.broadcast_to(charges[None, ...], (bsize,) + charges.shape)

  z_vec, z2_min, idx = _diffs_to_nearest_nuc(elec_pos, atoms_pos)
  charge_nn = jnp.take_along_axis(charges, idx, axis=-1)  # (b, ne)

  a = _crossover_parameter(z_vec, force, charge_nn)
  # Smooth the force magnitude
  av2tau = a * jnp.sum(force**2, axis=-1) * tau  # (b, ne)
  factor_mag = 2.0 / (jnp.sqrt(1.0 + 2.0 * av2tau) + 1.0)  # (b, ne)
  cleaned = factor_mag[..., None] * force

  # Final capping: (1) optional absolute cap, (2) prevent crossing nearest nucleus
  eps = jnp.finfo(elec_pos.dtype).eps
  norm_clean = jnp.clip(jnp.linalg.norm(cleaned, axis=-1), eps, None)  # (b, ne)
  one = jnp.ones_like(norm_clean)
  cap_abs = max_force_norm / norm_clean
  cap_cross = jnp.sqrt(z2_min) / (tau * norm_clean)
  norm_factor = jnp.minimum(one, jnp.minimum(cap_abs, cap_cross))
  return cleaned * norm_factor[..., None]


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, key, lp_new, num_accepts


def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
      Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
      N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
      the i-th electron and the atoms, otherwise the move proposal drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  lp_2 = 2.0 * f(
      params, x2, data.spins, data.atoms, data.charges
  )  # log prob of proposal
  ratio = lp_2 - lp_1
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def mala_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    blocks=1,
    i=0,
):
  """Performs one MALA step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MCMC move proposals accepted.
    stddev: width of Gaussian move proposal (also diffusion scale in MALA).
    atoms: Ignored (symmetric Gaussian only).
    ndim: dimensionality of system.
    blocks: Ignored; only all-electron move is supported.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # all-electron proposal only
  key, subkey = jax.random.split(key)
  x1 = data.positions
  ndim = 3

  # Define per-sample log-probability (2.0 * f) for forward-mode score.
  # Each sample uses its own spins/atoms/charges.
  def logprob_single(x_flat, spins_i, atoms_i, charges_i):
    return 2.0 * f(
        params,
        x_flat[None, :],
        spins_i[None, ...],
        atoms_i[None, ...],
        charges_i[None, ...],
    )[0]

  # Use reverse-mode gradients (more efficient for high-dimensional inputs).
  score1 = jax.vmap(jax.grad(logprob_single))(
      x1, data.spins, data.atoms, data.charges
  )

  # Clean the score to avoid electron-nucleus crossing during the drift step.
  # Use tau equal to the drift coefficient applied to the score in the proposal.
  drift_scale = 0.5 * (stddev ** 2)
  # Reshape flat (batch, nelec*ndim) -> (batch, nelec, ndim)
  bsz, dim = x1.shape
  nelec = dim // ndim
  x1_e = jnp.reshape(x1, (bsz, nelec, ndim))
  s1_e = jnp.reshape(score1, (bsz, nelec, ndim))
  cleaned1 = _clean_force(
      s1_e, x1_e, data.atoms, data.charges, tau=drift_scale, max_force_norm=jnp.inf
  )
  mu1 = jnp.reshape(x1_e + drift_scale * cleaned1, (bsz, dim))
  noise = stddev * jax.random.normal(subkey, shape=x1.shape)
  x2 = mu1 + noise

  # Compute reverse proposal term using score at the proposal location.
  # Fuse value and gradient to avoid an extra forward pass.
  lp_2, score2 = jax.vmap(jax.value_and_grad(logprob_single))(
      x2, data.spins, data.atoms, data.charges
  )
  x2_e = jnp.reshape(x2, (bsz, nelec, ndim))
  s2_e = jnp.reshape(score2, (bsz, nelec, ndim))
  cleaned2 = _clean_force(
      s2_e, x2_e, data.atoms, data.charges, tau=drift_scale, max_force_norm=jnp.inf
  )
  mu2 = jnp.reshape(x2_e + drift_scale * cleaned2, (bsz, dim))

  # Use Gaussian log-probabilities for forward/reverse proposals.
  n = x1.shape[0]
  x1r = jnp.reshape(x1, [n, -1, 1, ndim])
  x2r = jnp.reshape(x2, [n, -1, 1, ndim])
  mu1r = jnp.reshape(mu1, [n, -1, 1, ndim])
  mu2r = jnp.reshape(mu2, [n, -1, 1, ndim])
  sigma = jnp.ones_like(x1r) * stddev

  lq_1 = _log_prob_gaussian(x2r, mu1r, sigma)  # forward q(x2|x1)
  lq_2 = _log_prob_gaussian(x1r, mu2r, sigma)  # reverse q(x1|x2)

  ratio = lp_2 + lq_2 - lp_1 - lq_1

  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def mala_sign_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    blocks=1,
    i=0,
):
  """Performs one MALA step using sign of the input gradient (SignGD-style).
  
  Same as `mala_update` except the drift uses sign(score) before cleaning.
  """
  del i, blocks  # all-electron proposal only
  key, subkey = jax.random.split(key)
  x1 = data.positions
  ndim = 3

  def logprob_single(x_flat, spins_i, atoms_i, charges_i):
    return 2.0 * f(
        params,
        x_flat[None, :],
        spins_i[None, ...],
        atoms_i[None, ...],
        charges_i[None, ...],
    )[0]

  # Reverse-mode score at x1
  score1 = jax.vmap(jax.grad(logprob_single))(
      x1, data.spins, data.atoms, data.charges
  )

  drift_scale = 0.5 * (stddev ** 2)
  # Reshape flat -> electron form; apply sign to the score
  bsz, dim = x1.shape
  nelec = dim // ndim
  x1_e = jnp.reshape(x1, (bsz, nelec, ndim))
  s1_e = jnp.reshape(score1, (bsz, nelec, ndim))
  s1_e = jnp.sign(s1_e)
  cleaned1 = _clean_force(
      s1_e, x1_e, data.atoms, data.charges, tau=drift_scale, max_force_norm=jnp.inf
  )
  mu1 = jnp.reshape(x1_e + drift_scale * cleaned1, (bsz, dim))
  noise = stddev * jax.random.normal(subkey, shape=x1.shape)
  x2 = mu1 + noise

  # Value and score at proposal; use sign(score2) for reverse proposal mean
  lp_2, score2 = jax.vmap(jax.value_and_grad(logprob_single))(
      x2, data.spins, data.atoms, data.charges
  )
  x2_e = jnp.reshape(x2, (bsz, nelec, ndim))
  s2_e = jnp.reshape(score2, (bsz, nelec, ndim))
  s2_e = jnp.sign(s2_e)
  cleaned2 = _clean_force(
      s2_e, x2_e, data.atoms, data.charges, tau=drift_scale, max_force_norm=jnp.inf
  )
  mu2 = jnp.reshape(x2_e + drift_scale * cleaned2, (bsz, dim))

  # Gaussian terms
  n = x1.shape[0]
  x1r = jnp.reshape(x1, [n, -1, 1, ndim])
  x2r = jnp.reshape(x2, [n, -1, 1, ndim])
  mu1r = jnp.reshape(mu1, [n, -1, 1, ndim])
  mu2r = jnp.reshape(mu2, [n, -1, 1, ndim])
  sigma = jnp.ones_like(x1r) * stddev

  lq_1 = _log_prob_gaussian(x2r, mu1r, sigma)  # forward q(x2|x1)
  lq_2 = _log_prob_gaussian(x1r, mu2r, sigma)  # reverse q(x1|x2)
  ratio = lp_2 + lq_2 - lp_1 - lq_1

  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   ndim=3,
                   blocks=1,
                   algorithm: str = 'mh'):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.
    algorithm: Either 'mh' (default) for Metropolis-Hastings or 'mala' for
      Metropolis Adjusted Langevin Algorithm. MALA currently only supports
      all-electron moves (blocks==1) and uses a symmetric Gaussian proposal.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  if algorithm == 'mala':
    if blocks > 1:
      raise NotImplementedError('MALA not implemented for block updates; use blocks==1')
    inner_fun = mala_update
  elif algorithm == 'mala_sign':
    if blocks > 1:
      raise NotImplementedError('MALA (sign) not implemented for block updates; use blocks==1')
    inner_fun = mala_sign_update
  else:
    inner_fun = mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          blocks=blocks,
          i=i)

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves


def update_mala_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    target_acceptance: float = 0.57,
    min_effective_acceptance: float = 0.05,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the MALA step size to target a desired acceptance rate.

  Matches the reference proportional scaling behaviour: scale the proposal width
  by (mean_acceptance / target), with a small lower bound to avoid division by
  very small acceptances. Uses the same `adapt_frequency` and `pmoves` buffer
  convention as `update_mcmc_width` for consistency.

  Args:
    t: Current step.
    width: Current MCMC width (stddev).
    adapt_frequency: Number of iterations between width updates.
    pmove: Acceptance ratio in the last step.
    pmoves: Buffer of recent acceptance ratios of length `adapt_frequency`.
    target_acceptance: Target acceptance rate (default 0.57).
    min_effective_acceptance: Lower bound for acceptance used in scaling.

  Returns:
    (width, pmoves): Updated width and buffer.
  """
  t_since_update = t % adapt_frequency
  pmoves[t_since_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_update == 0:
    mean_acc = np.mean(pmoves)
    eff_acc = max(mean_acc, min_effective_acceptance)
    scale = eff_acc / target_acceptance
    width *= scale
  return width, pmoves
