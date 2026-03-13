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

"""Training and evaluation step factories for neural QMC."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import chex
from ferminet import constants
from ferminet import loss as qmc_loss_functions
from ferminet import networks
import jax
import jax.numpy as jnp
import kfac_jax
import optax
from typing_extensions import Protocol


@dataclass
class OptimizerStats:
  """Container for additional optimizer statistics."""
  param_norm: Optional[jnp.ndarray] = None
  grad_norm: Optional[jnp.ndarray] = None
  precon_grad_norm: Optional[jnp.ndarray] = None
  update_norm: Optional[jnp.ndarray] = None
  precon_grad_tree: Optional[networks.ParamTree] = None


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[OptimizerState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
    Optional[OptimizerStats],
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove, optimizer_stats).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
        optimizer_stats: Optional optimizer statistics (e.g., norms for KFAC).
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove, None

  return step


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_damping = constants.broadcast(
      jnp.full((jax.local_device_count(),), damping))
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))
  # Track the KFAC step counter on the host side to avoid
  # kfac_jax's get_first(step_counter) which does value[0] on a
  # PmapSharding array — crashes on non-primary hosts in multi-host JAX.
  kfac_step_counter = [0]

  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    mcmc_keys, loss_keys = constants.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    if reset_if_nan:
      old_params = copy_tree(params)
      old_state = copy_tree(state)

    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        damping=shared_damping,
        global_step_int=kfac_step_counter[0],
    )
    kfac_step_counter[0] += 1

    if reset_if_nan and jnp.any(jnp.isnan(stats['loss'])):
      new_params = old_params
      new_state = old_state
    
    optimizer_stats = OptimizerStats(
        param_norm=stats.get('param_norm'),
        grad_norm=stats.get('grad_norm'),
        precon_grad_norm=stats.get('precon_grad_norm'),
        update_norm=stats.get('update_norm'),
        precon_grad_tree=stats.get('precon_grad_tree'),
    )
    
    return data, new_params, new_state, stats['loss'], stats['aux'], pmove, optimizer_stats

  return step


def make_eval_step(
    mcmc_step,
    evaluate_loss: qmc_loss_functions.LossFn,
) -> Step:
  """Factory to create evaluation step (MCMC + loss eval, no updates).

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step.
    evaluate_loss: Loss function used for evaluation.

  Returns:
    step, a callable which performs a set of MCMC steps and then a loss
    evaluation without updating parameters or optimizer state.
  """

  optimizer_step = make_loss_step(evaluate_loss)

  @constants.pmap
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[OptimizerState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    _, _, loss, aux_data = optimizer_step(params, data, state, loss_key)

    return data, params, state, loss, aux_data, pmove, None

  return step
