"""Per-tensor logging utilities for optimizer and parameter statistics."""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import wandb


def flatten_param_tree_with_names(
    tree: Any, prefix: str = ''
) -> Dict[str, jnp.ndarray]:
  """Recursively flatten a nested param pytree into {path_string: leaf_array}.

  Handles dicts (keyed by str), lists/tuples (indexed by int).
  """
  flat = {}
  if isinstance(tree, dict):
    for key, subtree in tree.items():
      child_prefix = f'{prefix}.{key}' if prefix else str(key)
      flat.update(flatten_param_tree_with_names(subtree, child_prefix))
  elif isinstance(tree, (list, tuple)):
    for i, subtree in enumerate(tree):
      child_prefix = f'{prefix}.{i}' if prefix else str(i)
      flat.update(flatten_param_tree_with_names(subtree, child_prefix))
  elif isinstance(tree, jnp.ndarray) or isinstance(tree, np.ndarray):
    if prefix:
      flat[prefix] = tree
  return flat


def compute_per_tensor_rms(
    flat_dict: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
  """Compute RMS = sqrt(mean(x^2)) for each tensor in the flat dict."""
  rms_dict = {}
  for name, tensor in flat_dict.items():
    rms_dict[name] = float(jnp.sqrt(jnp.mean(tensor ** 2)))
  return rms_dict


def compute_per_tensor_param_stats(
    flat_dict: Dict[str, jnp.ndarray],
) -> Dict[str, Dict[str, float]]:
  """Compute RMS, abs min and abs max for each tensor in the flat dict."""
  stats = {}
  for name, tensor in flat_dict.items():
    abs_tensor = jnp.abs(tensor)
    stats[name] = {
        'rms': float(jnp.sqrt(jnp.mean(tensor ** 2))),
        'abs_min': float(jnp.min(abs_tensor)),
        'abs_max': float(jnp.max(abs_tensor)),
    }
  return stats


def log_optim_rms_to_wandb(
    precon_grad_tree: Any,
    step: int,
) -> None:
  """Log per-tensor RMS of preconditioned gradient to wandb under optim/.

  The precon_grad_tree comes from pmap and has a leading device axis on each
  leaf. We take [0] per leaf since values are identical across devices (pmean).
  """
  first_device_tree = _take_first_device(precon_grad_tree)
  flat = flatten_param_tree_with_names(first_device_tree)
  rms = compute_per_tensor_rms(flat)
  wandb_log = {f'optim/{name}': val for name, val in rms.items()}
  wandb_log['step'] = step
  wandb.log(wandb_log)


def log_param_stats_to_wandb(
    params_tree: Any,
    step: int,
) -> None:
  """Log per-tensor parameter RMS, abs min and abs max to wandb under param/."""
  first_device_tree = _take_first_device(params_tree)
  flat = flatten_param_tree_with_names(first_device_tree)
  stats = compute_per_tensor_param_stats(flat)
  wandb_log = {'step': step}
  for name, tensor_stats in stats.items():
    wandb_log[f'param/{name}/rms'] = tensor_stats['rms']
    wandb_log[f'param/{name}/abs_min'] = tensor_stats['abs_min']
    wandb_log[f'param/{name}/abs_max'] = tensor_stats['abs_max']
  wandb.log(wandb_log)


def _take_first_device(tree: Any) -> Any:
  """Strip the leading pmap device axis by taking index 0 on each leaf."""
  if isinstance(tree, dict):
    return {k: _take_first_device(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_take_first_device(v) for v in tree)
  elif isinstance(tree, (jnp.ndarray, np.ndarray)):
    return tree[0]
  return tree
