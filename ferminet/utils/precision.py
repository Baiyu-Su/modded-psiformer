"""Precision verification utilities."""

import os
from absl import logging
import jax
import jax.numpy as jnp


def verify_precision_config():
  """Log and verify that the runtime precision config is as expected.

  Checks:
    - JAX X64 mode (on or off)
    - JAX default matmul precision
    - NVIDIA_TF32_OVERRIDE env var
  """
  x64 = jax.config.jax_enable_x64
  matmul_prec = str(jax.config.jax_default_matmul_precision)
  tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE", "<unset>")

  logging.info(
      "Precision config: x64=%s, matmul_precision=%s, NVIDIA_TF32_OVERRIDE=%s",
      x64, matmul_prec, tf32_override,
  )

  errors = []
  if x64:
    errors.append("jax_enable_x64=True (expected False for FP32 runs)")
  if matmul_prec != "highest":
    errors.append(
        f"jax_default_matmul_precision={matmul_prec!r} (expected 'highest')"
    )
  if tf32_override != "0":
    errors.append(
        f"NVIDIA_TF32_OVERRIDE={tf32_override!r} (expected '0')"
    )

  if errors:
    raise ValueError(
        "Precision guard failed: " + "; ".join(errors)
    )


def log_tree_dtypes(tree, name: str = "tree"):
  """Log unique dtypes found in a pytree."""
  leaves = jax.tree_util.tree_leaves(tree)
  dtypes = {str(l.dtype) for l in leaves if hasattr(l, "dtype")}
  logging.info("%s dtypes: %s", name, dtypes)
