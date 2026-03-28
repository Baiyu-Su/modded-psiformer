# KFAC Gradient Norm Pipeline

## Fisher-norm mode (`constrain_l2=False`)

```
grads  ──→  raw-momentum EMA / Nesterov raw direction  ──→  F⁻¹·direction  ──→  [Fisher-norm clip]  ──→  delta = -lr·precon_dir
  │                                  │                            │                   │
grad_norm                        fisher_norm                 precon_grad_norm     update_norm
(raw, unclipped)                 (pre-clip)                 (pre-clip)           (post-clip)
```

- `fisher_norm` = `sqrt(gᵀ F⁻¹ g)` where `g` is the raw momentum direction
- Fisher clipping acts on the preconditioned momentum direction only

## L2-norm mode (`constrain_l2=True`)

```
grads  ──→  [raw L2 clip]  ──→  raw-momentum EMA / Nesterov raw direction  ──→  F⁻¹·direction  ──→  delta = -lr·precon_dir
  │                │                               │                            │
grad_norm     (fed into momentum)             fisher_norm                 precon_grad_norm
(raw)                                        (pre-clip)                  (pre-clip)
```

- Raw L2 clipping acts on the raw gradient before momentum
- Fisher clipping is skipped in this mode

## Logged quantities

| wandb key | What it is | Source |
|-----------|-----------|--------|
| `train/grad_norm` | `‖g_raw‖` before any raw L2 clip | `utils.norm(grads)` |
| `train/precon_grad_norm` | `‖F⁻¹·direction‖` before any Fisher clip | `utils.norm(preconditioned_direction_pre_clip)` |
| `train/fisher_norm` | `sqrt(directionᵀ F⁻¹ direction)` before any Fisher clip | `jnp.sqrt(utils.inner_product(preconditioned_direction_pre_clip, direction))` |
| `train/update_norm` | `‖delta‖` full param update incl. momentum | `utils.norm(delta)` |
| `train/param_norm` | `‖params‖` | `utils.norm(params)` |
