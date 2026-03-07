# GUIDE.md -- Agent Onboarding

This document is for AI agents working on this codebase. It covers what this repo is, how to set it up, and how the code is organized.

## What This Is

A modified fork of **FermiNet/PsiFormer** -- JAX-based neural network wavefunctions for solving the many-electron Schrodinger equation via variational Monte Carlo (VMC). The PsiFormer variant uses a transformer (self-attention) ansatz.

Key dependencies: JAX (with GPU/CUDA), kfac_jax (patched), ml_collections, PySCF, optax, wandb.

## Environment Setup

Use conda (not virtualenv). The environment should already exist on the cluster, but if you need to recreate it:

```bash
# 1. Create and activate a conda environment
conda create -n psiformer python=3.11 -y
conda activate psiformer

# 2. Install JAX with GPU support (adjust CUDA version as needed)
pip install jax[cuda12]

# 3. Install the package in editable mode (pulls most dependencies)
pip install -e .

# 4. Install kfac_jax from source and apply our patch
#    The pip install above installs kfac_jax, but we need to overwrite
#    its internal optimizer.py with our patched version.
#    Find where kfac_jax is installed, then copy our patch over it:
KFAC_DIR=$(python -c "import kfac_jax; import os; print(os.path.dirname(kfac_jax.__file__))")
cp patches/kfac_jax/_src/optimizer.py "${KFAC_DIR}/_src/optimizer.py"
```

The patch in `patches/kfac_jax/_src/optimizer.py` fixes multi-host pmap issues in kfac_jax (see the Memory section below for details). This step is **mandatory** -- without it, multi-node training will crash.

To verify the patch is applied:
```bash
diff patches/kfac_jax/_src/optimizer.py "$(python -c 'import kfac_jax, os; print(os.path.dirname(kfac_jax.__file__))')/_src/optimizer.py"
# Should produce no output (files are identical)
```

## Running

### Single command
```bash
conda activate psiformer
python3 ferminet/main.py \
    --config ferminet/configs/bicbut.py \
    --config.batch_size 256 \
    --config.pretrain.iterations 100
```

### SLURM (multi-GPU cluster)
```bash
sbatch train_example.slurm
```
The default SLURM script runs bicyclobutane (`ferminet/configs/bicbut.py`). Edit the script to change the config or hyperparameters.

### Inference (fixed parameters, accumulate statistics)
```bash
python3 ferminet/main.py \
    --config ferminet/configs/bicbut.py \
    --config.optim.optimizer none \
    --config.log.restore_path /path/to/checkpoint
```

## Repository Structure

```
ferminet/
  main.py              # CLI entry point, parses config, calls train.train(cfg)
  train.py             # Core training loop: init, pretrain, MCMC, VMC, checkpointing
  base_config.py       # Default ConfigDict with all hyperparameters
  networks.py          # FermiNet architecture (feature layers, orbitals, Slater dets)
  psiformer.py         # PsiFormer (attention) architecture, same init/apply interface
  network_blocks.py    # Shared building blocks (linear, residual, log-sum-det)
  envelopes.py         # Multiplicative envelope functions
  jastrows.py          # Jastrow correlation factors
  hamiltonian.py       # Local energy (kinetic + potential)
  loss.py              # VMC loss, custom gradients, folx forward-mode Laplacian
  mcmc.py              # MCMC sampling (Metropolis-Hastings, MALA)
  pretrain.py          # Pretraining against Hartree-Fock orbitals
  pseudopotential.py   # Pseudopotential support
  checkpoint.py        # Save/restore model parameters
  steps.py             # Optimizer step logic (KFAC, Adam wrappers)
  configs/             # System configs (atom.py, bicbut.py, benzene.py, etc.)
    excited/           # NES-VMC and ensemble penalty excited-state configs
  utils/
    system.py          # Atom namedtuple, molecular system setup
    scf.py             # PySCF interface for SCF calculations
    writers.py         # CSV/file output writers
    statistics.py      # Statistical analysis of MCMC data
    precision.py       # Precision control utilities
patches/
  kfac_jax/_src/optimizer.py   # Patched kfac_jax optimizer for multi-host support
train_example.slurm    # Example SLURM submission script
```

## Key Conventions

- **Config system**: `ml_collections.ConfigDict`. Override any param on CLI with `--config.path.to.param=value`.
- **Batchless networks**: Network functions are written without batch dimensions. Batching is handled by `pmap` (across GPUs) + `vmap` (within GPU).
- **Multi-GPU**: Automatic via JAX `pmap`. Multi-node requires `jax.distributed.initialize()` (already in main.py).
- **Optimizers**: `kfac` (default, via patched kfac_jax), `adam` (via optax), `none` (inference only).
- **Style**: Google Python style -- 2-space indent, max 80 char lines.
- **Logging**: wandb on by default (`cfg.log.use_wandb`). Output goes to `cfg.log.save_path`.
- **Output files**: `train_stats.csv` (energy, acceptance prob per iteration), `checkpoints/` directory.

## Multi-Host Training Notes

This fork has modifications for multi-node (multi-host) JAX training. Key rules:

- Use `broadcast_all_local_devices(x[None])` instead of `replicate_all_local_devices(x)` to ensure PmapSharding consistency across hosts.
- Never index PmapSharding arrays with `array[0]` on non-primary hosts -- use `array[local_pmap_idx]` where `local_pmap_idx = process_index * num_devices`.
- Only host 0 writes checkpoints, CSV logs, wandb data.
- The kfac_jax patch fixes `get_first(step_counter)` which would crash non-primary hosts.

Files modified for multi-host support: `train.py`, `steps.py`, `pretrain.py`, `checkpoint.py`, `utils/writers.py`, `main.py`, `utils/precision.py`.

## Available Configs

| Config | System | Electrons |
|--------|--------|-----------|
| `atom.py` | Single atoms (pass `--config.system.atom SYMBOL`) | varies |
| `bicbut.py` | Bicyclobutane (C4H6) | 30 (15, 15) |
| `benzene.py` | Benzene | varies |
| `c2h4.py` | Ethylene | varies |
| `ch4.py` | Methane | varies |
| `nh3.py` | Ammonia | varies |
| `o3.py` | Ozone | varies |
| `diatomic.py` | Generic diatomic | varies |
