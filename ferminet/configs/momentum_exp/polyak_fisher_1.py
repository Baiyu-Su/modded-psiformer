"""C2H4 Polyak config: momentum, Fisher-norm clipping, EMA variant."""

import os

from ferminet.configs import c2h4
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  cfg = c2h4.get_config()
  run_name = os.environ.get('FERMINET_RUN_NAME', 'c2h4_B_polyak_fisher_1')

  cfg.log.save_path = f'./runs/{run_name}'
  cfg.log.use_wandb = True
  cfg.log.wandb.name = run_name
  cfg.log.wandb.project = 'psiformer'
  cfg.log.wandb.notes = 'B_polyak_fisher_1 C2H4 momentum ablation'

  cfg.mcmc.algorithm = 'mh'
  cfg.debug.check_nan = False
  cfg.optim.reset_if_nan = False
  cfg.optim.kfac.momentum = 0.9
  cfg.optim.kfac.momentum_warmup_steps = 10000
  cfg.optim.kfac.nesterov = False
  cfg.optim.kfac.constrain_l2 = False
  cfg.optim.kfac.norm_constraint = 0.001
  cfg.optim.kfac.cov_ema_decay = 0.998
  cfg.optim.lr.rate = 5e-2
  return cfg
