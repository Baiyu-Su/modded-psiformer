"""Config D: Nesterov momentum, L2 raw-gradient clipping.

Molecule is specified via CLI overrides, e.g.:
  --config.system.atom=Li
"""

from ferminet import base_config
from ferminet.configs.atom import adjust_nuclear_charge
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  cfg = base_config.default()
  # KFAC settings: Nesterov momentum, L2 raw-gradient clipping
  cfg.optim.kfac.momentum = 0.9
  cfg.optim.kfac.nesterov = True
  cfg.optim.kfac.constrain_l2 = True
  cfg.optim.kfac.norm_constraint = 0.001
  cfg.optim.kfac.momentum_warmup_steps = 20000
  # Atom-style system setup
  cfg.system.atom = ''
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  with cfg.ignore_type():
    cfg.system.set_molecule = adjust_nuclear_charge
  return cfg
