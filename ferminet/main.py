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

"""Main wrapper for FermiNet in JAX."""

import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

from absl import app
from absl import flags
from absl import logging
from ferminet import base_config
from ferminet import train
from ml_collections.config_flags import config_flags

# internal imports

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def _infer_local_device_ids():
  """Infers process-local device ids from CUDA_VISIBLE_DEVICES when set."""
  visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
  if not visible_devices or visible_devices == 'NoDevFiles':
    return None
  device_tokens = [token.strip() for token in visible_devices.split(',')
                   if token.strip()]
  if not device_tokens:
    return None
  # CUDA_VISIBLE_DEVICES remaps ordinals inside the process to [0, n).
  return list(range(len(device_tokens)))


def main(_):
  cfg = FLAGS.config
  cfg = base_config.resolve(cfg)
  import jax
  jax.distributed.initialize(local_device_ids=_infer_local_device_ids())
  if jax.process_index() == 0:
    logging.info('System config:\n\n%s', cfg)
  train.train(cfg)


if __name__ == '__main__':
  app.run(main)
