# Copyright 2026 Simple PSI Authors.
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

"""Bicyclobutane (bicbut) example config."""

from ferminet import base_config
from ferminet.utils.system import Atom


def get_config():
  """Returns config for running bicyclobutane in bohr units."""
  cfg = base_config.default()
  cfg.system.electrons = (15, 15)
  cfg.system.molecule = [
      Atom(symbol='C', coords=(0.0, 2.13792, 0.58661), units='bohr'),
      Atom(symbol='C', coords=(0.0, -2.13792, 0.58661), units='bohr'),
      Atom(symbol='C', coords=(1.41342, 0.0, -0.58924), units='bohr'),
      Atom(symbol='C', coords=(-1.41342, 0.0, -0.58924), units='bohr'),
      Atom(symbol='H', coords=(0.0, 2.33765, 2.64110), units='bohr'),
      Atom(symbol='H', coords=(0.0, 3.92566, -0.43023), units='bohr'),
      Atom(symbol='H', coords=(0.0, -2.33765, 2.64110), units='bohr'),
      Atom(symbol='H', coords=(0.0, -3.92566, -0.43023), units='bohr'),
      Atom(symbol='H', coords=(2.67285, 0.0, -2.19514), units='bohr'),
      Atom(symbol='H', coords=(-2.67285, 0.0, -2.19514), units='bohr'),
  ]
  with cfg.system.ignore_type():
    cfg.system.atom_spin_configs = (
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
  return cfg
