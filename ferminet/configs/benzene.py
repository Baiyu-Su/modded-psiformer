# Copyright 2023 Bytedance Ltd. and/or its affiliate
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

from ferminet import base_config
from ferminet.utils import system
from ferminet.utils.system import Atom

# Settings in a a config files are loaded by executing the the get_config
# function.

# Geometry of Benzene sigle molecule is from https://pubs.acs.org/doi/10.1021/acs.jpclett.0c02621,
# which is at the MP2/6-31G* level.

def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (21, 21)  # Spins
    cfg.system.molecule = [
        Atom(symbol='C', coords=(0.00000, 2.63664, 0.00000), units='bohr'),
        Atom(symbol='C', coords=(2.28339, 1.31832, 0.00000), units='bohr'),
        Atom(symbol='C', coords=(2.28339, -1.31832, 0.00000), units='bohr'),
        Atom(symbol='C', coords=(0.00000, -2.63664, 0.00000), units='bohr'),
        Atom(symbol='C', coords=(-2.28339, -1.31832, 0.00000), units='bohr'),
        Atom(symbol='C', coords=(-2.28339, 1.31832, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(0.00000, 4.69096, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(4.06250, 2.34549, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(4.06250, -2.34549, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(0.00000, -4.69096, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(-4.06250, -2.34549, 0.00000), units='bohr'),
        Atom(symbol='H', coords=(-4.06250, 2.34549, 0.00000), units='bohr'),
    ]
    cfg.system.atom_spin_configs = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0),]
    return cfg