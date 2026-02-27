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

"""Implementation of Fermionic Neural Network in JAX."""
import enum
from typing import Any, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol


# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[
    jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']
]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class FermiNetData:
  """Data passed to network.

  Shapes given for an unbatched element (i.e. a single MCMC configuration).

  NOTE:
    the networks are written in batchless form. Typically one then maps
    (pmap+vmap) over every attribute of FermiNetData (nb this is required if
    using KFAC, as it assumes the FIM is estimated over a batch of data), but
    this is not strictly required. If some attributes are not mapped, then JAX
    simply broadcasts them to the mapped dimensions (i.e. those attributes are
    treated as identical for every MCMC configuration.

  Attributes:
    positions: walker positions, shape (nelectrons*ndim).
    spins: spins of each walker, shape (nelectrons).
    atoms: atomic positions, shape (natoms*ndim).
    charges: atomic charges, shape (natoms).
  """

  # We need to be able to construct instances of this with leaf nodes as jax
  # arrays (for the actual data) and as integers (to use with in_axes for
  # jax.vmap etc). We can type the struct to be either all arrays or all ints
  # using Generic, it just slightly complicates the type annotations in a few
  # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
  positions: Any
  spins: Any
  atoms: Any
  charges: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclei charges, shape: (natoms).
    """


class LogFermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclear charges, shape: (natoms).
    """


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      Sequence of orbitals.
    """


class InitLayersFn(Protocol):

  def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns output dim and initialized parameters for the interaction layers.

    Args:
      key: RNG state
    """


class ApplyLayersFn(Protocol):

  def __call__(
      self,
      params: ParamTree,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward evaluation of the equivariant interaction layers.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """


## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(
      self,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      **params: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(
      self,
      natoms: int,
      nspins: Sequence[int],
      ndim: int,
      **kwargs: Any,
  ) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      natoms: number of atoms.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
  """Options controlling the overall network architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    rescale_inputs: If true, rescale the inputs so they grow as log(|r|).
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
    jastrow: Type of Jastrow factor if used, or 'none' if no Jastrow factor.
  """

  ndim: int = 3
  determinants: int = 16
  full_det: bool = True
  rescale_inputs: bool = False
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  jastrow: jastrows.JastrowType = jastrows.JastrowType.NONE


# Network class.


@attr.s(auto_attribs=True)
class Network:
  options: BaseNetworkOptions
  init: InitFermiNet
  apply: FermiNetLike
  orbitals: OrbitalFnLike


## Network layers: features ##


def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
  return ae, ee, r_ae, r_ee[..., None]


def make_ferminet_features(
    natoms: int,
    nspins: Optional[Tuple[int, int]] = None,
    ndim: int = 3,
    rescale_inputs: bool = False,
) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (natoms * (ndim + 1), ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if rescale_inputs:
      log_r_ae = jnp.log(1 + r_ae)  # grows as log(r) rather than r
      ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=2)

      log_r_ee = jnp.log(1 + r_ee)
      ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)

    else:
      ae_features = jnp.concatenate((r_ae, ae), axis=2)
      ee_features = jnp.concatenate((r_ee, ee), axis=2)
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    return ae_features, ee_features

  return FeatureLayer(init=init, apply=apply)


## Network layers: orbitals ##


def make_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
    equivariant_layers: Tuple of init, apply functions for the equivariant
      interaction part of the network.
  """

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(options.jastrow)

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    # How many spin-orbitals do we need to create per spin channel?
    nspin_orbitals = []
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per
        # electron per determinant.
        norbitals = nspin * options.determinants
      nspin_orbitals.append(norbitals)

    # create envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params.
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # orbital shaping
    orbitals = []
    for nspin_orbital in nspin_orbitals:
      key, subkey = jax.random.split(key)
      orbitals.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_orbital_in,
              out_dim=nspin_orbital,
              include_bias=options.bias_orbitals,
          )
      )
    params['orbital'] = orbitals

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    h_to_orbitals = equivariant_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h_to_orbitals = envelope_factor * h_to_orbitals
    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    # Apply envelopes if required.
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
      r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
      r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
      for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
            ae=ae_channels[i],
            r_ae=r_ae_channels[i],
            r_ee=r_ee_channels[i],
            **params['envelope'][i],
        )

    # Reshape into matrices.
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]

    # Optionally apply Jastrow factor for electron cusp conditions.
    # Added pre-determinant for compatibility with pretraining.
    if jastrow_apply is not None:
      jastrow = jnp.exp(
          jastrow_apply(r_ee, params['jastrow'], nspins) / sum(nspins)
      )
      orbitals = [orbital * jastrow for orbital in orbitals]

    return orbitals

  return init, apply
