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

"""Neural network definitions for Psiformer in JAX."""
import enum
import functools
import itertools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
import jax
import jax.numpy as jnp
import numpy as np
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


class BackboneFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward evaluation up to the final backbone tensor H_theta(x)."""


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
  backbone: BackboneFnLike


## Network layers: building blocks ##


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions."""
  return list(itertools.accumulate(sizes))[:-1]


def init_linear_layer(
    key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True
) -> MutableMapping[str, jnp.ndarray]:
  """Initialises parameters for a linear layer, x w + b."""
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim))
  )
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}


def linear_layer(
    x: jnp.ndarray, w: jnp.ndarray, b: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b."""
  y = jnp.dot(x, w)
  return y + b if b is not None else y


def slogdet(x):
  """Computes sign and log of determinants of matrices."""
  if x.shape[-1] == 1:
    sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(
    xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Combines determinants and takes dot product with weights in log-domain."""
  det1d = functools.reduce(
      lambda a, b: a * b, [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1
  )
  phase_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1],
      (1, 0),
  )

  maxlogdet = jnp.max(logdet)
  det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.matmul(det, w)[0]
  phase_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return phase_out, log_out


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
          init_linear_layer(
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
        h_to_orbitals, array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = array_partitions(
        active_spin_channels
    )
    # Create orbitals.
    orbitals = [
        linear_layer(h, **p)
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


## Psiformer ##


@attr.s(auto_attribs=True, kw_only=True)
class PsiformerOptions(BaseNetworkOptions):
  """Options controlling the Psiformer part of the network architecture.

  Attributes:
    num_layers: Number of self-attention layers.
    num_heads: Number of multihead self-attention heads.
    heads_dim: Embedding dimension for each self-attention head.
    mlp_hidden_dims: Tuple of sizes of hidden dimension of the MLP. Note that
      this does not include the final projection to the embedding dimension.
    use_layer_norm: If true, include a layer norm on both attention and MLP.
  """

  num_layers: int = 2
  num_heads: int = 4
  heads_dim: int = 64
  mlp_hidden_dims: Tuple[int, ...] = (256,)
  use_layer_norm: bool = False
  activation: str = 'tanh'


def make_layer_norm() -> ...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(
      params: ParamTree, inputs: jnp.ndarray, axis: int = -1
  ) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply


def make_multi_head_attention(num_heads: int, heads_dim: int) -> ...:
  """FermiNet-style version of MultiHeadAttention."""

  def linear_projection(x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    y = jnp.dot(x, weights)
    return y.reshape(*x.shape[:-1], num_heads, heads_dim)

  def init(
      key: chex.PRNGKey,
      q_d: int,
      kv_d: int,
      output_channels: Optional[int] = None,
  ) -> Mapping[str, jnp.ndarray]:
    qkv_hiddens = num_heads * heads_dim
    if not output_channels:
      output_channels = qkv_hiddens

    key, *subkeys = jax.random.split(key, num=4)
    params = {}
    params['q_w'] = init_linear_layer(
        subkeys[0], in_dim=q_d, out_dim=qkv_hiddens, include_bias=False
    )['w']
    params['k_w'] = init_linear_layer(
        subkeys[1], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
    )['w']
    params['v_w'] = init_linear_layer(
        subkeys[2], in_dim=kv_d, out_dim=qkv_hiddens, include_bias=False
    )['w']

    key, subkey = jax.random.split(key)
    params['attn_output'] = init_linear_layer(
        subkey, in_dim=qkv_hiddens, out_dim=output_channels, include_bias=False
    )['w']

    return params

  def apply(
      params: ParamTree, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray
  ) -> jnp.ndarray:
    """Computes MultiHeadAttention with keys, queries and values."""
    q = linear_projection(query, params['q_w'])
    k = linear_projection(key, params['k_w'])
    v = linear_projection(value, params['v_w'])

    attn_logits = jnp.einsum('...thd,...Thd->...htT', q, k)
    scale = 1.0 / np.sqrt(heads_dim)
    attn_logits *= scale

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, v)
    attn = jnp.reshape(attn, (*query.shape[:-1], -1))
    return linear_layer(attn, params['attn_output'])

  return init, apply


def make_mlp(activation: str = 'tanh') -> ...:
  """Construct MLP, with final linear projection to embedding size."""

  if activation.lower() == 'tanh':
    act_fn = jnp.tanh
  elif activation.lower() == 'relu':
    act_fn = jax.nn.relu
  elif activation.lower() == 'silu':
    act_fn = jax.nn.silu
  else:
    raise ValueError(f'Unsupported activation: {activation}')

  def init(
      key: chex.PRNGKey, mlp_hidden_dims: Tuple[int, ...], embed_dim: int
  ) -> Sequence[Param]:
    params = []
    dims_one_in = [embed_dim, *mlp_hidden_dims]
    dims_one_out = [*mlp_hidden_dims, embed_dim]
    for i in range(len(dims_one_in)):
      key, subkey = jax.random.split(key)
      params.append(
          init_linear_layer(
              subkey,
              in_dim=dims_one_in[i],
              out_dim=dims_one_out[i],
              include_bias=True,
          )
      )
    return params

  def apply(params: Sequence[Param], inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)):
      x = act_fn(linear_layer(x, **params[i]))
    return x

  return init, apply


def make_self_attention_block(
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool = False,
    activation: str = 'tanh',
) -> ...:
  """Create a QKV self-attention block."""
  attention_init, attention_apply = make_multi_head_attention(num_heads, heads_dim)
  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()
  mlp_init, mlp_apply = make_mlp(activation)

  def init(key: chex.PRNGKey, qkv_d: int) -> ParamTree:
    attn_dim = qkv_d
    params = {}
    attn_params = []
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, attn_key, mlp_key = jax.random.split(key, 3)
      attn_params.append(
          attention_init(attn_key, q_d=qkv_d, kv_d=qkv_d, output_channels=attn_dim)
      )
      if use_layer_norm:
        ln_params.append([layer_norm_init(attn_dim), layer_norm_init(attn_dim)])
      mlp_params.append(mlp_init(mlp_key, mlp_hidden_dims, attn_dim))

    params['attention'] = attn_params
    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: ParamTree, qkv: jnp.ndarray) -> jnp.ndarray:
    x = qkv
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x, x, x)

      x = x + attn_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

      assert isinstance(params['mlp'][layer], (tuple, list))
      mlp_output = mlp_apply(params['mlp'][layer], x)

      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)

    return x

  return init, apply


def make_psiformer_layers(
    nspins: Tuple[int, ...], natoms: int, options: PsiformerOptions
) -> Tuple[InitLayersFn, ApplyLayersFn]:
  """Creates the permutation-equivariant layers for Psiformer."""
  del nspins, natoms  # Unused.

  attn_dim = options.num_heads * options.heads_dim
  self_attn_init, self_attn_apply = make_self_attention_block(
      num_layers=options.num_layers,
      num_heads=options.num_heads,
      heads_dim=options.heads_dim,
      mlp_hidden_dims=options.mlp_hidden_dims,
      use_layer_norm=options.use_layer_norm,
      activation=options.activation,
  )

  def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    feature_dim = one_electron_feature_dim + 1

    key, subkey = jax.random.split(key)
    params['embed'] = init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=attn_dim, include_bias=False
    )['w']

    key, subkey = jax.random.split(key)
    params.update(self_attn_init(key, attn_dim))

    return attn_dim, params

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the Psiformer interaction layers to a walker configuration."""
    del charges  # Unused.

    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)
    x = jnp.dot(ae_features, params['embed'])

    return self_attn_apply(params, x)

  return init, apply


def make_psiformer(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    bias_orbitals: bool = False,
    rescale_inputs: bool = False,
    num_layers: int,
    num_heads: int,
    heads_dim: int,
    mlp_hidden_dims: Tuple[int, ...],
    use_layer_norm: bool,
    activation: str,
) -> Network:
  """Psiformer with stacked self-attention layers."""
  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = make_ferminet_features(
        natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
    )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.SIMPLE_EE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = PsiformerOptions(
      ndim=ndim,
      determinants=determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      bias_orbitals=bias_orbitals,
      full_det=True,
      rescale_inputs=rescale_inputs,
      num_layers=num_layers,
      num_heads=num_heads,
      heads_dim=heads_dim,
      mlp_hidden_dims=mlp_hidden_dims,
      use_layer_norm=use_layer_norm,
      activation=activation,
  )  # pytype: disable=wrong-keyword-args

  psiformer_layers = make_psiformer_layers(nspins, charges.shape[0], options)
  _, psiformer_layers_apply = psiformer_layers
  orbitals_init, orbitals_apply = make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=psiformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Psiformer."""
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    return logdet_matmul(orbitals)

  def backbone_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns final Psiformer backbone features H_theta(x) with shape [L, d]."""
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    return psiformer_layers_apply(
        params['layers'],
        ae=ae,
        r_ae=r_ae,
        ee=ee,
        r_ee=r_ee,
        spins=spins,
        charges=charges,
    )

  return Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
      backbone=backbone_apply,
  )
