"""Memory-bounded chunking for multiplier-bootstrap weight matrices.

The multiplier bootstrap perturbs cached influence functions with a dense
``(n_bootstrap, n_units)`` weight matrix. At large ``n_units`` that matrix
dominates peak memory (e.g. ``999 x 5_000_000 x 8`` bytes is ~40 GB). Every
consumer is a left-multiply ``weights @ influence_vector`` whose result is small
(``(n_bootstrap,)`` or ``(n_bootstrap, n_gt)``), so the bootstrap can be tiled
over the *draw* dimension: generate and consume the weights in row-blocks of
``B``, capping the live intermediate at ``(B, n_units)``. FLOPs are identical to
the un-chunked path -- only the draw axis is tiled. The generated weight stream
is *bit-identical* to the un-chunked matrix (see below); the downstream
``weights @ influence`` matmuls go through BLAS, whose reduction order depends on
the operand row-count, so the resulting statistics match the un-chunked path to
within floating-point reassociation (typically <~1 ULP), far below bootstrap
Monte-Carlo error -- not bit-for-bit.

Bit-identity of the weight *generation* is preserved on **both** backends:

- **Rust** seeds each row absolutely as ``base_seed + row_index``
  (``rust/src/bootstrap.rs``), so calling the generator per block with base seed
  ``base_seed + chunk_start`` reproduces the exact un-chunked rows. Exactly one
  ``rng.integers`` draw is consumed, matching the un-chunked wrapper.
- The **NumPy** fallback draws the matrix row-major from the ``Generator``
  stream, so consuming it in contiguous, in-order blocks from the same generator
  reproduces the identical sequence.

The *influence* side can be tiled too. When a fit needs many perturbation
columns (per-cell IFs, the overall combined IF, per-event-time combined IFs),
computing them one column at a time degenerates into per-column slicing/GEMV
over the weight blocks -- memory-bandwidth-bound, not FLOP-bound. Instead,
:func:`tiled_if_matmul` scatters the columns into an ``(n_units, tile_cols)``
buffer and runs one BLAS GEMM per (weight block, column tile). The column axis
is tiled under a byte cap, and each column tile makes its own full pass over
the weight stream via :class:`ReplayableWeightStream`, which snapshots the
generator state at construction and restores it per pass -- so every pass sees
the bit-identical weight stream on both backends (Rust re-draws the same
``base_seed`` from the restored state; the NumPy fallback replays the stream
itself; the survey stratified branch regenerates its full PSU matrix from the
restored state). GEMM results match the per-column GEMV path to within BLAS
reassociation, same as the draw-axis chunking above.
"""

from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from diff_diff._backend import HAS_RUST_BACKEND, _rust_bootstrap_weights
from diff_diff.bootstrap_utils import generate_bootstrap_weights_batch_numpy

if TYPE_CHECKING:
    from diff_diff.survey import ResolvedSurveyDesign

# Byte ceiling for a single ``(B, n_units)`` float64 weight block. 256 MB keeps
# the live intermediate small at millions of units while staying large enough
# that the per-block matmuls remain BLAS-efficient and chunk overhead (a handful
# of extra Python iterations / FFI calls) is negligible.
_TARGET_BLOCK_BYTES = 256 * 1024 * 1024


def effective_weight_backend() -> str:
    """The weight-generation backend :func:`iter_weight_blocks` would use NOW.

    Returns ``"rust"`` exactly when the generator branch below does — the
    predicate must stay identical to :func:`iter_weight_blocks`'s own
    ``rust_gen`` resolution. The two backends produce DIFFERENT draws from
    the same bit-generator state (Rust draws one base seed and row-seeds
    Xoshiro absolutely; the NumPy fallback consumes the PCG64 stream
    directly), so a captured RNG state replays bit-identically only within
    one backend. Post-fit bootstrap replay (CallawaySantAnna's
    ``BootstrapReplaySpec``) stamps this value at fit and fails closed on a
    mismatch rather than silently regenerating a different realization.
    """
    return "rust" if (HAS_RUST_BACKEND and _rust_bootstrap_weights is not None) else "numpy"


def compute_block_size(
    n_units: int, n_bootstrap: int, target_bytes: int = _TARGET_BLOCK_BYTES
) -> int:
    """Number of bootstrap rows per block so a ``(B, n_units)`` float64 block
    stays under ``target_bytes``. Always in ``[1, n_bootstrap]``."""
    if n_units <= 0:
        return max(1, n_bootstrap)
    b = target_bytes // (n_units * 8)
    return int(max(1, min(max(1, n_bootstrap), b)))


def iter_weight_blocks(
    n_bootstrap: int,
    n_gen: int,
    weight_type: str,
    rng: np.random.Generator,
    *,
    expand_index: Optional[np.ndarray] = None,
    block_size: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(chunk_start, block)`` pairs covering all ``n_bootstrap`` draws.

    ``block`` has shape ``(B, width)`` where ``width = len(expand_index)`` when
    ``expand_index`` is given, else ``n_gen``. Weights are generated at width
    ``n_gen`` (unit / cluster / PSU level) and, when ``expand_index`` is given,
    expanded to unit level via ``block[:, expand_index]`` (cluster->unit or
    PSU->unit fan-out). The concatenation of all yielded blocks is bit-identical
    to a single ``generate_bootstrap_weights_batch(n_bootstrap, n_gen, ...)``
    followed by the same expansion.

    Generation is in-order and stateful on ``rng`` (NumPy fallback) -- the caller
    must consume the iterator sequentially, which the chunk loop does.
    """
    width = n_gen if expand_index is None else int(len(expand_index))
    if block_size is None:
        block_size = compute_block_size(width, n_bootstrap)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    rust_gen = (
        _rust_bootstrap_weights
        if (HAS_RUST_BACKEND and _rust_bootstrap_weights is not None)
        else None
    )
    # Draw exactly one base seed (matching the un-chunked Rust wrapper); the
    # NumPy fallback consumes the rng stream directly per block instead.
    base_seed = int(rng.integers(0, 2**63 - 1)) if rust_gen is not None else 0

    for chunk_start in range(0, n_bootstrap, block_size):
        rows = min(block_size, n_bootstrap - chunk_start)
        if rust_gen is not None:
            block = rust_gen(rows, n_gen, weight_type, base_seed + chunk_start)
        else:
            block = generate_bootstrap_weights_batch_numpy(rows, n_gen, weight_type, rng)
        if expand_index is not None:
            block = block[:, expand_index]
        yield chunk_start, block


def iter_survey_multiplier_weight_blocks(
    n_bootstrap: int,
    resolved_survey: Optional["ResolvedSurveyDesign"],
    weight_type: str,
    rng: np.random.Generator,
    *,
    block_size: int,
) -> Tuple[np.ndarray, Iterator[Tuple[int, np.ndarray]]]:
    """Chunked PSU-level multiplier weights for the survey-aware bootstrap.

    Returns ``(psu_ids, blocks)`` where ``blocks`` yields
    ``(chunk_start, (B, n_psu))`` PSU-weight blocks covering all draws.

    For UNSTRATIFIED designs (``strata is None``, ``n_psu >= 2``) the
    ``(n_bootstrap, n_psu)`` matrix is generated one draw-block at a time via
    :func:`iter_weight_blocks` plus the unstratified FPC scalar -- bit-identical
    to the unstratified branch of
    :func:`diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch`,
    but the full matrix is never materialized. This is the path taken by
    ``cluster="unit"`` (each unit its own PSU, ``n_psu == n_units``), the case
    that otherwise dominates bootstrap memory at large n_units.

    Stratified designs (and the ``n_psu < 2`` degenerate case) fall back to full
    generation + sliced blocks: per-stratum / lonely-PSU generation is not tiled
    here, but stratified designs have few PSUs so the full matrix is small.
    """
    from diff_diff.bootstrap_utils import generate_survey_multiplier_weights_batch

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    # Callers only reach the survey-multiplier path with a resolved design.
    assert resolved_survey is not None
    psu = getattr(resolved_survey, "psu", None)
    strata = getattr(resolved_survey, "strata", None)
    if psu is None:
        n_psu = len(resolved_survey.weights)
        psu_ids = np.arange(n_psu)
    else:
        psu_ids = np.unique(psu)
        n_psu = len(psu_ids)

    if strata is not None or n_psu < 2:
        # Stratified or degenerate single-PSU: full generation (small here).
        weights, psu_ids = generate_survey_multiplier_weights_batch(
            n_bootstrap, resolved_survey, weight_type, rng
        )

        def _sliced() -> Iterator[Tuple[int, np.ndarray]]:
            for chunk_start in range(0, n_bootstrap, block_size):
                yield chunk_start, weights[chunk_start : chunk_start + block_size]

        return psu_ids, _sliced()

    # Unstratified, n_psu >= 2: tile the generation over draws. Mirror the
    # unstratified FPC scaling from generate_survey_multiplier_weights_batch.
    fpc = getattr(resolved_survey, "fpc", None)
    fpc_scale = 1.0
    fpc_zero = False
    if fpc is not None:
        # psu=None already sets n_psu = len(weights), so n_units_for_fpc == n_psu
        # on both branches of the original generator.
        n_units_for_fpc = n_psu
        if fpc[0] < n_units_for_fpc:
            raise ValueError(
                f"FPC ({fpc[0]}) is less than the number of PSUs "
                f"({n_units_for_fpc}). FPC must be >= number of PSUs."
            )
        f = n_units_for_fpc / fpc[0]
        if f < 1.0:
            fpc_scale = float(np.sqrt(1.0 - f))
        else:
            fpc_zero = True

    def _generated() -> Iterator[Tuple[int, np.ndarray]]:
        for chunk_start, block in iter_weight_blocks(
            n_bootstrap, n_psu, weight_type, rng, block_size=block_size
        ):
            if fpc_zero:
                block = np.zeros_like(block)
            elif fpc_scale != 1.0:
                block = block * fpc_scale
            yield chunk_start, block

    return psu_ids, _generated()


# Byte ceiling for the ``(n_units, tile_cols)`` float64 scatter buffer used by
# ``tiled_if_matmul``. Sibling of ``_TARGET_BLOCK_BYTES``; together they bound
# the kernel's live intermediates at ~384 MB regardless of n_units or the
# number of perturbation columns. Read at CALL time inside ``tiled_if_matmul``
# (never bound as a def-time default) so tests can monkeypatch it.
_TARGET_TILE_BYTES = 128 * 1024 * 1024

# One perturbation column of ``tiled_if_matmul``'s influence matrix: a list of
# (index, values) contributions. ``index=None`` assigns the full-width dense
# ``values``; otherwise ``values`` is scattered at ``index``. A column may also
# be a zero-argument callable returning such a list, materialized only when its
# tile is filled (keeps peak memory at one O(n_units) temporary for callers
# whose columns are computed on the fly).
IFColumn = List[Tuple[Optional[np.ndarray], np.ndarray]]
IFColumnSpec = Union[IFColumn, Callable[[], IFColumn]]


class ReplayableWeightStream:
    """Multiplier-weight block stream that can be re-iterated bit-identically.

    Wraps a weight-block iterator factory together with the ``Generator`` that
    feeds it, snapshotting the generator state at construction. Every
    ``__iter__`` restores the snapshot before invoking the factory, so each
    pass yields the exact same ``(chunk_start, block)`` sequence:

    - **Rust backend**: ``iter_weight_blocks`` draws its per-run ``base_seed``
      from the rng lazily at the first ``next()``; the restored state
      reproduces the same base seed, and rows are seeded absolutely from it.
    - **NumPy fallback**: generation consumes the rng stream directly; the
      restored state replays the identical stream.
    - **Survey stratified branch**: ``iter_survey_multiplier_weight_blocks``
      generates its full PSU matrix eagerly at factory-call time -- from the
      restored state, so per-pass recreation is also bit-identical.

    The snapshot MUST precede any rng-consuming call: construct this object
    before probing generators that draw at call time (the survey stratified
    branch draws eagerly; resolve ``psu_ids`` from the design's ``psu`` array
    directly instead of via a probe call). After the final pass the rng ends
    exactly where a single pass would leave it, so downstream consumers of the
    same generator are unaffected by the number of passes.
    """

    def __init__(
        self,
        make_iter: Callable[[np.random.Generator], Iterator[Tuple[int, np.ndarray]]],
        rng: np.random.Generator,
    ) -> None:
        self._make_iter = make_iter
        self._rng = rng
        self._state = rng.bit_generator.state

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        self._rng.bit_generator.state = self._state
        return self._make_iter(self._rng)


def tiled_if_matmul(
    weight_stream: Iterable[Tuple[int, np.ndarray]],
    n_bootstrap: int,
    n_units: int,
    columns: Sequence[IFColumnSpec],
    *,
    tile_bytes: Optional[int] = None,
) -> np.ndarray:
    """Compute ``W @ M`` for the multiplier bootstrap without materializing
    ``W`` (streamed in row blocks) or ``M`` (scattered in column tiles).

    ``W`` is the ``(n_bootstrap, n_units)`` weight matrix yielded by
    ``weight_stream`` as ``(chunk_start, block)`` row blocks; ``M`` is the
    ``(n_units, len(columns))`` influence matrix described by ``columns``
    (see :data:`IFColumnSpec`). Returns the dense ``(n_bootstrap, n_cols)``
    product. Columns are tiled under ``tile_bytes`` (default: module attribute
    ``_TARGET_TILE_BYTES``, resolved at call time); each tile is scattered
    once into an F-order buffer and then consumed by one BLAS GEMM per weight
    block. With more than one tile, ``weight_stream`` is iterated once per
    tile and therefore must be re-iterable with an identical stream --
    :class:`ReplayableWeightStream` provides exactly that.

    Scatter contract: contributions are written by plain assignment, so index
    arrays must be unique within each contribution and the contributions of a
    single column must be mutually disjoint (duplicate or overlapping indices
    keep the last write instead of summing like ``W[:, idx] @ values`` would).
    All in-repo callers satisfy this by construction (treated/control unit
    sets are disjoint; dense columns are single-contribution).

    A non-finite influence value poisons only its own output column (every
    output element is an independent dot product), matching the per-column
    GEMV behavior this kernel replaces. Results match the per-column GEMV
    path to within BLAS reassociation (~1 ULP), not bit-for-bit.
    """
    n_cols = len(columns)
    out = np.empty((n_bootstrap, n_cols))
    if n_cols == 0 or n_bootstrap == 0:
        return out
    if tile_bytes is None:
        tile_bytes = _TARGET_TILE_BYTES
    tile_cols = int(max(1, min(n_cols, tile_bytes // (max(1, n_units) * 8))))

    # A single-pass iterator silently exhausts after the first tile, leaving
    # uninitialized rows in `out` -- fail loudly instead. isinstance (not an
    # iter() probe) so re-iterable streams see no extra pass.
    if tile_cols < n_cols and isinstance(weight_stream, abc.Iterator):
        raise ValueError(
            "tiled_if_matmul needs more than one column tile but weight_stream "
            "is a single-pass iterator; pass a re-iterable stream "
            "(ReplayableWeightStream) so each tile can replay the weights."
        )

    buf = np.zeros((n_units, tile_cols), order="F")
    for lo in range(0, n_cols, tile_cols):
        hi = min(lo + tile_cols, n_cols)
        width = hi - lo
        if lo > 0:
            buf[:, :width] = 0.0
        for c in range(lo, hi):
            spec = columns[c]
            if callable(spec):
                spec = spec()
            col = buf[:, c - lo]
            for idx, values in spec:
                if idx is None:
                    np.copyto(col, values)
                else:
                    col[idx] = values
        tile = buf[:, :width]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for chunk_start, w_block in weight_stream:
                out[chunk_start : chunk_start + w_block.shape[0], lo:hi] = w_block @ tile
    return out
