# Task 10 Report: SE(3)-Equivariant Pocket Encoder and Joint Ligand Backbone

## Outcome

Implemented the first public joint pocket-conditioned ligand model used by
Tasks 11--13. `ECloudFlowModel` is a device-agnostic `torch.nn.Module` with no
implicit transfers or rank-local global state. It consumes the canonical
flattened `MolecularState`/`GenerationCondition` contracts and returns typed
`ModelPrediction` tensors for position and packed-electron flow/score,
categorical atom/charge/bond endpoints, atom count, affinity, and interaction.

## Architecture and contract decisions

- `ModelConfig.vector_dim` remains the pocket/backbone vector multiplicity.
  The packed Task 8 electron multiplicity is a separate, validated and
  overrideable `electron_vector_dim=8` constructor/from-config default.
  Together with `electron_latent_dim=48,lmax=2`, the model validates and uses
  the exact `19x0e + 8x1o + 1x2e` layout.
- Packed electron blocks are never flattened into invariant channels.
  Invariant conditioning uses scalar blocks and irrep norms. Output heads apply
  shared scalar gates across every component of an irrep; only the `1o` block
  receives Cartesian relative-vector injection.
- Pocket, ligand, and pocket-to-ligand neighborhoods use PyG radius operations
  when the optional compiled `torch-cluster` extra is installed. A deterministic
  sparse native fallback is used when PyG raises its documented missing-extra
  `ImportError`. Neither path allocates dense `[N,N,C]` features.
- Coordinate outputs are learned scalar weights multiplying ligand/pocket
  relative unit vectors, encoded pocket vectors, and packed `1o` contractions.
  Proper rotations are equivariant and translations cancel. Reflections are
  intentionally outside the contract.
- Pocket input features are reduced through channel-independent invariant
  summaries before learned projection. The actual positive width is recorded
  and cache-validated, so Task 6 width 50 works without becoming a hidden
  fixture assumption; width 7 is independently tested.
- Pocket cache keys cover pocket position/feature/batch bytes, dtype, device,
  actual feature layout, null mode, encoder architecture, process-local encoder
  ownership, and parameter version counters. A changed pocket/frame/device,
  changed encoder parameters, or another model instance is rejected before
  stale conditioning is used. The caller explicitly owns the cache object.
- Classifier-free null conditioning uses a trainable null scalar embedding,
  zero pocket vectors, retained geometry/batch metadata, and removal of task
  and property signals. Stable SHA-256-derived task features make arbitrary
  non-empty fragment task IDs real conditioning without a process-global
  vocabulary.
- Canonical halfedges remain one unordered `src < dst` row per pair. The bond
  head uses commutative `source+target` and `abs(source-target)` features and
  produces `O(E)` logits.
- Fragment coordinate velocities/scores are zeroed before output. Fixed
  atom/charge/bond categorical fields are functionally restored from the input
  state before output, remaining compatible with downstream exact clamping.
  Count logits are masked before normalization, giving exact zero probability
  below each complex's own fixed-atom count.
- Empty ligand states derive batch size from pocket/time and still return count
  and auxiliary values. Zero-valued RMS contractions clamp the mean-square
  before square root so the backward derivative stays finite.
- `PocketEncoding`, `PocketEncoder.encode`, `AtomCountPredictor.forward`,
  `ECloudFlowModel.encode_pocket`, `ECloudFlowModel.forward`, and
  `ModelPrediction` validation are in the semantic documentation registry with
  shape/dtype/device/frame/units/mask/cache/equivariance/gradient/distributed
  requirements as applicable.

## TDD RED evidence

Initial required command:

`conda run -n 3dmolecule python -m pytest tests/unit/models tests/integration/test_model_equivariance.py -v`

collected no tests and stopped with four expected collection errors:

- `ImportError: cannot import name 'ECloudFlowModel' from 'ecloudflow.models'`
- `ModuleNotFoundError: No module named 'ecloudflow.models.heads'`
- `ImportError: cannot import name 'AtomCountPredictor' from 'ecloudflow.models'`
- the integration test repeated the missing `ECloudFlowModel` public export.

The first implementation run exposed an invalid hand-built test halfedge that
crossed complexes. The canonical `MolecularState` correctly rejected it; the
fixture was repaired to use two independently derived within-complex pairs.

Additional behavior cycles were also observed RED before their minimal fixes:

- The Task 10 documentation test failed because all six required designated
  APIs were absent from `DESIGNATED_APIS`; after registry addition, it reported
  two missing shape topics, which were documented before GREEN.
- With the pocket translated 100 angstroms outside the cross cutoff, moving a
  ligand atom left atom logits exactly equal. This caught the missing ligand
  radius-message branch. Sparse ligand scalar/vector messages made it GREEN.
- All-zero atom, charge, bond, electron, and pocket features produced NaN input
  gradients from unguarded RMS square roots. Clamping mean-square values at the
  dtype epsilon made every asserted gradient finite.
- A pocket cache made by a parameter-diverged second model was accepted because
  the key covered architecture but not learned-feature ownership. Encoder
  ownership and parameter versions now make it fail clearly.
- BF16 pocket encoding raised `TypeError: Got unsupported ScalarType BFloat16`
  while hashing through NumPy. Hashing the raw `uint8` view preserves exact
  bytes/dtype and passes.

Every test names the mutation it catches and asserts real outputs with literal
or independently transformed expected values. No dependency mocks are used.

## Files changed

- `src/ecloudflow/models/layers.py`
- `src/ecloudflow/models/pocket_encoder.py`
- `src/ecloudflow/models/count_predictor.py`
- `src/ecloudflow/models/heads.py`
- `src/ecloudflow/models/backbone.py`
- `src/ecloudflow/models/ecloudflow.py`
- `src/ecloudflow/models/__init__.py`
- `tests/unit/models/test_layers.py`
- `tests/unit/models/test_pocket_encoder.py`
- `tests/unit/models/test_ecloudflow.py`
- `tests/integration/test_model_equivariance.py`
- `tests/unit/test_source_docs.py`
- `tools/check_python_docs.py`
- this report.

## Final commands and results

- `conda run -n 3dmolecule python -m pytest tests/unit/models
  tests/integration/test_model_equivariance.py -v`: **22 passed**, including
  deterministic CPU and available CUDA coverage.
- `conda run -n 3dmolecule python -m pytest -q`: **248 passed, 1 skipped**;
  the skip is the existing external xTB smoke test. The 14 warnings are existing
  third-party Matplotlib/PyParsing deprecations.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/models
  tests/unit/models tests/integration/test_model_equivariance.py
  tests/unit/test_source_docs.py tools/check_python_docs.py`: passed.
- `conda run -n 3dmolecule python -m ruff format --check
  src/ecloudflow/models tests/unit/models
  tests/integration/test_model_equivariance.py tests/unit/test_source_docs.py
  tools/check_python_docs.py`: 13 files already formatted.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/models tools/check_python_docs.py`: passed with no output.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/models`: success,
  no issues in 7 source files.
- `git diff --check`: passed; Git emitted only Windows LF-to-CRLF checkout
  notices for three already tracked files.

## Self-review and concerns

The test matrix covers proper rotation plus translation for Cartesian outputs,
the complete e3nn packed electron representation, every scalar head, cache
reuse/rejection, endpoint symmetry, mask/count semantics, null/task behavior,
empty and multi-complex inputs, malformed time/layout/cache values, all-zero and
ordinary finite gradients, BF16 cache bytes, and CUDA placement.

The only remaining operational concern is scale when `torch-cluster` is absent:
the native fallback creates per-complex pair index candidates with quadratic
index memory before radius filtering. It is correct and sparse in feature
channels for the tested installation, but production large-pocket training
should install the PyG-compatible `torch-cluster` wheel so compiled radius
queries are used. No correctness blocker remains within Task 10.

## Commit

All Task 10 source, tests, documentation-gate, and report changes are included
in the planned commit message `feat: add joint SE3 ECloudFlow backbone`; the
final commit identifier is reported in the task handoff because a commit cannot
contain its own final hash.
