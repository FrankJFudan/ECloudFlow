# Task 9 Report: Hybrid Stochastic Interpolant Paths

## Implemented conventions

Continuous tensors use the explicit direction `t=0` prior and `t=1` data:

\[
x_t = (1-a(t))x_0 + a(t)x_1 + g(t)\epsilon,
\qquad \epsilon \sim \mathcal N(0,I).
\]

The exact conditional targets for the same retained noise draw are

\[
u_t = a'(t)(x_1-x_0)+g'(t)\epsilon,
\qquad s_t=-\epsilon/g(t).
\]

`LinearBridge` uses `a(t)=t` and `g(t)=s*t*(1-t)`. `CosineBridge` uses
`a(t)=sin(pi*t/2)` and `g(t)=s*sin(pi*t)`. Both have analytic derivatives.
Score targets reject `t=0` and `t=1`; a configured `numerical_epsilon` only
limits the denominator for already-valid interior times.

Categorical paths use the normalized simplex interpolation

\[
q_t=(1-t)p_0+t\operatorname{onehot}(y).
\]

`t=0` is exactly the validated prior and `t=1` is exactly the data one-hot
endpoint. Class draws use `torch.multinomial` with an optional generator.
Fixed fragment rows are functionally replaced by exact one-hot/data values.
`endpoint_loss` is explicitly the data-endpoint (`t=1`) cross entropy and
averages only editable rows; fixed rows enter neither numerator nor denominator.

## RED evidence

Before implementation, `conda run -n 3dmolecule python -m pytest
tests/unit/process -v` failed at collection with three `ModuleNotFoundError`
errors for `ecloudflow.process`.

## GREEN evidence

- Focused process and source-documentation tests: 26 passed.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/process
  tests/unit/process tools/check_python_docs.py tests/unit/test_source_docs.py` passed.
- `conda run -n 3dmolecule python -m ruff format --check src/ecloudflow/process
  tests/unit/process tools/check_python_docs.py tests/unit/test_source_docs.py` passed.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/process` passed.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/process` passed.
- Full suite: 207 passed, 1 skipped (external xTB smoke test), 14 unrelated
  third-party deprecation warnings.

## Self-review and concerns

The APIs validate time, finite floating endpoints/logits/probabilities,
shape/dtype/device compatibility, class ranges, prior normalization, and mask
contracts without in-place writes. Prefix time broadcasting maps `[B]` to
`[B,...]`, and antithetic times use deterministic `u, 1-u` ordering. BF16 and
FP16 bridge/loss computations accumulate in float32; gradient tests cover
continuous masked velocity loss and categorical endpoint CE. The categorical
production formulation is the deterministic simplex path plus generator-driven
class realization rather than an additional random Dirichlet concentration;
this is intentional because it preserves exact endpoint/fixed-row semantics
and provides normalized probability targets directly.

## Fix Round 1

### Review findings addressed

- Score targets now use the exact expression `-epsilon / gamma(t)` whenever
  `abs(gamma(t)) >= numerical_epsilon`. Endpoint times and interior times below
  that threshold raise `ValueError`; no accepted target is denominator-clamped.
  `sample_times(..., mode="score")` is now the default and rejection-samples
  only score-trainable times. Explicit `mode="flow"` retains unrestricted flow
  time sampling.
- FP16/BF16 continuous samples retain the generating epsilon as float32 in
  `ContinuousSample.noise`. Their velocity and score targets are float32 and
  use that same epsilon, while sampled values remain endpoint dtype.
- Empty unmasked continuous reductions now produce differentiable finite zero,
  just as all-false masks do, instead of calling `mean` on an empty tensor.
- Categorical prior mass is evaluated in float32 or float64 with a strict
  fixed tolerance (float64 `1e-8`, float32 `1e-5`, low precision `4e-3` only
  for representational rounding). Valid priors and interpolation are normalized
  in stable precision; fixed rows remain exact zero/one rows.
- Float64 categorical probabilities remain float64 through `multinomial`; only
  actual FP16/BF16 distributions are promoted for sampling.
- Sampling RuntimeError behavior and abstract schedule tensor/endpoint,
  dtype/device, exception, mutation, determinism, and gradient contracts are
  now documented and covered by the semantic-document registry.

### Added regressions and GREEN evidence

- `t=1e-8` is rejected with default `numerical_epsilon=1e-6`; a nearby accepted
  time verifies bit-exact `-epsilon/gamma` without clamping.
- BF16 samples retain float32 epsilon and produce exact float32 formula targets
  with finite endpoint gradients.
- Empty unmasked and empty masked velocity losses produce finite differentiable
  float32 zero with zero gradients.
- BF16 `[0.30, 0.30, 0.45]` prior rejection, tight BF16 simplex normalization,
  and float64 tiny-probability `multinomial` input identity are tested.
- Focused process and source-doc tests: 31 passed.
- Ruff check/format, process documentation checker, and scoped Mypy passed.
- Full suite: 212 passed, 1 skipped (external xTB smoke), with 14 unrelated
  third-party deprecation warnings.

## Fix Round 2

### Review findings addressed

- Score-time validation now evaluates float64 candidates as float64. For FP16
  and BF16 outputs it uses the float32 target-computation dtype, exactly
  matching the dtype consumed by `targets`; default score-mode results are
  therefore target-safe in their effective output precision.
- The categorical prior normalization tolerance now uses `rtol=0`, so the
  documented absolute tolerance is the whole acceptance bound. Construction
  stores one canonical stable `path.prior`. Probability endpoints branch before
  interior normalization: `t=0` is bitwise the broadcast canonical prior and
  `t=1` is bitwise one-hot data.
- `_normalize_shape` now validates non-sequence shape objects explicitly with
  `TypeError`; its detailed API contract is added to the semantic-doc registry.
- Antithetic score sampling handles singleton and all odd flattened sizes
  without an empty concatenation. Pairs retain deterministic first-half
  `u`, second-half `1-u` ordering, followed by one independently safe odd draw.

### Added regressions and GREEN evidence

- A float64 `1.00000098e-6` boundary time is rejected according to its true
  float64 gamma rather than being rounded into float32 acceptance; a nearby
  float64 time verifies the exact score formula.
- Property-style score-time checks show every default returned FP32, FP64, and
  BF16 time can form finite velocity/score targets with same-dtype endpoints.
- Antithetic sizes `1`, `3`, and multidimensional odd total `(1, 3)` are
  deterministic, paired where applicable, and score-safe.
- Exact canonical prior/data endpoint checks and BF16 sum `1.0078125`
  rejection cover the tightened categorical contract.
- Focused process and source-doc tests: 40 passed.
- Ruff check/format, process documentation checker, and scoped Mypy passed.
- Full suite: 221 passed, 1 skipped (external xTB smoke), with 14 unrelated
  third-party deprecation warnings.

## Fix Round 3

### RED evidence

The previous exact-endpoint branch selected `prior`/one-hot values with
`torch.where`. Autograd therefore selected the endpoint branch and reported a
zero gradient with respect to `time`, instead of the intended one-sided affine
path derivative `one_hot(target) - prior`. The public `sample_times` docstring
also omitted the intentional `TypeError` thrown for a non-integer,
non-sequence shape.

### Changes

- `CategoricalPath.probabilities` now uses a straight-through forward-value
  correction: its forward values remain exactly canonical prior at `t=0` and
  exact one-hot data at `t=1`, while backwards uses the affine interpolation
  derivative. Interior forward rows retain stable normalization.
- `ContinuousPath.sample_times` now declares `TypeError` in its public Sphinx
  exception contract. It was added to the semantic documentation registry and
  a focused source-doc assertion verifies that field remains present.
- Added float32/float64 endpoint-gradient, exact endpoint, near-endpoint
  continuity, and normalized-interior regressions, while existing fixed-mask
  and simplex checks remain in the focused suite.

### GREEN evidence

- `conda run -n 3dmolecule python -m pytest tests/unit/process
  tests/unit/test_source_docs.py -v`: 42 passed.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/process
  tests/unit/process tests/unit/test_source_docs.py tools/check_python_docs.py`:
  passed.
- `conda run -n 3dmolecule python -m ruff format --check src/ecloudflow/process
  tests/unit/process tests/unit/test_source_docs.py tools/check_python_docs.py`:
  passed.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/process`: passed.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/process`: success,
  no issues in 4 source files.
- `conda run -n 3dmolecule python -m pytest -v`: 223 passed, 1 skipped
  (external xTB smoke), with 14 unrelated third-party deprecation warnings.

### Changed files and self-review

Changed `src/ecloudflow/process/categorical.py`,
`src/ecloudflow/process/continuous.py`, `tools/check_python_docs.py`,
`tests/unit/process/test_categorical.py`, and `tests/unit/test_source_docs.py`.
The value correction is intentionally detached only for the exact-forward
endpoint correction; it neither mutates inputs nor changes fixed-mask handling,
prior validation, retained float64 behavior, or interior simplex values. The
endpoint-gradient tests use a non-constant class weighting so the analytically
nonzero derivative cannot be hidden by simplex-sum cancellation.

## Fix Round 4

### RED evidence

At reviewed head `93e397f`, adversarial priors were added for the two retained
probability dtypes. The float32 construction produced a canonical prior sum of
`0.9999999403953552`; the float64 construction produced
`0.9999999999999999`. Normalizing the closest representable interior value
again changed a component by `2.9802322387695312e-08` and
`2.7755575615628914e-17`, respectively, even though the exact `t=0` branch
returned the stored prior.

Command:

`conda run -n 3dmolecule python -m pytest tests/unit/process/test_categorical.py -k adversarial -vv`

Output: `2 failed, 9 deselected`; both failures reported the non-unit stored
sums above before reaching the closest-interior equality assertion.

### Changes and numerical contract

- `CategoricalPath.probabilities` now uses the construction-time canonical prior
  directly in the affine expression at every time. It no longer applies a
  distinct interior division followed by endpoint-only straight-through value
  correction.
- Exact `t=0` and `t=1` values now arise from the same affine expression as the
  interior. Its autograd derivative is therefore exactly the documented
  `one_hot(target) - prior` affine derivative at both endpoints.
- The accepted prior is still validated with the existing strict, dtype-specific
  absolute tolerances and normalized exactly once into the retained float32 or
  float64 probability dtype. No acceptance tolerance was widened.
- New adversarial float32 and float64 tests use `nextafter(0, 1)` and
  `nextafter(1, 0)`. They require bitwise equality at `t=0`, `t=1`, and the
  closest value after zero where affine rounding cannot yet change a component;
  the near-one and simplex checks are bounded by machine epsilon. The same
  priors also exercise both endpoint gradients. Existing BF16, strict rejection,
  multinomial dtype, and exact fixed-mask regressions remain active.

### GREEN evidence

- `conda run -n 3dmolecule python -m pytest tests/unit/process
  tests/unit/test_source_docs.py -v`: `44 passed in 1.84s`.
- `conda run -n 3dmolecule python -m ruff check src/ecloudflow/process
  tests/unit/process tests/unit/test_source_docs.py tools/check_python_docs.py`:
  `All checks passed!`
- `conda run -n 3dmolecule python -m ruff format --check
  src/ecloudflow/process tests/unit/process tests/unit/test_source_docs.py
  tools/check_python_docs.py`: `9 files already formatted`.
- `conda run -n 3dmolecule python -m mypy src/ecloudflow/process`:
  `Success: no issues found in 4 source files`.
- `conda run -n 3dmolecule python tools/check_python_docs.py
  src/ecloudflow/process`: exit code 0 with no output.
- `conda run -n 3dmolecule python -m pytest -v`: `225 passed, 1 skipped,
  14 warnings in 15.15s`; the skip is the external xTB smoke test and all
  warnings are the existing third-party Matplotlib/PyParsing deprecations.

A preliminary attempt to launch multiple `conda run` gates concurrently hit a
Windows temporary-file collision (`__conda_tmp_14799.txt`) in four launchers;
all required gates were then rerun serially with the successful outputs above.

### Changed files and self-review

Changed `src/ecloudflow/process/categorical.py`,
`tests/unit/process/test_categorical.py`, and this report. The final formulation
is smaller than the round-3 endpoint correction and has no endpoint-specific
forward representation. Convex interpolation preserves non-negativity, and its
sum differs from mathematical one only by retained-dtype reduction rounding,
which the regressions bound at machine-epsilon scale for the adversarial probes.
Exact fixed rows remain supplied by the unchanged functional mask replacement.
No mutation, device, target-shape, validation, sampling, loss, or low-precision
code path changed. No known concern remains within the Task 9 contract.
