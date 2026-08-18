# ECloudFlow Design Specification

**Status:** Approved in interactive design review; awaiting review of this written specification  
**Date:** 2026-08-18  
**Project:** Pocket-conditioned, electron-cloud-aware 3D ligand generation and lead optimization  
**Target hardware:** Local smoke tests on one RTX 4060 8GB; production training and evaluation on 4× NVIDIA H100 80GB

## 1. Purpose and delivery boundary

ECloudFlow is a research-grade, runnable deep-learning codebase that generates complete ligand molecules directly inside a supplied protein pocket. A generated ligand contains atom types, formal charges, bond orders, three-dimensional binding-pose coordinates, and an electron-cloud latent representation. The same checkpoint supports de novo generation and fragment-conditioned lead optimization.

The first delivery is a complete research implementation, not a claim of a converged state-of-the-art checkpoint. It must include modular model code, preprocessing, staged training, multi-GPU execution, sampling, chemical constraints, evaluation, visualization, tests, a ten-minute quick start, and detailed theory documentation. It must complete a tiny end-to-end train → sample → evaluate → report smoke workflow locally. Full-data configurations target the 4×H100 server.

The scientific hypotheses to test are:

1. Pocket and ligand electron-field latents improve pocket conditioning and binding-pose quality beyond conventional atom and residue features.
2. A stochastic-interpolant model using flow matching for the main path and a score-based terminal corrector retains diffusion-like sample quality with fewer network evaluations.
3. Constraint-aware trajectory updates and exact final graph decoding improve chemical validity without silently repairing or replacing the generated structure.
4. Joint masked-task training supports de novo generation, fragment growth, fragment linking, fragment replacement, and multi-fragment integration with one model.

These hypotheses require ablation experiments. Documentation must describe them as hypotheses until supported by measured results.

## 2. Reference systems and attribution

The design draws on the following systems while remaining an independently structured repository:

- DiffGui: pocket-conditioned equivariant atom/bond diffusion, classifier-free property guidance, bond guidance, and fragment-conditioned generation.
- ECloudGen: xTB electron-density construction, electron-cloud latent diffusion, and electron-cloud-to-molecule concepts. Necessary reused or adapted portions may be copied with clear source attribution as authorized by the project owner.
- PropMolFlow: geometry-complete continuous/discrete flow matching, categorical simplex paths, and efficient ODE sampling.
- JODO: joint 2D molecular graph and 3D coordinate modeling, formal charges, and accelerated diffusion solvers.
- CoCoGraph: data-derived valence rules, action-level constrained graph updates, connectivity checks, and inpainting masks.
- FLOWR and other current pocket-conditioned flow systems: efficiency and fragment-generation baselines that establish the minimum comparison standard in 2026.

`THIRD_PARTY_NOTICES.md` will identify reused files or algorithms, upstream repository and paper, license when available, and local modifications. No reference repository will be modified by ECloudFlow development.

## 3. Task definition

Let the protein pocket be

\[
P=(R^p,A^p,U^p),
\]

where \(R^p\in\mathbb{R}^{N_p\times3}\) are pocket coordinates, \(A^p\) are atom identities, and \(U^p\) contains residue type, backbone flag, partial charge, donor/acceptor state, aromaticity, hydrophobicity, and optional metal annotations.

A complete ligand is

\[
M=(R,A,Q,B),
\]

where \(R\in\mathbb{R}^{N\times3}\) are coordinates in the pocket frame, \(A\) are atom types, \(Q\) are formal charges, and \(B\) is a symmetric bond-order matrix with no self-edges. Training targets use Kekulé single, double, and triple bond orders; aromaticity is recovered by sanitization after decoding.

The joint generated state is

\[
X=(R,A,Q,B,z_l),
\]

and its condition is

\[
C=(P,z_p,F,y).
\]

Here \(z_p\) is a learned equivariant pocket electron-field latent, \(z_l\) is a jointly generated ligand electron-field latent, \(F\) is an optional fragment condition, and \(y\) contains optional property or protein–ligand interaction targets. The model approximates

\[
p_\theta(M,z_l\mid P,z_p,F,y).
\]

The given pocket is not generated. Its continuous electron field and latent representation are deterministically generated from the supplied structure. The ligand electron latent is sampled jointly with the molecular graph and binding pose.

## 4. Coordinate convention and SE(3) contract

Every complex is centered at the pocket centroid during preprocessing. The centering translation and original coordinate-frame metadata are retained. Sampling occurs in this centered pocket frame, and final ligand coordinates are transformed back into the original PDB frame. The model does not generate a free conformer and redock it to masquerade as a generated pose.

For a proper rotation \(G\in SO(3)\) and translation \(b\), the required behavior is

\[
f_\theta(GR^p+b,GR+b)=Gf_\theta(R^p,R)+b
\]

for coordinate/vector outputs, while atom types, charges, bond probabilities, and scalar properties remain invariant. Reflection invariance is not forced because chirality must be preserved. Random rotations are augmentation, not the mechanism that creates equivariance.

Equivariance is obtained by:

- centering coordinates;
- using relative displacement vectors in message passing;
- representing geometric features with irreducible spherical-tensor channels;
- updating coordinates using equivariant vector outputs;
- representing electron fields with radial bases and real spherical harmonics;
- testing transformed inputs against transformed outputs numerically.

## 5. Hybrid electron-field representation

### 5.1 Ligand reference density

For supported ligands, xTB produces an electron-density cube after standardized protonation, formal-charge assignment, and conformer validation. Density is retained at multiple resolutions for supervision and visualization, then projected around ligand atoms into radial basis functions and real spherical harmonics:

\[
c_{inlm}=\int \rho(r_i+r)R_n(\lVert r\rVert)Y_{lm}(\hat r)\,dr.
\]

The \(l=0\) components are rotational scalars; \(l>0\) components transform by the corresponding irreducible representation. The coefficients form per-atom equivariant latent tokens. Fixed-axis voxels are therefore not the model's primary representation.

If xTB fails, the structure may remain usable for graph/geometry training, but `qm_mask=0` disables QM-density losses. The pipeline records the command, version, charge, multiplicity, stderr category, and failure status. It never substitutes approximate density while claiming a QM label.

### 5.2 Pocket field

Running xTB on every full pocket is too costly and sensitive to protonation, metals, truncation, and net charge. The default pocket field is a continuous physically informed approximation assembled from element-specific Gaussian density bases plus partial-charge, donor, acceptor, hydrophobic, and aromatic channels. An equivariant field tokenizer converts it to \(z_p\).

An optional refined-pocket path runs xTB on carefully cropped, protonated, charge-defined pockets and uses those densities as auxiliary supervision. This path is opt-in and never a prerequisite for standard PDBBind training.

### 5.3 Graph–field cycle consistency

The ligand field decoder produces \(\hat\rho(z_l)\). A differentiable analytical builder produces \(\tilde\rho(M)\) from predicted atoms, coordinates, and charges. The training objective compares them through:

- multiresolution density L1/Huber error;
- density-gradient error;
- integrated electron-count error;
- dipole and optional quadrupole error;
- atom-centered latent consistency.

This prevents \(z_l\) from becoming a free nuisance latent unrelated to the generated chemistry.

## 6. Model architecture

### 6.1 Pocket encoder

The pocket encoder processes \((P,z_p)\) once using SE(3)-equivariant self-attention/message passing. Its outputs contain invariant scalar features and equivariant vector/tensor features. Because pocket encoding does not depend on flow time or the noisy ligand, it is cached and reused across all time steps and all molecules generated for the same pocket.

### 6.2 Ligand state and prior

The ligand atom count is predicted from pocket volume/field features, property targets, and fragment size. The initial coordinate prior is an anisotropic cavity-aware distribution supported inside the pocket rather than an unconstrained isotropic cloud far from the binding site. Fragment atoms occupy their supplied pose exactly. Free atom types, charges, bonds, and electron tokens start from learned or dataset-derived priors.

### 6.3 Joint equivariant backbone

The ligand decoder is a dual-stream SE(3)-equivariant transformer:

1. ligand self-attention couples noisy ligand atoms, bonds, coordinates, and electron tokens;
2. pocket-to-ligand cross-attention injects cached pocket graph and electron-field representations;
3. optional property, interaction, fragment-task, and time embeddings modulate each block;
4. graph, coordinate, charge, bond, electron velocity, endpoint, score, and auxiliary interaction heads share the backbone.

The production `large` configuration uses roughly 12–16 equivariant blocks and is sized for 4×H100. `base` supports ablations. `tiny` preserves the same interfaces for local smoke tests.

### 6.4 Stochastic interpolant

For a continuous target \(x_1\) and prior \(x_0\), including free coordinates and ligand electron latents, define

\[
x_t=\alpha(t)x_0+\beta(t)x_1+\gamma(t)\epsilon,
\quad \epsilon\sim\mathcal N(0,I),
\]

with \(\alpha(0)=1\), \(\beta(0)=0\), \(\alpha(1)=0\), \(\beta(1)=1\), and \(\gamma(0)=\gamma(1)=0\). A default path is \(\alpha=1-t\), \(\beta=t\), with a configurable interior noise schedule. The conditional velocity target is

\[
u_t=\dot\alpha(t)x_0+\dot\beta(t)x_1+\dot\gamma(t)\epsilon.
\]

The flow loss over editable components is

\[
\mathcal L_{flow}=\mathbb E\left[\left\|m_{free}\odot(v_\theta(x_t,t,C)-u_t)\right\|_2^2\right].
\]

The shared score head learns the denoising target on interior times:

\[
\mathcal L_{score}=\mathbb E\left[\left\|\gamma(t)s_\theta(x_t,t,C)+\epsilon\right\|_2^2\right].
\]

Atom, charge, and bond variables follow simplex/Dirichlet probability paths from their priors to one-hot endpoints. Endpoint cross-entropy and categorical flow objectives are both exposed through a stable interface; the selected production formulation must be fixed in the implementation plan and covered by normalization tests.

At inference, an ODE solver follows the learned velocity for the main trajectory. A configurable terminal score corrector adds only the final \(K\) stochastic refinement steps. Setting \(K=0\) yields the fastest deterministic flow path.

## 7. Fragment-conditioned generation

One checkpoint supports these modes:

- `de_novo`: no fixed ligand atoms;
- `grow`: preserve one supplied fragment and add atoms/bonds;
- `link`: preserve two or more fragments and generate linkers;
- `replace`: preserve the retained region and regenerate a masked subgraph;
- `merge`: jointly integrate multiple supplied fragments into a compatible scaffold.

Each `FragmentCondition` contains:

- `fixed_atom_mask`;
- `fixed_bond_mask`;
- `fixed_coord_mask`;
- `attachment_mask`;
- component and task identifiers.

At every ODE, score-correction, and constraint-projection step, fixed atom identities, charges, internal bonds, and coordinates are copied back exactly. Fragment preservation is thus an algorithmic invariant rather than a soft penalty. The default input contract requires fragments to contain coordinates already placed in the pocket. A separate initialization interface handles fragments without a supplied pose; it does not silently alter the fixed-pose mode.

Training fragment tasks are created online using BRICS cuts, Bemis–Murcko scaffolds, ring-aware cuts, linker masking, and multi-component masks. Approximately 30–50% of paired-complex batches use a fragment task, configurable by stage. Classifier-free task/condition dropout lets one model retain de novo capability.

## 8. Data system

### 8.1 Sources

- PDBBind v2020 is the primary high-quality paired dataset.
- CrossDocked2020 expands pocket and pose coverage.
- GEOM-Drugs, ChEMBL, or another explicitly licensed ligand corpus supports ligand graph–geometry–electron pretraining.
- The existing DiffGui processed PDBBind LMDB is accepted through a read-only compatibility importer.

Paths are configuration values; no machine-specific path is committed.

### 8.2 Leakage-controlled splits

Default benchmark splits group complexes by protein sequence and/or pocket similarity. Highly homologous proteins and near-identical ligands must not cross train/test boundaries. The split builder stores grouping method, thresholds, seed, source identifiers, and hashes. Published baseline splits remain available only as named compatibility profiles.

### 8.3 `ComplexSample` contract

Each sample contains:

- pocket graph and coordinates;
- ligand graph, charges, bonds, and crystal coordinates;
- pocket and ligand electron-field tensors/coefficients and masks;
- affinity and physicochemical properties when available;
- fragment condition when sampled;
- centering/inverse-transform metadata;
- source files, hashes, tool versions, and preprocessing status.

The contract uses explicit typed fields rather than arbitrary dictionary keys.

### 8.4 Storage and loading

Processed production data use WebDataset tar shards of approximately 0.5–2GB, with graph tensors, field coefficients, and metadata stored per sample. The loader shards first by distributed rank and then by worker, so an epoch contains no duplicate or omitted samples. Batches are bucketed by pocket and ligand node counts to control padding and memory variance. Persistent workers, prefetching, local shard cache, and deterministic epoch seeding keep H100s fed.

xTB preprocessing is CPU-parallel and resumable through a manifest. Each sample is written atomically, and a failed record never corrupts a completed shard.

## 9. Training curriculum and objectives

### 9.1 Stages

1. **Electron tokenizer:** learn ligand density reconstruction and pocket-field encoding, including electron count and multipoles.
2. **Ligand-only joint pretraining:** learn graph, charge, bond, coordinate, and ligand-field paths on a large ligand corpus.
3. **Pocket-conditioned multitask training:** train de novo and fragment modes on PDBBind/CrossDocked with pocket interactions and optional affinity/property conditions.
4. **High-quality fine-tuning:** use a curated high-resolution cocrystal subset, lower learning rate, EMA, and stronger pose/electron consistency weights.

Each stage is independently resumable and emits a checkpoint compatible with the next stage.

### 9.2 Total loss

\[
\mathcal L=
\lambda_v\mathcal L_{flow}+
\lambda_s\mathcal L_{score}+
\lambda_{cat}\mathcal L_{discrete}+
\lambda_\rho\mathcal L_{ecloud}+
\lambda_{chem}\mathcal L_{chem}+
\lambda_{int}\mathcal L_{interaction}.
\]

`L_chem` includes differentiable expected-valence overflow, element/bond-conditioned length priors, short-range ligand and protein clashes, ring-strain surrogates, and a connectivity surrogate. `L_interaction` includes pocket contact or interaction-fingerprint supervision and a heteroscedastic affinity auxiliary loss where labels exist.

Every component is normalized by a running scale before its configured weight is applied. Weights and warm-up schedules are explicit configuration values and logged. The model must report each raw and weighted loss separately.

### 9.3 Classifier-free conditioning

Property, affinity, interaction, and fragment task embeddings are independently dropped with configurable probabilities during training. Sampling can combine conditional and null predictions with a configured guidance scale. Guidance never overrides fixed fragment fields.

## 10. Distributed execution

The default production strategy is PyTorch DistributedDataParallel launched by `torchrun` with NCCL:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m ecloudflow.cli train \
  experiment=pdbbind_large \
  trainer.strategy=ddp \
  trainer.precision=bf16-mixed
```

Defaults for the H100 server are BF16 mixed precision, TF32 matrix multiplication, fused AdamW where compatible, node-count bucketed batches, gradient accumulation, persistent workers, asynchronous prefetch, and selective activation checkpointing. FP8 is experimental and limited to compatible scalar MLP/attention operations; it is not required for correctness. FSDP is optional for models that no longer fit comfortably per H100, not the default for the initial large model.

Checkpoints include model, optimizer, scheduler, EMA, global step, epoch, per-rank RNG state, data-stream position/epoch, resolved configuration, dataset manifest hash, preprocessing version, and code revision. A resumed deterministic smoke run must match an uninterrupted run within documented floating-point tolerance.

Distributed validation reduces sufficient statistics across ranks and writes artifacts only from rank zero. WebDataset sample IDs are checked for duplicate or missing coverage in distributed smoke tests.

## 11. Constraint-aware sampling

### 11.1 Sampling sequence

1. Validate and standardize pocket and optional fragment files.
2. Center the pocket, build/cache \(z_p\), predict atom count, and create a cavity-aware prior.
3. Integrate continuous and categorical paths with Euler or Heun.
4. After every integration step, clamp fragments and project chemical/geometric constraints.
5. Optionally apply the final \(K\) score-correction steps, each followed by the same projections.
6. Discretize atom and charge types and solve the final bond graph under exact constraints.
7. Sanitize, audit, inverse-transform coordinates, optionally relax, dock, rank, and report.

### 11.2 Trajectory constraints

Hard or projected constraints include:

- symmetric bond matrix and no self-bonds;
- fixed fragment invariants;
- allowed element–formal-charge combinations;
- atom-specific maximum degree and valence feasibility masks;
- coordinate support inside or near the pocket cavity;
- no impossible new bonds to saturated atoms.

Soft differentiable energies include:

- bond-length, angle, and dihedral priors;
- ligand internal and protein–ligand short-range repulsion;
- small/large ring and strain penalties;
- electron-density, electrostatic, hydrogen-bond, hydrophobic, and hotspot complementarity;
- optional classifier-free affinity/property guidance.

### 11.3 Exact final graph decoding

The preferred final decoder uses OR-Tools CP-SAT to maximize model atom/bond log probability subject to fixed bonds, allowed bond orders, valence bounds, and single-component connectivity. A bounded timeout is required. A constrained greedy decoder is available as a documented fallback and must return a lower-confidence status. The decoder works on Kekulé bond orders; RDKit performs aromaticity perception after a feasible graph is found.

No decoder is described as guaranteeing universal chemical validity for metals, radicals, or chemistry outside the configured vocabulary. Unsupported cases return explicit status codes.

### 11.4 Sampling profiles

- `fast`: approximately 20 Euler steps, no score corrector, high-throughput screening.
- `balanced`: approximately 40 Heun steps and a small terminal corrector; default.
- `quality`: approximately 80–100 Heun steps, more terminal correction, and optional constrained relaxation.

Exact values are configuration defaults, not hard-coded constants.

### 11.5 Raw and relaxed outputs

The model output is saved as `raw.sdf`. Optional MMFF/UFF or external constrained relaxation is saved separately as `relaxed.sdf`; fixed fragment atoms remain constrained. Evaluation reports raw and relaxed metrics separately. Docking is an evaluation/ranking operation and never overwrites the raw generated pose.

### 11.6 Failure semantics

Each attempt has one of `valid`, `rejected`, or `failed` plus structured reason codes such as valence infeasible, disconnected graph, clash, unsupported element, CP-SAT timeout, sanitization failure, or docking failure. Resampling is bounded. The software never loops indefinitely, silently changes a fixed fragment, or substitutes another dataset sample.

## 12. Requested sample count, ranking, and molecular IDs

The primary command is:

```bash
ecloudflow sample pocket.pdb --num-molecules 100 --profile balanced
```

`num_molecules` means the target number of valid, unique final molecules. Uniqueness is based on canonical isomeric SMILES by default. `max_attempts` defaults to five times the target and is configurable. If the target is not reached, all completed results are saved and the shortfall, attempts, duplicate rate, rejection rate, and failure reasons are reported.

Within each pocket, successfully docked unique molecules are sorted deterministically by:

1. Vina docking score ascending, because more negative is better;
2. QED descending;
3. synthetic accessibility score ascending, using the conventional 1–10 SA score where lower is easier;
4. canonical isomeric SMILES lexicographically.

Formal molecule IDs follow

```text
<POCKET_ID>-<RANK:06d>
```

For example, `3ZTX-000001` is the best successfully docked candidate for pocket `3ZTX`. Each pocket starts at one. Candidates with docking failure or with docking disabled retain temporary attempt IDs, appear in failure/unranked outputs, and do not receive a formal rank ID. The ID contains no `ECLF` component.

The run writes:

- `samples.csv` for convenient inspection;
- `samples.parquet` with complete typed records;
- `summary.xlsx` with `ranked`, `failed`, and `aggregate` sheets;
- `ranked.sdf` in the same rank order;
- `summary.json` with aggregate statistics;
- `report.html` with sortable tables, plots, and top-molecule panels.

Every ranked row and SDF record contains at minimum rank, molecule ID, pocket ID, canonical/isomeric SMILES, SA, QED, docking score, generation status, raw/relaxed artifact paths, seed, and provenance. Aggregate statistics include count, mean, standard deviation, median, minimum, maximum, first quartile, and third quartile for SA, QED, and docking score.

## 13. Evaluation protocol

### 13.1 Chemical validity

- RDKit sanitization validity;
- PoseBusters molecular and complex validity;
- allowed valence and formal-charge validity;
- connected-molecule rate;
- atom stability and whole-molecule stability;
- PAINS/medicinal-chemistry filters where requested.

### 13.2 Distribution quality and diversity

- uniqueness and canonical-SMILES novelty;
- scaffold novelty and Bemis–Murcko diversity;
- internal Tanimoto diversity and nearest-neighbor similarity;
- Fréchet ChemNet Distance when the dependency is available;
- descriptor KL/Jensen–Shannon distances;
- fragment and scaffold distribution similarity.

### 13.3 Three-dimensional quality

- bond-length, angle, and dihedral distribution JSD;
- ring-size distribution;
- intra-ligand clashes and strain energy;
- raw-to-relaxed RMSD;
- PoseBusters 3D checks and compatible Validity3D/GenBench3D metrics;
- symmetry-corrected pose RMSD only where a meaningful matched reference exists.

### 13.4 Pocket compatibility

- Vina/QVina score under a recorded protocol;
- protein–ligand interaction fingerprint recovery;
- hydrogen bond, salt bridge, hydrophobic, and aromatic contact statistics;
- shape/volume occupancy and steric clash rate;
- electrostatic or electron-field complementarity;
- per-pocket success rate and best-of-N statistics.

### 13.5 Electron-field quality

- density L1/RMSE and correlation at multiple resolutions;
- density-gradient agreement;
- electron-count and dipole errors;
- graph–field cycle consistency;
- optional electrostatic-potential similarity.

### 13.6 Conditional and fragment success

- requested property MAE and tolerance success rate;
- exact fixed-atom, fixed-bond, and fixed-coordinate preservation;
- valid attachment rate;
- growth, linking, replacement, and merge completion rate;
- linker length and geometry distributions;
- retained interaction recovery.

### 13.7 Efficiency

- network function evaluations;
- wall-clock seconds per molecule and molecules per second;
- peak GPU memory;
- preprocessing throughput;
- valid unique molecule yield per attempt and per GPU-hour;
- 1/2/4 GPU throughput, speedup, and scaling efficiency.

### 13.8 Statistical reporting and fairness

Metrics use per-pocket macro averages so pockets with more surviving molecules do not dominate. Reports include bootstrap 95% confidence intervals, multiple seeds, and paired pocket-level tests where appropriate. Raw and relaxed poses are separate. Baselines use the same splits, atom vocabularies when possible, number of requested samples, docking box/protocol, and postprocessing policy. Models that do not solve the same conditional task are labeled as partial comparisons rather than forced into a misleading ranking.

Ablations must cover electron latents, score corrector, hard constraints, fragment multitask training, NFE/profile, model scale, and distributed throughput.

## 14. Visualization

The interactive viewer uses Py3Dmol and/or Plotly to display:

- pocket cartoon/surface and ligand sticks;
- fixed fragments with distinct highlighting;
- hydrogen bonds, salt bridges, aromatic contacts, and clashes;
- pocket and ligand electron-density isosurfaces;
- density differences and optional electrostatic surfaces;
- raw versus relaxed poses;
- selected sampling trajectory frames.

Static publication figures use deterministic, colorblind-safe styles and export SVG, PDF, and high-resolution PNG. Standard panels include ECDF/violin plots with confidence intervals, per-pocket paired comparisons, geometry JSD heatmaps, property distributions, top-molecule grids, and quality–speed Pareto plots. Radar plots are not a default because their scaling can mislead.

Commands are:

```bash
ecloudflow visualize molecule runs/job --id 3ZTX-000001
ecloudflow visualize ecloud runs/job --id 3ZTX-000001
ecloudflow report runs/benchmark --format paper
```

## 15. User interface and configuration

The five primary workflows are:

```bash
ecloudflow doctor
ecloudflow data prepare data=pdbbind
ecloudflow train experiment=pdbbind_large
ecloudflow sample pocket.pdb -n 100
ecloudflow evaluate runs/job
ecloudflow report runs/job
```

The Python API uses the same application services:

```python
from ecloudflow import ECloudFlowPipeline

pipeline = ECloudFlowPipeline.from_pretrained(
    "checkpoints/ecloudflow-large.ckpt"
)
result = pipeline.generate(
    pocket="3ztx_pocket.pdb",
    fragment="hit_fragment.sdf",
    num_molecules=100,
    profile="balanced",
)
result.to_excel("summary.xlsx")
```

Configuration is grouped into `model`, `data`, `train`, `sample`, `evaluate`, and `visualization` YAML files. `tiny/base/large` and `fast/balanced/quality` are composable presets. Typed dataclass or Pydantic schemas define defaults, bounds, enums, and help text. Unknown keys are errors. CLI dot-list overrides may change any field. The resolved configuration is printed and saved with every run.

```bash
ecloudflow config show experiment=pdbbind_large
ecloudflow config explain model.hidden_dim
ecloudflow sample pocket.pdb -n 500 sample.corrector_steps=4
```

Common workflows expose a small top-level parameter set; specialized scientific parameters remain available in nested configuration without cluttering the basic interface.

## 16. Repository structure

```text
EcloudFlow/
├── pyproject.toml
├── environment.yml
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── configs/
│   ├── model/{tiny,base,large}.yaml
│   ├── data/{pdbbind,crossdocked,ligand_pretrain}.yaml
│   ├── train/{stage1,stage2,stage3,stage4}.yaml
│   ├── sample/{fast,balanced,quality}.yaml
│   └── experiment/*.yaml
├── src/ecloudflow/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── ecloud/
│   ├── chemistry/
│   ├── models/
│   ├── process/
│   ├── training/
│   ├── sampling/
│   ├── evaluation/
│   └── visualization/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── distributed/
│   └── fixtures/
├── examples/
└── docs/
    ├── theory.md
    ├── data.md
    ├── training.md
    ├── sampling.md
    ├── evaluation.md
    ├── configuration.md
    ├── distributed.md
    ├── visualization.md
    └── reproducibility.md
```

Files remain focused: parsing does not train models, model modules do not invoke docking, evaluation does not mutate generated molecules, and CLI modules delegate to application services rather than duplicate logic.

### 16.1 Python source documentation standard

All Python source-code comments and docstrings must be written in English, including inline comments, module docstrings, class docstrings, public APIs, test comments, and explanatory notes. User-facing documentation may be multilingual, but Python comments must not mix Chinese text into English source files.

Important functions and methods require detailed Sphinx/reStructuredText-style docstrings comparable to the approved reference image. This applies to public APIs, tensor transformations, electron-field construction and projection, stochastic-interpolant kernels, loss functions, equivariant coordinate updates, fragment masking, chemical projection and exact decoding, distributed data logic, checkpoint operations, xTB/Vina adapters, ranking, and report generation. A detailed docstring must explain:

- the scientific or algorithmic purpose, including why the operation is needed;
- every parameter through `:param <name>:` entries;
- tensor shape, dtype, device, units, coordinate frame, masks, and accepted ranges where relevant;
- the return structure and meanings through `:return:` and `:rtype:`;
- raised errors through `:raises <Exception>:`;
- invariants, side effects, numerical stability concerns, and distributed behavior when relevant.

For example, a core tensor function follows this form:

```python
def project_electron_field(coefficients, centers, basis, mask):
    """Project atom-centered electron coefficients onto a spatial field.

    The projection preserves the irreducible-representation layout used by
    the equivariant backbone and ignores padded atoms through ``mask``.

    :param coefficients: Electron coefficients with shape ``[B, N, C]``.
        The tensor must use the same floating-point dtype and device as
        ``centers``.
    :param centers: Atom centers with shape ``[B, N, 3]`` in angstroms,
        expressed in the centered pocket coordinate frame.
    :param basis: Precomputed radial and spherical-harmonic basis values.
    :param mask: Boolean tensor with shape ``[B, N]``; ``True`` marks a
        physical atom and ``False`` marks padding.
    :return: A dictionary containing the reconstructed density, density
        gradients, and integrated electron count.
    :rtype: dict[str, torch.Tensor]
    :raises ValueError: If shapes, devices, or irreducible layouts disagree.
    """
```

Small private helpers may use shorter English docstrings when their contract is obvious. Comments should explain scientific intent, invariants, or non-obvious decisions instead of restating individual lines. Automated source-quality tests scan Python comment tokens and docstring nodes for disallowed CJK text and verify that designated core APIs contain the required documentation fields.

## 17. Error handling and observability

`ecloudflow doctor` checks Python packages, CUDA/GPU visibility, xTB, Vina/QVina, OpenBabel where configured, dataset paths, write permissions, and checkpoint compatibility. Missing optional tools disable only the named metrics after an explicit warning; missing required tools fail before an expensive run begins.

No broad exception handler may replace a bad sample with sample zero or silently continue. Structured exceptions include sample/run ID, stage, source path, reason category, and remediation. Data skips are written to manifests. NaN/Inf batches produce bounded diagnostic artifacts and stop once the configured threshold is exceeded. Outputs use temporary files and atomic rename.

Logging supports console, CSV, TensorBoard, and optional Weights & Biases. Credentials are supplied through environment variables or local ignored configuration; no key is committed.

## 18. Test and acceptance matrix

### 18.1 Unit and scientific-invariant tests

- rotation/translation equivariance for pocket encoder, backbone, coordinate head, and electron irreps;
- invariance of scalar atom/bond outputs;
- electron-density projection/reconstruction and electron-count conservation;
- coordinate centering and exact inverse transform;
- fragment atoms, bonds, and coordinates unchanged after every solver step;
- valence masks, symmetric bonds, no self-edges, CP-SAT feasibility, and timeout handling;
- deterministic ranking and IDs such as `3ZTX-000001`;
- SA directionality and Vina ascending-sort convention;
- config bounds, unknown-key rejection, and resolved-config serialization.
- English-only Python comments/docstrings and required detailed fields on designated core APIs.

### 18.2 Integration tests

- parse a tiny pocket/ligand fixture and build both fields;
- run one tokenizer and one joint-training optimizer step;
- save and restore a checkpoint;
- run tiny de novo and each fragment mode;
- execute sample → evaluate → rank → report end to end;
- ensure CLI and Python API produce the same typed result schema;
- mock xTB and Vina adapters, plus one opt-in real-tool smoke test when installed.

### 18.3 Distributed tests

- two-process CPU/Gloo smoke test where CUDA is unavailable;
- 2-rank and 4-rank NCCL tests on the server;
- no duplicate or missing sample IDs across ranks/workers;
- rank-zero-only artifact writing;
- interrupted versus resumed loss/weight equivalence within tolerance;
- 1/2/4 GPU throughput and memory regression report.

### 18.4 Delivery acceptance

The implementation is accepted when:

1. installation and `ecloudflow doctor` succeed in the documented environment;
2. the local tiny end-to-end workflow passes;
3. all scientific-invariant and ordinary tests pass;
4. the 4×H100 launch configuration and distributed smoke protocol are documented and runnable on the server;
5. de novo and all four fragment modes share one pipeline and preserve fixed fragments exactly;
6. sampling count, bounded attempts, docking-based ranking, IDs without `ECLF`, CSV/Parquet/Excel/SDF output, and aggregate statistics match this specification;
7. README and theory/data/training/sampling/evaluation/visualization/distributed documents contain no incomplete placeholders;
8. no unmeasured SOTA or binding-quality claim is presented as a result.
9. every Python comment/docstring is English, and designated important functions pass the detailed docstring contract check.

## 19. Documentation deliverables

`README.md` is the operational entry point and contains installation, data preparation, one-GPU and four-GPU training, de novo sampling, fragment sampling, evaluation, reporting, troubleshooting, repository map, and citation instructions.

`docs/theory.md` derives the electron representation, stochastic interpolant, continuous/discrete objectives, SE(3) behavior, fragment masks, constraint projections, exact graph decoder, and guidance. Other documents isolate operational concerns so users can find answers without reading source code.

## 20. Primary references

- Hu et al., “Target-aware 3D molecular generation based on guided equivariant diffusion,” *Nature Communications* 16, 7928 (2025), DOI: 10.1038/s41467-025-63245-0.
- Zhang et al., “ECloudGen: leveraging electron clouds as a latent variable to scale up structure-based molecular design,” *Nature Computational Science* 5, 1017–1028 (2025), DOI: 10.1038/s43588-025-00886-7.
- “PropMolFlow: property-guided molecule generation with geometry-complete flow matching,” *Nature Computational Science* (2025), DOI: 10.1038/s43588-025-00946-y.
- Huang et al., “Learning Joint 2D & 3D Diffusion Models for Complete Molecule Generation,” arXiv:2305.12347.
- Ruiz-Botella et al., “A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules,” arXiv:2505.16365.
- Cremer et al., “FLOWR: flow matching for structure-aware de novo, interaction- and fragment-based ligand generation,” *Nature Computational Science* 6, 565–574 (2026), DOI: 10.1038/s43588-026-00998-8.
