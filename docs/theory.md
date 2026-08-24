# Theory and model contract

ECloudFlow generates a complete ligand graph and a binding pose conditioned on
a pocket. The state contains centered coordinates `x` in angstroms, atom and
charge categorical logits, sparse unordered halfedges, and packed electron
latents. Pocket and ligand coordinates use the same `CoordinateFrame`; the
inverse transform is applied only when writing global artifacts.

## Electron fields

For atom centers `r_i`, a local electron density is expanded in radial basis
functions and real spherical harmonics:

```text
rho(r) = sum_i sum_n sum_l sum_m c[i,n,l,m] R_n(||r-r_i||) Y_lm((r-r_i)/||r-r_i||).
```

The coefficients are grouped into irreducible representations up to `lmax`.
The scalar `l=0` channels carry invariant density information; `l>0` channels
transform with the Wigner-D matrices. Numerical integration on a fixed grid
records electron-count and dipole checks. Missing QM fields are represented as
an unavailable provenance record, never as a zero target.

## SE(3) behavior

For a proper rotation `R` and translation `t`, coordinates transform as
`x' = x R^T + t`. Scalar logits are invariant, vector features rotate by `R`,
and spherical coefficients transform by the matching representation. The
model is equivariant to translations and proper rotations; reflection
invariance is not imposed. A centered pocket frame removes arbitrary global
translation while retaining a reversible `to_global` transform.

## Stochastic interpolant

Training couples a data state `z_1` to a simple prior `z_0` with
`z_t = (1-t) z_0 + t z_1 + sigma(t) eps`. The velocity target is
`u_t = d z_t / dt`, and the network minimizes weighted mean squared error
`L_flow = E[w(t)||v_theta(z_t,t,c)-u_t||^2]`. A terminal score head uses the
denoising target `s_t = -eps/sigma(t)` and is applied only in the final
corrector window. Coordinates and electron latents share this continuous path.

## Categorical paths and graph cycle

Atoms, formal charges, bond classes, and atom count use simplex interpolation
between a prior distribution and one-hot endpoints. Cross-entropy at sampled
`t` trains the categorical heads. The sparse halfedge convention stores each
unordered pair once (`i < j`); decoding materializes a symmetric matrix only at
the end. Pocket encoder outputs condition the ligand backbone, while decoded
ligand features can reconstruct a ligand field. The cycle loss compares this
field with the encoded latent and discourages pocket-conditioning shortcuts.

## Constraints and guidance

At every solver and corrector step, fragment masks overwrite fixed atom type,
charge, internal bond, and coordinate values. Valence projection masks classes
that would exceed the configured table, excludes self-edges, and preserves
halfedge symmetry. A clash and connectivity penalty supplies differentiable
guidance; an optional property head adds a calibrated conditional loss.

The final decoder solves a small CP-SAT feasibility problem over bond classes,
valence, connectivity, fragment edges, and allowed element vocabulary. RDKit
sanitization is a separate check. The generated raw pose is written before any
MMFF/UFF relaxation; the relaxed pose is a second artifact and is never used to
rewrite the raw structure.

## Solver accounting

Euler uses one network evaluation per step. Heun evaluates the endpoint twice;
the terminal corrector adds its declared number of score evaluations. Profiles
therefore report nominal NFE contracts of 20 (`fast`), 82 (`balanced`), and 208
(`quality`). Actual counts, wall time, and rejection reasons are persisted so
quality claims can be audited. `num_molecules` counts valid unique canonical
isomeric SMILES; attempts stop at `5 * num_molecules` unless overridden.

## Scientific limits

The equations describe an executable hypothesis, not a guarantee of affinity.
Electron labels depend on QM settings and field grids, docking scores are not
binding free energies, and a toy smoke backend has no physical interpretation.
Comparisons require matched splits, seeds, postprocessing, docking protocol,
and tool versions. Ablations for fields, corrector, constraints, fragments,
and NFE are required before attributing an observed improvement.
