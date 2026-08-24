# Theory and model contract

ECloudFlow models the conditional distribution of a complete ligand graph and
its binding pose,

```text
p_theta(G_L, X_L, Z_L | G_P, X_P, F_P, C),
```

where `G_P, X_P` are the pocket graph and coordinates, `F_P` is the pocket
electron/interaction field, and `C` contains fragment, property, and task
conditions. The generated variables are the ligand graph `G_L`, pose `X_L`,
and ligand electron latent `Z_L`. The model is an executable research
hypothesis: binding quality is established by held-out evaluation, docking
under a declared protocol, and ultimately experiment, not by this factorization
alone.

## Canonical state and coordinate frame

One batched ligand state is

```text
S_t = (X_t, A_t, Q_t, E_t, B_t, Z_t, b_N, b_E).
```

`X_t in R^(N x 3)` contains coordinates in angstroms; `A_t`, `Q_t`, and `B_t`
are atom, charge, and bond-class values; `Z_t` is the packed equivariant
electron latent. `E_t in N^(2 x E)` stores each unordered ligand halfedge once
with `E_t[0,e] < E_t[1,e]`. `b_N` and `b_E` map nodes and halfedges to complexes.
Dense `N x N` bond tensors are prohibited inside the learned trajectory and
are constructed only for final graph decoding or presentation.

For pocket center `mu_P`, local and global coordinates are

```text
x_local  = x_global - mu_P,
x_global = x_local  + mu_P.
```

The pocket, optional positioned fragment, and generated ligand use the same
frame. Generation therefore produces a pose in the pocket frame, and the
single inverse transform writes coordinates back into the supplied PDB frame.

## Electron-cloud representation

Around atom center `r_i`, the electron density uses compact radial functions
and real spherical harmonics:

```text
rho(r) = sum_i sum_n sum_l=0^L sum_m=-l^l
         c[i,n,l,m] R_n(||r-r_i||) Y_lm(unit(r-r_i)).
```

The radial basis controls resolution and cutoff. Coefficients with angular
order `l` form irreducible `SO(3)` blocks of width `2l+1`. The tokenizer maps
field coefficients to a packed latent,

```text
Z_i = [Z_i^(0), Z_i^(1), ..., Z_i^(L)],
```

while the decoder reconstructs density, spatial gradient, electron count, and
dipole. Useful consistency terms are

```text
N_e_hat = integral rho_hat(r) dr,
mu_hat  = integral r rho_hat(r) dr,
L_rho   = mean_q |rho_hat(q)-rho(q)|^2,
L_grad  = mean_q ||grad rho_hat(q)-grad rho(q)||^2.
```

QM labels are active only where `qm_mask` is true. A failed xTB calculation is
recorded as unavailable provenance; replacing it with zeros would teach a
false physical target and is forbidden.

## SE(3) equivariance

For a proper rotation `R` (`det R = +1`) and translation `t`, coordinates obey

```text
X' = X R^T + 1 t^T.
```

Scalar channels and categorical logits remain invariant. Cartesian vectors
rotate by `R`; an order-`l` electron block transforms by the Wigner matrix
`D^(l)(R)`. The required model relation is

```text
f_theta(R X + t, D(R) Z, c) = (R v_X, D(R) v_Z, invariant logits).
```

All geometric messages depend on relative displacements, distances, invariant
contractions, and equivariant tensor products. Translation cancels from
relative vectors. The architecture retains parity-sensitive channels and does
not impose reflection invariance, allowing it to distinguish chiral geometry.

## Architecture

```mermaid
flowchart LR
    P[Pocket atoms and coordinates] --> PE[SE(3) pocket encoder]
    PF[Pocket electron field] --> FT[Field tokenizer]
    FT --> PE
    PE --> C[Cached pocket context]
    L[Noisy ligand graph, pose, and electron latent] --> JB[Joint equivariant backbone]
    C --> JB
    F[Fragment and property conditions] --> JB
    T[Time embedding] --> JB
    JB --> V[Coordinate and electron velocity]
    JB --> S[Terminal diffusion score]
    JB --> D[Atom, charge, bond, and count heads]
    JB --> A[Affinity and interaction auxiliaries]
    V --> SOL[Flow solver]
    S --> SOL
    D --> SOL
    SOL --> HC[Fragment clamp and chemistry projection]
    HC --> GD[Exact graph decoder]
    GD --> OUT[Raw pocket-frame ligand pose]
```

The pocket encoder is evaluated once per pocket and cached across integration
steps. Time FiLM, ligand-pocket cross messages, field moments, fragment task
embeddings, and named property targets condition each ligand block. Output
heads jointly predict coordinate/electron flow, terminal score, categorical
endpoints, atom count, affinity mean/log-variance, and interaction logits.

## Continuous stochastic interpolant

Let `z_1` be a data sample and `z_0` a cavity-aware prior. A general bridge is

```text
z_t = alpha(t) z_0 + beta(t) z_1 + gamma(t) epsilon,
epsilon ~ Normal(0, I),
alpha(0)=1, beta(0)=0, alpha(1)=0, beta(1)=1, gamma(0)=gamma(1)=0.
```

For the linear mean path, `alpha=1-t` and `beta=t`. The exact conditional
velocity target is

```text
u_t = alpha_dot(t) z_0 + beta_dot(t) z_1 + gamma_dot(t) epsilon,
```

and the denoising score target is

```text
s_t = -epsilon / gamma(t).
```

Times whose `gamma(t)` is numerically indistinguishable from zero are excluded
from score supervision rather than clamping the denominator and changing the
target. Coordinates and electron latents share this formulation. The primary
continuous objective is

```text
L_flow = E[w_v(t) ||v_theta(z_t,t,c)-u_t||^2],
L_score = E[w_s(t) ||s_theta(z_t,t,c)-s_t||^2].
```

Flow matching supplies a deterministic ODE with few evaluations; the score
head is used only in a terminal correction window to recover diffusion-style
local refinement without paying that cost throughout the trajectory.

## Categorical paths

Atoms, formal charges, bonds, and count evolve on probability simplices. For a
prior `pi_0` and one-hot endpoint `y`, the affine probability path is

```text
p_t = (1-t) pi_0 + t one_hot(y),
dp_t/dt = one_hot(y) - pi_0.
```

The model predicts categorical endpoint logits or an explicitly adapted
probability drift. Training uses masked cross-entropy/KL terms. Fixed fragment
categories bypass sampling and remain exact endpoints. Probability tensors are
renormalized after each numerical update, and invalid BF16 simplex mass fails
validation instead of being silently accepted.

## Joint objective

The total objective contains six named components:

```text
L = lambda_v   L_flow
  + lambda_s   L_score
  + lambda_cat L_discrete
  + lambda_rho L_ecloud
  + lambda_chem L_chem
  + lambda_int L_interaction.
```

Each component is divided by a detached distributed running RMS before its
configured weight and warm-up are applied. This prevents one unit system from
dominating while preserving within-component gradient direction. The field
component includes density, gradient, electron-count, dipole, and graph-field
cycle reconstruction. The interaction component includes contact/fingerprint
supervision and, where affinity labels exist, heteroscedastic regression,

```text
L_aff = 0.5 exp(-ell) (y-mu)^2 + 0.5 ell,
```

where `mu` and `ell` are the predicted affinity mean and log variance.

## Chemical and geometric regularization

For expected bond order `o_ij` and allowed valence `V_i`, a differentiable
overflow term is

```text
L_valence = sum_i relu(sum_j o_ij - V_i)^2.
```

Additional active terms cover element/bond-conditioned length priors,
short-range ligand-ligand and protein-ligand repulsion, ring strain, and a
connectivity surrogate. These are training/guidance energies, not a substitute
for exact final validity. During sampling, the chemical projector masks
impossible bond classes and saturated attachment sites after every numerical
substep.

## Fragment-conditioned generation

A fragment condition supplies immutable reference values and boolean masks.
For every observable solver state `S_t`, clamping implements

```text
S_t[field][fixed_mask] = S_ref[field][fixed_mask]
```

for atom identity, formal charge, coordinates, and fixed internal bonds. The
assignment is performed after Euler candidates, both Heun stages, score
corrector steps, and chemistry projections. The five task modes differ only in
which non-fixed atoms and attachment halfedges may change:

| Mode | Fixed input | Editable operation |
| --- | --- | --- |
| `de_novo` | none | create a complete ligand |
| `grow` | one retained fragment | add atoms and attachment bonds |
| `link` | two or more retained components | create a linker |
| `replace` | retained scaffold context | replace a selected region |
| `merge` | overlapping/adjacent fragments | integrate into one graph |

## Conditional guidance

Classifier-free guidance evaluates conditional and null predictions and forms

```text
v_guided = v_null + w (v_cond - v_null).
```

Pocket geometry, pocket field, fragment/task identity, interaction targets,
and property name-value pairs are removed from the null condition. Guidance is
applied only to editable values; the exact fragment clamp follows it, so no
guidance scale can move a fixed atom or change a fixed bond.

## Numerical solver and NFE

Euler updates `z_{k+1}=z_k+h v_theta(z_k,t_k,c)`. Heun uses a predictor and a
second evaluation,

```text
z_pred = z_k + h v_k,
z_{k+1} = z_k + h (v_k + v_pred)/2.
```

Each candidate passes through the canonical hook order: editable chemistry and
cavity projection, exact fragment clamp, then trajectory recording. Terminal
Langevin correction uses caller-owned random generators and score steps only in
the declared final window. Nominal NFE is therefore 20 for `fast`, 82 for
`balanced` (`2*40+2`), and 208 for `quality` (`2*100+8`). Actual calls, wall
time, memory, and rejection causes are persisted.

## Exact graph decoding

After the continuous trajectory, atom/charge classes are selected and bond
classes are decoded with CP-SAT where available. Binary variables `b[e,k]`
select one class per halfedge subject to

```text
sum_k b[e,k] = 1,
sum_(e incident to i) sum_k order(k) b[e,k] <= V_i,
b[e,k_fixed] = 1 for fixed fragment bonds.
```

Connectivity constraints or a bounded fallback join all required components.
The decoder then builds an RDKit molecule, assigns the generated conformer, and
runs sanitization. Non-integral reconstructed bond orders, unsupported atoms,
valence overflow, disconnected graphs, sanitization failure, and conformer
failure are rejected with typed reasons. Relaxation writes a second molecule;
it never overwrites the raw generated pose.

## Complexity and sampling contract

Pocket encoding is cached once. Sparse ligand, pocket, and cross-radius graphs
keep message passing approximately linear in retained neighbors rather than
quadratic in all atom pairs. Exact decoding is bounded and occurs once per
candidate. `num_molecules` is the target count of valid unique canonical
isomeric SMILES. Attempts stop at `max_attempts`, defaulting to five times the
target, so sampling cannot loop indefinitely.

## Scientific limits and required ablations

Electron labels depend on QM method, charge, multiplicity, basis/grid choices,
and coordinate alignment. Docking scores are not binding free energies, and
the deterministic smoke scorer has no physical interpretation. Comparisons
must match data splits, seeds, requested valid count, postprocessing, docking
box/protocol, NFE, and optional-tool availability. At minimum, report ablations
for pocket/ligand electron latents, score correction, chemistry projection,
exact decoding, fragment clamping, conditioning dropout/guidance, and sampling
profile before attributing an observed improvement to a component.
