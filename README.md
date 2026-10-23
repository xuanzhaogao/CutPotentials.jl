# CutPotentials.jl

CutPotentials.jl evaluates the Newton potential of a smooth density ρ screened by a piecewise-constant coefficient:

    U(y) = ∫ G(y, x) a(x) ρ(x) dx,    G = 1/(4π|y − x|),    a = a_j on regions Ω_j, a0 outside.

It also returns ∇U and the Hessian ∇∇U.

- **Density.** ρ is given by samples on a lattice, which may be anisotropic or skewed.
- **Regions.** The Ω_j are bounded by closed surfaces of any shape: unions of boxes, spheres, ellipsoids, or any
  parametrized patches.
- **Targets.** A target can be anywhere: in the volume, near a surface, or exactly on one, including on edges and
  junctions for U and ∇U.

A typical use is a = 1/ε, the screened source of a dielectric problem.

## Method

Spectral (TKM / NUFFT) or point-charge evaluation of the screened samples aρ sees the jumps of aρ. It converges only
like O(h) to O(h²), which measured as 1e-2 to 1e-1 relative errors. CutPotentials.jl instead keeps every volume
computation on the smooth, unscreened ρ, and moves the jumps into surface integrals of known smooth data.

### Smooth fields

φ0 = N[ρ], ψ = ∇φ0 and H = ∇ψ come from `SmoothField`:

- **Inside the padded source box:** the truncated-kernel method. A type-1 NUFFT (FINUFFT) of the samples is
  multiplied by the truncated Laplace kernel 2(sin(Lk/2)/k)² and placed on one real scalar fine grid. φ0, ψ and H
  are then gathered from that grid with the window and its first two derivatives, w, w′ and w″. The window is an
  analytic ES window tapered by (1 − z²)³.
- **Elsewhere:** point charges, summed by SCTL's `ParticleFMM` (`Laplace3D_FxU`, `_FxdU`, `_Fxd2U`).

### Identities

For a region Ω with outward normal n, write S[q] = ∫_Γ G q dS and D[q] = ∫_Γ ∂_{n_x}G q dS.
Green's third identity gives:

    N[χ ρ]   = χ φ0 + D[φ0] − S[∂_n φ0]
    ∇N[χ ρ]  = χ ψ  + D[ψ]  − S[H n − tr(H) n]                    (Δφ0 = −ρ)
    ∇∂_i N[χ ρ] = χ H e_i + ∇×S[n × H e_i] − ∇S[(H n)_i − tr(H) n_i]

**How the Hessian identity is derived:**
1. Differentiate the gradient identity.
2. The term ∇D[ψ_i] is hypersingular. Replace it by Maue's identity, ∇D[f] = ∇×S[n × ∇_Γ f], which holds on closed
   surfaces. Here n × ∇_Γ ψ_i = n × H e_i is known pointwise.
3. The Hessian then needs only single-layer gradients of densities built from H. It needs no third derivatives, no
   ∇ρ and no differentiation on panels. These integrals are near-singular off the surface.

**Checks.**
- The jumps reproduce [∇∇N[χρ]] = −ρ n nᵀ.
- The identity is verified numerically to 1e-14 with exact fields (tests).

**Merging the regions.** Write U = a0 φ0 + Σ_j (a_j − a0) N[χ_j ρ] and store each interface once:
- its normal n_m points from side A to side B;
- its coefficient is δa_m = a_A − a_B.

Shared faces then combine, and faces with δa = 0 drop out. The free term Σ_j (a_j − a0) χ_j(y) is written as
−W(y), with W = Σ_m δa_m D_m[1] computed by the same quadrature:

    U   = (a0 − W) φ0 + Σ_m δa_m (D_m[φ0] − S_m[ψ·n])
    ∇U  = (a0 − W) ψ  + Σ_m δa_m (D_m[ψ] − S_m[Hn − tr(H) n])
    ∇∇U = (a0 − W) H  + Σ_m δa_m (∇×S_m[n × H e_i] − ∇S_m[(Hn)_i − tr(H) n_i])_i   (symmetrized)

### On-surface targets

U and ∇U equal a0 φ0 + Σ δa D[f − f(y)] − …. Their double layers have densities that vanish at the target, so they
are continuous through every interface, edge and junction.

- **No snapping needed.** A target exactly on a surface, or 1e-16 off it, gets the same value. The free term is
  automatically (a_A + a_B)/2 on a face and Σ α_j a_j at a junction.
- **Rounding.** This replaces the face-plane snapping that the research code needed. Box face coordinates are still
  merged within rounding, so that neighbouring boxes share exact rectangles.

`coef = a0 − W` is returned as a diagnostic.

The Hessian jumps by −δa ρ n nᵀ across an interface.

- **Flagged targets.** For a target flagged with a `side` vector ν, the package returns the one-sided limit on the
  side ν points to. It is extrapolated from y + kδν̂ for k = 1..4 (H ≈ 4H₁ − 6H₂ + 4H₃ − H₄, with δ =
  `hess_offset`, default 1e-5·L0).
- **Unflagged targets** are evaluated where they are. A target exactly in the plane of a flat face gets the
  two-sided average.

### Discretization

- **Surface elements.** The surface integrals use their own SCTL Gauss–Legendre elements (`QuadElemList`, order 12
  by default). Each patch is split until its size is at most κ(d + L0), where d is the distance to the density's
  support (κ = 0.5).
- **Nodal data.** ψ and H are evaluated pointwise at the nodes.
- **Far field.** The node rule is applied by `ParticleFMM`: `DxU` for the dipoles, `FxU` for the charges and `FxdU`
  for the single-layer gradients.
- **Near and singular pairs.** SCTL's `near_correction` handles them, for the kernels `Laplace3D_DxU`, `_FxU` and
  `_FxdU`. It resolves targets at any distance, down to 1e-12, as one-sided limits, to about 1e-14.

## Usage

```julia
using CutPotentials
src   = VolumeSource(x -> rho(x), origin, basis, (n1, n2, n3))     # or VolumeSource(values, origin, basis)
field = SmoothField(src; tol = 1e-12)                               # once per density
itf   = vcat(box_interfaces([(lo1, hi1), (lo2, hi2)], [a1, a2], a0),  # unique rectangles, δa = a_A − a_B
             [sphere_interface(c, R, a_in - a_out)])                  # also ellipsoid_interface, Interface(patches, δa)
sq    = SurfaceQuadrature(itf; support = density_support(src), L0 = σ)   # graded elements, once per geometry
plan  = CutPlan(sq, targets; a0, hess = true, side = side_normals)      # near corrections, once per target set
r     = evaluate(plan, field)            # r.pot (Nt), r.grad (3×Nt), r.hess (6×Nt: xx yy zz xy xz yz), r.coef
```

**General surfaces.** `Interface(patches, δa)` takes any closed-surface pieces. Each patch is a function
`X(u, v)` on [0, 1]², with ∂uX × ∂vX pointing from side A to side B. Together, the interfaces must make up the
region boundaries, each piece weighted by its δa.

**Shortcut.** `cut_potential(src, itf, targets; a0, ...)` does all four steps at once.

**Accuracy knobs:**

| knob | what it controls |
|---|---|
| `SmoothField`: `tol`, `c_pad` | TKM accuracy and padding |
| `SurfaceQuadrature`: `order`, `κ`, `L0`, `max_size` | surface resolution |
| `CutPlan`: `tol` | near-quadrature accuracy |
| `CutPlan`: `fmm_digits` | layer FMM accuracy |
| `CutPlan`: `hess_offset` | on-surface Hessian extrapolation step |

## Installation

SCTL.jl is a local package. `Project.toml` points to it with `[sources] SCTL = {path = "../../sctl_jl/SCTL.jl"}`.
Adjust that path, or `Pkg.develop` SCTL.jl.

Run the tests with:

    julia --project=test test/runtests.jl

`Pkg.test` is broken on this machine. Set `CUTPOT_ONLY=<name>` to run a subset.

## Measured accuracy (test/runtests.jl)

- **Errors.** Each entry is the max error over the targets divided by the max |reference|.
- **Density.** ρ is a Gaussian (σ = 0.5), sampled on a skewed, anisotropic lattice (spacing about 0.1).
- **Settings.** The smooth field uses tol 1e-12. The surface elements use order 12 and κ = 0.5.
- **Two field sources.** "TKM" is the full pipeline. "Exact" feeds the exact ψ and H, to isolate the surface part.

| case (reference) | targets | pot | grad | hess |
|---|---|---|---|---|
| smooth field alone (exact Gaussian fields) | in and out of the box | 1.6e-11 | 1.6e-11 | 5.2e-12 |
| one box (erf/heat kernel), TKM | volume, 1e-3…1e-12 off faces, near the edge | 5.0e-11 | 1.0e-10 | 2.0e-10 |
| one box, TKM | on faces, one-sided H | 1.0e-11 | 5.0e-12 | 4.5e-10 |
| one box, TKM | on and 1e-4…1e-6 from the edge | 9.2e-12 | 4.5e-12 | — |
| one box, exact ψ, H | off / on / edge | 9e-14 / 3e-15 / 2e-15 | 3e-14 / 7e-15 / 1e-14 | 2.0e-10 / 4.5e-10 / — |
| four boxes along an edge (mixed a), TKM | off / on / edge | 1.1e-10 / 2.3e-11 / 2.2e-11 | 2.0e-10 / 9.6e-12 / 7.1e-12 | 2.3e-10 / 2.4e-11 / — |
| three boxes (shared faces, an edge, a corner), TKM | off / on / edge+corner | 8.3e-11 / 1.7e-11 / 1.5e-11 | 1.7e-10 / 7.3e-12 / 6.2e-12 | 2.9e-10 / 4.1e-10 / — |
| sphere (noncentral-χ heat kernel), TKM | off (≥1e-6) / 1e-9…1e-13 / on, one-sided H | 3.1e-11 / 2.3e-11 / 2.3e-11 | 3.1e-11 / 2.9e-11 / 2.9e-11 | 6.5e-12 / — / 1.9e-11 |
| sphere, exact ψ, H | same | 2.5e-14 / 8.3e-15 / 2.9e-15 | 3.6e-14 / 2.5e-14 / 1.0e-14 | 5.2e-12 / — / 2.3e-11 |
| rotated ellipsoid (adaptive cubature), TKM | inside and outside | 2.5e-11 | 8.7e-12 | 4.4e-12 |
| sphere inside a box (both interface types), TKM | off / on | 2.4e-11 / 2.4e-11 | 1.5e-11 / 1.6e-11 | 2.7e-11 / 1.5e-11 |
| equal coefficients (interface plus its reverse) | mixed | 6.0e-15 | 3.1e-15 | 3.3e-15 |

**Other checks:**
- **Laplacian.** tr H + aρ < 1e-9·max|H| everywhere off the surfaces, and one-sided on them (with a of that side).
  The Hessian components are never overwritten using this identity.
- **coef.** It equals a off the surfaces to 1e-10, and (a_A + a_B)/2 on flat faces.
- **Equal coefficients.** With all a = a0, the result is exactly a0 times the smooth field.
- **Merging.** The merged, the union-box and the region-by-region (unmerged) descriptions of two equal boxes agree
  to 1e-10.

**Where the remaining Hessian errors come from:**
- **The 2e-10 off-surface (exact fields)** comes from the reference. At 1e-8 from a face it violates its own trace
  identity by the same amount, while the computed Hessian satisfies it to 1e-16.
- **The 4.5e-10 on faces** is a target 1e-3 from an edge. Its extrapolation error is (δ/ℓ)⁴ with δ = 5e-6 and
  ℓ = 1e-3. At 1e-2 from the edge it is 7e-14.

## Cost

Measured on ccmlin078, a shared and loaded workstation, with 16 threads. The setup:
- the Gaussian sampled at h = 0.1 (358k samples);
- four boxes along an edge (20 rectangles, 118,656 surface nodes);
- 1e5 random targets in a cube of side 3 around the density;
- tol 1e-10.

| step | pot + grad + hess | pot + grad |
|---|---|---|
| `SmoothField` | 1.1 s | 1.1 s |
| `SurfaceQuadrature` | 0.04 s | 0.04 s |
| `CutPlan` (near corrections: 23M DxU/FxU and 70M FxdU nonzeros, FMM trees) | 8.8 s | 3.9 s |
| `evaluate` (second call; field values 0.3 s, layer FMMs and near matvecs the rest) | 9.9 s | 4.1 s |
| peak RSS | 14 GB | 8.9 GB |

At h = 0.05 (2.8M samples), `SmoothField` takes 7.4 s and the other steps are unchanged.

`evaluate` runs one DxU FMM for W, two for the potential and six for the gradient. The Hessian adds twelve FxdU
FMMs, so the layer cost grows linearly in the number of densities.

## Limitations

- **Hessian at edges and corners.** It is singular there (logarithmically, at box edges). The on-surface one-sided
  limit needs `hess_offset` to be well below the distance ℓ to the nearest edge, with error about (δ/ℓ)⁴. Off the
  surface, targets 1e-6 from an edge are accurate (2e-11).
- **On-surface Hessian targets must be flagged** with `side`. For curved surfaces, a point of the exact surface lies
  within rounding of the discretized one, on either side. U and ∇U do not care; the Hessian needs the flag, and
  `hess_offset` must exceed the geometric error of the elements.
- **Discretized geometry.** The regions are those bounded by the discretized surfaces: SCTL's polynomial
  interpolants of the patches. Curved surfaces therefore carry a geometric error, set by `order` and the element
  size.
- **Closedness is not checked.** The interfaces must make up closed region boundaries, with consistent orientation
  and δa. Maue's identity, and the cancellation of free terms into W, rely on it.
- **Per-target-set setup.** The near corrections depend on the targets. Many targets close to large surfaces cost
  memory: about 1 GB per 1e7 near-pair nonzeros, three times that for the Hessian kernel.
- **The density must be resolved by its samples** and decayed at the edge of the sample box, which is the TKM
  assumption. Surface nodes and targets outside the padded box use point charges, which are accurate there.
- **Grading parameters.** The default `L0` and `support` are heuristics; give the density's width and support when
  known. Elements whose patches do not resolve the geometry need `max_size` or more patches (`nsub`).
- **Platform.** The package is MPI-free and `Float64` only. `ParticleFMM` sums directly below 40,000 targets
  (SCTL's choice).
