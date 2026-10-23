# Accuracy, cost and limitations

## Measured accuracy

From `test/runtests.jl`.

- **Errors.** Each entry is the max error over the targets divided by the max ``|\text{reference}|``.
- **Density.** ``\rho`` is a Gaussian (``\sigma = 0.5``), sampled on a skewed, anisotropic lattice (spacing about ``0.1``).
- **Settings.** The smooth field uses tol 1e-12. The surface elements use order 12 and κ = 0.5.
- **Floor.** The skewed lattice cuts the Gaussian at about ``7\sigma`` in one corner, which limits the smooth field
  to about ``10^{-11}``. With the tail inside the box (``\pm 8\sigma``) it reaches ``10^{-13}``, as in the
  [examples](examples.md).
- **Two field sources.** "TKM" is the full pipeline. "Exact" feeds the exact ``\psi`` and ``H``, to isolate the surface part.

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
- **Laplacian.** ``\operatorname{tr} H + a\rho < 10^{-9} \max|H|`` everywhere off the surfaces, and one-sided on them
  (with the ``a`` of that side).
  The Hessian components are never overwritten using this identity.
- **coef.** It equals ``a`` off the surfaces to ``10^{-10}``, and ``(a_A + a_B)/2`` on flat faces.
- **Equal coefficients.** With all ``a = a_0``, the result is exactly ``a_0`` times the smooth field.
- **Merging.** The merged, the union-box and the region-by-region (unmerged) descriptions of two equal boxes agree
  to 1e-10.

**Where the remaining Hessian errors come from:**
- **The 2e-10 off-surface (exact fields)** comes from the reference. At 1e-8 from a face it violates its own trace
  identity by the same amount, while the computed Hessian satisfies it to 1e-16.
- **The 4.5e-10 on faces** is a target 1e-3 from an edge. Its extrapolation error is ``(\delta/\ell)^4`` with
  ``\delta = 5 \times 10^{-6}`` and ``\ell = 10^{-3}``. At 1e-2 from the edge it is 7e-14.

## Cost

Measured on ccmlin078, a shared and loaded workstation, with 16 threads. The setup:
- the Gaussian sampled at ``h = 0.1`` (358k samples);
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

At ``h = 0.05`` (2.8M samples), `SmoothField` takes 7.4 s and the other steps are unchanged.

`evaluate` runs one DxU FMM for ``W``, two for the potential and six for the gradient. The Hessian adds twelve FxdU
FMMs, so the layer cost grows linearly in the number of densities.

## Limitations

- **Hessian at edges and corners.** It is singular there (logarithmically, at box edges). The on-surface one-sided
  limit needs `hess_offset` to be well below the distance ``\ell`` to the nearest edge, with error about ``(\delta/\ell)^4``. Off the
  surface, targets 1e-6 from an edge are accurate (2e-11).
- **On-surface Hessian targets must be flagged** with `side`. For curved surfaces, a point of the exact surface lies
  within rounding of the discretized one, on either side. ``U`` and ``\nabla U`` do not care; the Hessian needs the flag, and
  `hess_offset` must exceed the geometric error of the elements.
- **Discretized geometry.** The regions are those bounded by the discretized surfaces: SCTL's polynomial
  interpolants of the patches. Curved surfaces therefore carry a geometric error, set by `order` and the element
  size.
- **Closedness is not checked.** The interfaces must make up closed region boundaries, with consistent orientation
  and ``\delta a``. Maue's identity, and the cancellation of free terms into ``W``, rely on it.
- **Per-target-set setup.** The near corrections depend on the targets. Many targets close to large surfaces cost
  memory: about 1 GB per 1e7 near-pair nonzeros, three times that for the Hessian kernel.
- **The density must be resolved by its samples** and decayed at the edge of the sample box, which is the TKM
  assumption. The error is about ``|\hat\rho(\pi/h)|``, ``e^{-\sigma^2\pi^2/2h^2}`` for a Gaussian, plus the
  mass the box leaves out. That is ``10^{-10}`` at ``h = \sigma/2`` and ``10^{-3}`` at ``h = \sigma``. The
  surface corrections remove only the error that screening adds, not under-resolution. See
  [Examples](examples.md#When-the-smooth-field-evaluation-fails). Surface nodes and targets outside the padded box
  use point charges, which are accurate there.
- **Grading parameters.** The default `L0` and `support` are heuristics; give the density's width and support when
  known. Elements whose patches do not resolve the geometry need `max_size` or more patches (`nsub`).
- **Platform.** The package is MPI-free and `Float64` only. `ParticleFMM` sums directly below 40,000 targets
  (SCTL's choice).
