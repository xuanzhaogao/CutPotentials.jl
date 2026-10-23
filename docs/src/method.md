# Method

Spectral (TKM / NUFFT) or point-charge evaluation of the screened samples aρ sees the jumps of aρ. It converges only
like O(h) to O(h²), which measured as 1e-2 to 1e-1 relative errors. CutPotentials.jl instead keeps every volume
computation on the smooth, unscreened ρ, and moves the jumps into surface integrals of known smooth data.

## Smooth fields

φ0 = N[ρ], ψ = ∇φ0 and H = ∇ψ come from `SmoothField`:

- **Inside the padded source box:** the truncated-kernel method. A type-1 NUFFT (FINUFFT) of the samples is
  multiplied by the truncated Laplace kernel 2(sin(Lk/2)/k)² and placed on one real scalar fine grid. φ0, ψ and H
  are then gathered from that grid with the window and its first two derivatives, w, w′ and w″. The window is an
  analytic ES window tapered by (1 − z²)³.
- **Elsewhere:** point charges, summed by SCTL's `ParticleFMM` (`Laplace3D_FxU`, `_FxdU`, `_Fxd2U`).

## Identities

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

## On-surface targets

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

## Discretization

- **Surface elements.** The surface integrals use their own SCTL Gauss–Legendre elements (`QuadElemList`, order 12
  by default). Each patch is split until its size is at most κ(d + L0), where d is the distance to the density's
  support (κ = 0.5).
- **Nodal data.** ψ and H are evaluated pointwise at the nodes.
- **Far field.** The node rule is applied by `ParticleFMM`: `DxU` for the dipoles, `FxU` for the charges and `FxdU`
  for the single-layer gradients.
- **Near and singular pairs.** SCTL's `near_correction` handles them, for the kernels `Laplace3D_DxU`, `_FxU` and
  `_FxdU`. It resolves targets at any distance, down to 1e-12, as one-sided limits, to about 1e-14.
