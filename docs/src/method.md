# Method

Spectral (TKM / NUFFT) or point-charge evaluation of the screened samples ``a\rho`` sees the jumps of ``a\rho``. It
converges only like ``O(h)`` to ``O(h^2)``, which measured as ``10^{-2}`` to ``10^{-1}`` relative errors.
CutPotentials.jl instead keeps every volume computation on the smooth, unscreened ``\rho``, and moves the jumps into
surface integrals of known smooth data.

The target quantity is

```math
U(y) = \int G(y, x)\, a(x)\, \rho(x)\, dx, \qquad G(y, x) = \frac{1}{4\pi |y - x|},
```

with ``a = a_j`` on the regions ``\Omega_j`` and ``a = a_0`` outside, together with ``\nabla U`` and ``\nabla\nabla U``.
Write ``N[f] = \int G f\, dx`` for the Newton potential.

## Smooth fields

The smooth fields ``\varphi_0 = N[\rho]``, ``\psi = \nabla \varphi_0`` and ``H = \nabla \psi`` come from
[`SmoothField`](@ref):

- **Inside the padded source box:** the truncated-kernel method. A type-1 NUFFT (FINUFFT) of the samples is
  multiplied by the truncated Laplace kernel ``2\,(\sin(Lk/2)/k)^2`` and placed on one real scalar fine grid.
  ``\varphi_0``, ``\psi`` and ``H`` are then gathered from that grid with the window and its first two derivatives,
  ``w``, ``w'`` and ``w''``. The window is an analytic ES window tapered by ``(1 - z^2)^3``; see
  [Derivatives through the NUFFT](nufft.md) for the window and its width.
- **Elsewhere:** point charges, summed by SCTL's `ParticleFMM` (`Laplace3D_FxU`, `_FxdU`, `_Fxd2U`).

## Identities

For a region ``\Omega`` with indicator ``\chi``, boundary ``\Gamma`` and outward normal ``n``, write

```math
S[q](y) = \int_\Gamma G(y, x)\, q(x)\, dS_x, \qquad D[q](y) = \int_\Gamma \partial_{n_x} G(y, x)\, q(x)\, dS_x .
```

Green's third identity, with ``\Delta \varphi_0 = -\rho``, gives:

```math
\begin{aligned}
N[\chi\rho] &= \chi\, \varphi_0 + D[\varphi_0] - S[\partial_n \varphi_0], \\
\nabla N[\chi\rho] &= \chi\, \psi + D[\psi] - S\big[H n - \operatorname{tr}(H)\, n\big], \\
\nabla \partial_i N[\chi\rho] &= \chi\, H e_i + \nabla\times S[n \times H e_i]
  - \nabla S\big[(H n)_i - \operatorname{tr}(H)\, n_i\big].
\end{aligned}
```

**How the Hessian identity is derived:**
1. Differentiate the gradient identity.
2. The term ``\nabla D[\psi_i]`` is hypersingular. Replace it by Maue's identity,
   ``\nabla D[f] = \nabla\times S[n \times \nabla_\Gamma f]``, which holds on closed surfaces. Here
   ``n \times \nabla_\Gamma \psi_i = n \times H e_i`` is known pointwise.
3. The Hessian then needs only single-layer gradients of densities built from ``H``. It needs no third derivatives,
   no ``\nabla\rho`` and no differentiation on panels. These integrals are near-singular off the surface.

**Checks.**
- The jumps reproduce ``[\nabla\nabla N[\chi\rho]] = -\rho\, n n^\mathsf{T}``.
- The identity is verified numerically to ``10^{-14}`` with exact fields (tests).

**Merging the regions.** Write

```math
U = a_0\, \varphi_0 + \sum_j (a_j - a_0)\, N[\chi_j \rho]
```

and store each interface ``m`` once:
- its normal ``n_m`` points from side A to side B;
- its coefficient is ``\delta a_m = a_A - a_B``.

Shared faces then combine, and faces with ``\delta a = 0`` drop out. The free term
``\sum_j (a_j - a_0)\, \chi_j(y)`` is written as ``-W(y)``, with ``W = \sum_m \delta a_m D_m[1]`` computed by the
same quadrature. The Hessian line below is symmetrized:

```math
\begin{aligned}
U &= (a_0 - W)\, \varphi_0 + \sum_m \delta a_m \big(D_m[\varphi_0] - S_m[\psi \cdot n]\big), \\
\nabla U &= (a_0 - W)\, \psi + \sum_m \delta a_m \big(D_m[\psi] - S_m[H n - \operatorname{tr}(H)\, n]\big), \\
\partial_i \nabla U &= (a_0 - W)\, H e_i + \sum_m \delta a_m \big(\nabla\times S_m[n \times H e_i]
  - \nabla S_m[(H n)_i - \operatorname{tr}(H)\, n_i]\big).
\end{aligned}
```

## On-surface targets

``U`` and ``\nabla U`` equal ``a_0 \varphi_0 + \sum \delta a\, D[f - f(y)] - \dots``. Their double layers have
densities that vanish at the target, so they are continuous through every interface, edge and junction.

- **No snapping needed.** A target exactly on a surface, or ``10^{-16}`` off it, gets the same value. The free term
  is automatically ``(a_A + a_B)/2`` on a face and ``\sum_j \alpha_j a_j`` at a junction, with ``\alpha_j`` the
  solid-angle fractions.
- **Rounding.** This replaces the face-plane snapping that the research code needed. Box face coordinates are still
  merged within rounding, so that neighbouring boxes share exact rectangles.

The free term ``a_0 - W`` is returned as `coef`, a diagnostic.

The Hessian jumps by ``-\delta a\, \rho\, n n^\mathsf{T}`` across an interface.

- **Flagged targets.** For a target flagged with a `side` vector ``\nu``, the package returns the one-sided limit on
  the side ``\nu`` points to. It is extrapolated from ``H_k = \nabla\nabla U(y + k\delta\hat\nu)``,
  ``k = 1, \dots, 4``:

  ```math
  \nabla\nabla U(y^\pm) \approx 4H_1 - 6H_2 + 4H_3 - H_4,
  ```

  with ``\delta`` = `hess_offset`, default ``10^{-5} L_0``.
- **Unflagged targets** are evaluated where they are. A target exactly in the plane of a flat face gets the
  two-sided average.

## Discretization

- **Surface elements.** The surface integrals use their own SCTL Gauss–Legendre elements (`QuadElemList`, order 12
  by default). Each patch is split until its size is at most ``\kappa (d + L_0)``, where ``d`` is the distance to the
  density's support (``\kappa = 0.5``).
- **Nodal data.** ``\psi`` and ``H`` are evaluated pointwise at the nodes.
- **Far field.** The node rule is applied by `ParticleFMM`: `DxU` for the dipoles, `FxU` for the charges and `FxdU`
  for the single-layer gradients.
- **Near and singular pairs.** SCTL's `near_correction` handles them, for the kernels `Laplace3D_DxU`, `_FxU` and
  `_FxdU`. It resolves targets at any distance, down to ``10^{-12}``, as one-sided limits, to about ``10^{-14}``.
