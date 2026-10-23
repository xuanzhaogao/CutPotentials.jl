# CutPotentials.jl

CutPotentials.jl evaluates the Newton potential of a smooth density ``\rho`` screened by a piecewise-constant coefficient,

```math
U(y) = \int G(y, x)\, a(x)\, \rho(x)\, dx, \qquad G = \frac{1}{4\pi |y - x|}, \qquad a = a_j \text{ on } \Omega_j,\ a_0 \text{ outside},
```

together with ``\nabla U`` and the Hessian ``\nabla\nabla U``.

- **Density.** ``\rho`` is given by samples on a lattice, which may be anisotropic or skewed.
- **Regions.** The ``\Omega_j`` are bounded by closed surfaces of any shape: unions of boxes, spheres, ellipsoids,
  or any parametrized patches.
- **Targets.** A target can be anywhere: in the volume, near a surface, or exactly on one, including on edges and
  junctions for ``U`` and ``\nabla U``.

All volume work is done on the smooth, unscreened ``\rho`` (truncated-kernel method with
[FINUFFT](https://github.com/flatironinstitute/finufft), point charges with
[SCTL](https://github.com/xuanzhaogao/SCTL.jl)'s FMM). The jumps of ``a\rho`` become surface integrals of known
smooth data, discretized with SCTL's Gauss–Legendre elements and singular/near-singular corrections. See
[Method](method.md).

## Installation

CutPotentials depends on [SCTL.jl](https://github.com/xuanzhaogao/SCTL.jl), which is not registered and compiles a
C++ shim when built. Develop it first, then add CutPotentials:

```julia
using Pkg
Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl")
Pkg.build("SCTL")      # set ENV["SCTL_JL_MARCH"] = "-march=x86-64-v3" for a portable build
Pkg.add(url = "https://github.com/xuanzhaogao/CutPotentials.jl")
```

## Quick start

```julia
using CutPotentials
src   = VolumeSource(x -> exp(-sum(abs2, x) / 0.5), (-2.0, -2.0, -2.0), 0.1 .* [1 0 0; 0 1 0; 0 0 1], (41, 41, 41))
itf   = box_interfaces([((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))], [2.0], 1.0)   # a = 2 in the box, 1 outside
r     = cut_potential(src, itf, randn(3, 100); a0 = 1.0, tol = 1e-10, hess = true)
r.pot, r.grad, r.hess      # Nt, 3×Nt, 6×Nt (xx yy zz xy xz yz)
```

See [Usage](usage.md) for the step-by-step plan API and the accuracy knobs.
