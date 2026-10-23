# CutPotentials.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/CutPotentials.jl/dev/)
[![CI](https://github.com/xuanzhaogao/CutPotentials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/CutPotentials.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CutPotentials.jl computes the potential, gradient and Hessian of a smooth 3D density $\rho$ screened by a
piecewise-constant coefficient $a$:

$$
U(y) = \int \frac{a(x)\, \rho(x)}{4\pi |y - x|}\, dx, \qquad a = a_j \text{ in region } \Omega_j, \quad a = a_0 \text{ outside}.
$$

- **Density.** $\rho$ is given as samples on a lattice, which may be anisotropic or skewed.
- **Regions.** The regions can be any closed surfaces: unions of boxes, spheres, ellipsoids, or user-supplied
  parametric patches.
- **Targets.** A target can be anywhere: in the volume, near an interface, or exactly on one, including edges and
  junctions.
- **Accuracy.** Errors reach about 1e-13 relative to the largest value, when the samples resolve ρ. Evaluating
  $a\rho$ directly with FFTs or point charges reaches only 1e-2.

A typical use is $a = 1/\varepsilon$, the screened source of a dielectric problem.

![A Gaussian density straddling a sphere (a = 0.2 inside, 1 outside): a, the sphere's surface elements, refined toward the density; b, the potential U on a slice; c, the normal derivative ∂U/∂n at targets on the sphere](docs/src/assets/sphere.png)

More figures, including where the method stops working (an under-resolved density), are on the
[examples page](https://xuanzhaogao.github.io/CutPotentials.jl/dev/examples/).

## Installation

CutPotentials depends on [SCTL.jl](https://github.com/xuanzhaogao/SCTL.jl). SCTL.jl is not registered, and it
compiles a C++ shim when built. Develop it first, then add CutPotentials:

```julia
using Pkg
Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl")   # Pkg.add(url = ...) fails on its git submodules
Pkg.build("SCTL")        # ENV["SCTL_JL_MARCH"] = "-march=x86-64-v3" for a portable build
Pkg.add(url = "https://github.com/xuanzhaogao/CutPotentials.jl")
```

## Quick start

```julia
using CutPotentials

# Gaussian density sampled on a 41³ lattice with spacing 0.1
src = VolumeSource(x -> exp(-sum(abs2, x) / 0.5), (-2.0, -2.0, -2.0), 0.1 .* [1 0 0; 0 1 0; 0 0 1], (41, 41, 41))

# a = 2 inside the box [-0.5, 0.5]^3 and a0 = 1 outside
itf = box_interfaces([((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))], [2.0], 1.0)

r = cut_potential(src, itf, randn(3, 100); a0 = 1.0, tol = 1e-10, hess = true)
r.pot, r.grad, r.hess    # Nt, 3×Nt, 6×Nt (xx yy zz xy xz yz)
```

**Reusing work.** For many densities, geometries or target sets, build the steps once and reuse them:

```julia
field = SmoothField(src; tol = 1e-10)                                   # per density
sq    = SurfaceQuadrature(itf; support = density_support(src), L0 = 0.5)  # per geometry
plan  = CutPlan(sq, targets; a0 = 1.0, hess = true)                     # per target set
r     = evaluate(plan, field)
```

The [usage guide](https://xuanzhaogao.github.io/CutPotentials.jl/dev/usage/) covers general surfaces, on-surface
Hessians and the accuracy settings.

## How it works

The method never evaluates the discontinuous $a\rho$.

- **Smooth fields.** The volume part is computed only for the smooth $\rho$. Near the samples this uses the
  truncated-kernel method, with [FINUFFT](https://github.com/flatironinstitute/finufft); elsewhere it uses
  point charges summed by SCTL's FMM.
- **Interfaces.** Green's identities, plus Maue's identity for the Hessian, turn each jump of $a$ into single-
  and double-layer integrals of those smooth fields.
- **Surface integrals.** These are computed on SCTL's Gauss–Legendre surface elements, with its corrections for
  nearby and on-surface targets.

The derivation, measured accuracy, timings and limitations are in the
[documentation](https://xuanzhaogao.github.io/CutPotentials.jl/dev/).

## Development

```sh
julia --project=. -e 'using Pkg; Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl"); Pkg.build("SCTL")'
julia -t 8 --project=. -e 'using Pkg; Pkg.test()'      # CUTPOT_ONLY=<name> runs a subset
```

Contributor and agent notes are in [AGENT.md](AGENT.md).

## License

MIT, see [LICENSE](LICENSE).
