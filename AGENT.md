# AGENT.md: guidelines for coding agents working on CutPotentials.jl

## What the package does

CutPotentials.jl evaluates U = ∫ G a ρ, ∇U and ∇∇U for G = 1/(4π|y − x|), a smooth sampled density ρ and a
piecewise-constant coefficient a.

- **Volume work.** It stays on the smooth ρ:
  - TKM (FINUFFT type-1 times the truncated kernel, gathered with an ES window and its derivatives) inside the
    padded sample box;
  - SCTL `ParticleFMM` point charges outside it.
- **The jumps of aρ.** Green's identities (with Maue's identity for the Hessian) turn them into single- and
  double-layer integrals of φ0, ψ and H. These are discretized on SCTL Gauss–Legendre elements, with SCTL's
  near/singular corrections.

The derivation is in `docs/src/method.md`. Read it before changing any formula.

## Layout

| path | contents |
|---|---|
| `src/CutPotentials.jl` | module, exports, module docstring (conventions) |
| `src/source.jl` | `VolumeSource`, `density_support` |
| `src/tkm.jl` | `SmoothField`, `field_values` (TKM grid, window gather, point-charge far field) |
| `src/surfaces.jl` | `Interface`, `rectangle_patch`, `box_interfaces`, `sphere_interface`, `ellipsoid_interface`, `SurfaceQuadrature` |
| `src/plan.jl` | `CutPlan` (near corrections, FMM trees), `evaluate`, `cut_potential` |
| `test/runtests.jl`, `test/references.jl` | accuracy tests against independent references (erf/heat kernels, adaptive cubature) |
| `docs/` | Documenter site: `make.jl`; `src/` holds index, method, usage, accuracy, api |
| `.github/workflows/CI.yml` | tests on Julia lts / 1 / pre and a docs build + deploy |

## Dependencies: SCTL.jl is not registered

SCTL.jl (https://github.com/xuanzhaogao/SCTL.jl) wraps a C++ library through a shim that `Pkg.build("SCTL")`
compiles.

- **No `[sources]` entry.** Do not add one with a local path to `Project.toml`.
- **No `Pkg.add(url = ...)`.** It fails on SCTL.jl's git submodules.
- **Develop it instead.** A develop clone builds from the sources pinned in SCTL.jl's `deps/sources.toml`.
- **On CI,** `SCTL_JL_MARCH=-march=x86-64-v3` selects a portable build. Without it, the default is `-march=native`.

## Developer setup

```sh
julia --project=. -e 'using Pkg; Pkg.develop(url = "https://github.com/xuanzhaogao/SCTL.jl"); Pkg.build("SCTL"); Pkg.instantiate()'
```

If SCTL.jl is already checked out locally, use `Pkg.develop(path = "/path/to/SCTL.jl")` instead.
`Manifest.toml` is git-ignored.

## Tests

```sh
OMP_NUM_THREADS=8 julia -t 8 --project=. -e 'using Pkg; Pkg.test()'
```

- **Test dependencies.** They are in `[extras]`/`[targets]` in `Project.toml`. There is no `test/Project.toml`,
  because Julia 1.10 (the lts) cannot resolve `[sources]`.
- **If `Pkg.test()` is unusable locally** (a broken registry setup), run the file in a test environment:
  `julia --project=. -e 'using TestEnv; TestEnv.activate(); include("test/runtests.jl")'`.
- **Environment variables:**
  - `CUTPOT_ONLY=<substring>` runs only the matching testsets;
  - `CUTPOT_VERBOSE=0` silences the error tables.
- **Duration.** The full suite has 74 checks and takes ~70 s at 16 threads, several minutes at 2 (CI).

## Docs

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path = "."), PackageSpec(url = "https://github.com/xuanzhaogao/SCTL.jl")]); Pkg.build("SCTL"); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

- **Output.** The build goes to `docs/build/`.
- **Deployment.** CI deploys `main` to the `gh-pages` branch. In the repository settings, Pages must serve from
  `gh-pages`.
- **Docstrings.** `api.md` collects all of them with `@autodocs`. Every exported name needs a docstring.

## Conventions

- **Kernels.**
  - G = 1/(4π r) and S[q] = ∫ G q dS.
  - D[q] = ∫ ∂_{n_x} G q dS, with the normal n of the integration point.
  - Δφ0 = −ρ.
- **Interfaces.** Each is stored once:
  - its normal points from side A to side B;
  - its coefficient is δa = a_A − a_B.
  - Patches are functions X(u, v) on [0, 1]², with ∂uX × ∂vX along that normal.
  - Interfaces with δa = 0 may be dropped.
  - The set must form closed region boundaries. Nothing checks this.
- **Free term.** It is coef = a0 − W, with W = Σ δa D[1] computed by the same quadrature. Never snap targets to
  surfaces or special-case on-surface points for U and ∇U; continuity comes from the densities f − f(y).
- **Hessian.**
  - Storage is 6 × Nt in the order xx, yy, zz, xy, xz, yz.
  - On-surface targets need a `side` vector; the one-sided limit is by 4-point extrapolation (`hess_offset`).
- **Types.** `Float64` only, MPI-free.
- **Style.** Match the surrounding code: short comments stating the math, keyword arguments with defaults, and no
  new dependencies without need.

## Rules for changes

- **Measure every accuracy claim.** Add or extend a testset in `test/runtests.jl` that compares against an
  independent reference, and update the tables in `docs/src/accuracy.md` and the `README.md`.
- **Do not loosen test tolerances** to make a change pass. Find the cause.
- **Keep the README and the docs in sync** when the API or a measured number changes.
- **Run the full test suite** before committing. CI must be green before merging.
