# Usage

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
