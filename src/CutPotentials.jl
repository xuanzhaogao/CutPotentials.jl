"""
    CutPotentials

Newton potentials of a smooth density screened by a piecewise-constant coefficient,

    U(y) = ∫ G(y, x) a(x) ρ(x) dx,    G = 1/(4π|y - x|),   a = a_j on regions Ω_j, a0 outside,

with ∇U and ∇∇U, at arbitrary targets (inside or outside the regions, near or on their surfaces). Every spectral or
point-charge computation acts on the smooth, unscreened ρ ([`SmoothField`](@ref): TKM near the samples, SCTL FMM
elsewhere); the jumps of aρ become surface integrals of the smooth fields φ0 = N[ρ], ψ = ∇φ0, H = ∇ψ over the
interfaces ([`Interface`](@ref), [`SurfaceQuadrature`](@ref): SCTL Gauss-Legendre elements graded toward the
density, FMM plus SCTL near corrections, [`CutPlan`](@ref)).

Workflow:

    src   = VolumeSource(values, origin, basis)                  # samples on a lattice
    field = SmoothField(src; tol = 1e-10)
    itf   = vcat(box_interfaces(boxes, a, a0), [sphere_interface(c, R, a_in - a_out)])
    sq    = SurfaceQuadrature(itf; support = density_support(src), L0 = σ)
    plan  = CutPlan(sq, targets; a0, hess = true, side = side_normals)
    r     = evaluate(plan, field)                                 # r.pot, r.grad, r.hess, r.coef

Conventions: interface normals point from side A to side B and `δa = a_A - a_B`; Hessian rows are ∂xx, ∂yy, ∂zz,
∂xy, ∂xz, ∂yz; on-surface Hessians are one-sided limits on the side the target's `side` vector points to.
"""
module CutPotentials

using LinearAlgebra
using SparseArrays
using FFTW
using FINUFFT
using SCTL

export VolumeSource, density_support
export SmoothField, field_values
export Interface, rectangle_patch, box_interfaces, sphere_interface, ellipsoid_interface
export SurfaceQuadrature, CutPlan, evaluate, cut_potential

include("source.jl")
include("tkm.jl")
include("surfaces.jl")
include("plan.jl")

end # module CutPotentials
