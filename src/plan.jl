# Geometry-only setup at a set of targets (near corrections, FMM trees) and the evaluation for a density.

const _LEVI = ((2, 3, 1), (3, 1, 2), (1, 2, 3))    # (j, k, l) with ε_jkl = +1; ε_jlk = -1

"""
    CutPlan(sq::SurfaceQuadrature, targets; a0, pot = true, grad = true, hess = false, side = nothing,
            hess_offset = 1e-5 * sq.L0, tol = 1e-12, fmm_digits = nothing)

Everything that depends on the geometry and the targets only, for evaluating `U = N[a ρ]` (and derivatives) at
`targets` (`3 × Nt`) for any density by [`evaluate`](@ref): SCTL's near corrections (accurate minus node rule, at
accuracy `tol`) of the double layer (`Laplace3D_DxU`), the single layer (`Laplace3D_FxU`) and, for `hess`, the
single-layer gradient (`Laplace3D_FxdU`) from the nodes of `sq` to the targets, and SCTL `ParticleFMM` trees for
their node rules (`fmm_digits`, default `-log10(tol) + 2`, at most 15). `a0` is the background coefficient.

**On-surface targets.** Potential and gradient need no flag: the formulas used are continuous across the
interfaces (see [`evaluate`](@ref)), so a target on a surface, or within rounding of it, gets the (unique) value.
The Hessian jumps across an interface; for a target on a surface give its column of `side` (`3 × Nt`, zero
columns for the other targets): a vector pointing into the side whose limit is wanted. That Hessian is extrapolated
from `y + kδν̂` (k = 1..4; `δ = hess_offset`), H ≈ 4H(δ) - 6H(2δ) + 4H(3δ) - H(4δ), with relative error about
(δ/ℓ)⁴, ℓ the distance to the nearest edge or corner of the interfaces (where H is singular) or the length scale of
the density. Keep δ well above the geometric error of curved elements (the shifted points must be on the wanted
side of the discretized surface). Unflagged targets are evaluated where they are; a target exactly in the plane of
a flat face gets the two-sided average of the Hessian.
"""
struct CutPlan
    sq::SurfaceQuadrature
    a0::Float64
    targets::Matrix{Float64}
    Ye::Matrix{Float64}             # evaluation points: the targets, then 4 shifted points per side target
    sidx::Vector{Int}               # the side targets (Hessian extrapolation)
    pot::Bool
    grad::Bool
    hess::Bool
    CD::SparseMatrixCSC{Float64, Int}
    CS::SparseMatrixCSC{Float64, Int}
    CG::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    fD::Union{Nothing, SCTL.ParticleFMM{SCTL.Laplace3D_DxU}}    # nothing: no surface nodes
    fS::Union{Nothing, SCTL.ParticleFMM{SCTL.Laplace3D_FxU}}
    fG::Union{Nothing, SCTL.ParticleFMM{SCTL.Laplace3D_FxdU}}
    hess_offset::Float64
    setup_time::Float64
end

function CutPlan(sq::SurfaceQuadrature, targets::AbstractMatrix{<:Real}; a0::Real, pot::Bool = true, grad::Bool = true,
                 hess::Bool = false, side = nothing, hess_offset::Real = 1e-5 * sq.L0, tol::Real = 1e-12,
                 fmm_digits::Union{Nothing, Integer} = nothing)
    t0 = time()
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × Nt"))
    (pot || grad || hess) || throw(ArgumentError("nothing to evaluate"))
    Y = Matrix{Float64}(targets); nt = size(Y, 2)
    sidx = Int[]; shifted = Vector{Float64}[]
    if side !== nothing
        hess || throw(ArgumentError("`side` selects the one-sided limit of the Hessian: pass hess = true"))
        size(side) == (3, nt) || throw(DimensionMismatch("side must be 3 × Nt"))
        for j in 1:nt
            ν = Float64.(side[:, j]); nν = norm(ν)
            nν == 0 && continue
            push!(sidx, j)
            for k in 1:4
                push!(shifted, Y[:, j] .+ (k * hess_offset / nν) .* ν)
            end
        end
    end
    Ye = isempty(shifted) ? Y : hcat(Y, reduce(hcat, shifted))
    el = sq.elist; ne = size(Ye, 2)
    if el === nothing
        z = spzeros(ne, 0)
        return CutPlan(sq, Float64(a0), Y, Ye, sidx, pot, grad, hess, z, z, hess ? spzeros(3ne, 0) : nothing,
                       nothing, nothing, nothing, Float64(hess_offset), time() - t0)
    end
    CD, _ = SCTL.near_correction(SCTL.Laplace3D_DxU(), el; targets = Ye, tol = tol)
    CS, _ = SCTL.near_correction(SCTL.Laplace3D_FxU(), el; targets = Ye, tol = tol)
    CG = hess ? first(SCTL.near_correction(SCTL.Laplace3D_FxdU(), el; targets = Ye, tol = tol)) : nothing
    d = fmm_digits === nothing ? clamp(ceil(Int, -log10(tol)) + 2, 1, 15) : Int(fmm_digits)
    fD = SCTL.ParticleFMM(SCTL.Laplace3D_DxU(); digits = d)
    SCTL.set_sources!(fD, sq.X, sq.normal); SCTL.set_targets!(fD, Ye)
    fS = SCTL.ParticleFMM(SCTL.Laplace3D_FxU(); digits = d)
    SCTL.set_sources!(fS, sq.X); SCTL.set_targets!(fS, Ye)
    fG = nothing
    if hess
        fG = SCTL.ParticleFMM(SCTL.Laplace3D_FxdU(); digits = d)
        SCTL.set_sources!(fG, sq.X); SCTL.set_targets!(fG, Ye)
    end
    return CutPlan(sq, Float64(a0), Y, Ye, sidx, pot, grad, hess, sparse(CD), sparse(CS),
                   CG === nothing ? nothing : sparse(CG), fD, fS, fG, Float64(hess_offset), time() - t0)
end

Base.show(io::IO, p::CutPlan) =
    print(io, "CutPlan(", size(p.targets, 2), " targets (", length(p.sidx), " on-surface Hessian), ", p.sq, ")")

# Σ_m δa_m K_m[f] at the evaluation points, for nodal values f (length Ns) already multiplied by δa: the node rule
# (FMM with weights) plus the near correction. Returns trg_dim × Ne.
function _layer(fmm, C::SparseMatrixCSC, w::Vector{Float64}, f::AbstractVector{Float64}, tdim::Int)
    fmm === nothing && return zeros(tdim, size(C, 1) ÷ tdim)
    SCTL.set_density!(fmm, reshape(w .* f, 1, :))
    U = SCTL.fmm_evaluate(fmm)
    U = reshape(U, tdim, :)
    U .+= reshape(C * f, tdim, :)
    return U
end

"""
    evaluate(plan::CutPlan, field::SmoothField) -> (; pot, grad, hess, coef, timings)

`U(y) = ∫ G(y, x) a(x) ρ(x) dx` (G = 1/(4π|y - x|)) at the plan's targets, with `grad = ∇U` (`3 × Nt`) and
`hess = ∇∇U` (`6 × Nt`, rows ∂xx, ∂yy, ∂zz, ∂xy, ∂xz, ∂yz), those not requested `nothing`. `coef` is the
coefficient the formulas see at each target, `a0 - W(y)` with `W = Σ_m δa_m D_m[1]`: a(y) off the interfaces,
(a_A + a_B)/2 on a smooth point of an interface, Σ_j α_j a_j (solid-angle fractions) at edges and corners.

With φ0, ψ = ∇φ0 and H = ∇ψ from `field` (the full, smooth density), the merged identities (Green's third identity
per region, the two copies of each shared face combined, W from the same quadrature) are

    U   = (a0 - W) φ0 + Σ_m δa_m (D_m[φ0] - S_m[ψ·n])
    ∇U  = (a0 - W) ψ  + Σ_m δa_m (D_m[ψ]  - S_m[H n - tr(H) n])
    ∇∂_iU = (a0 - W) H e_i + Σ_m δa_m (∇×S_m[n × H e_i] - ∇S_m[(H n)_i - tr(H) n_i])

all from nodal values of φ0, ψ, H on the surface elements (no differentiation on panels). The first two equal
`a0 φ0 + Σ δa D[φ0 - φ0(y)] - ...`, whose double layers have densities vanishing at y: continuous through the
surfaces, rounding-safe. The Hessian (the curl term is ∇D[ψ_i] by Maue's identity) is symmetrized.
"""
function evaluate(p::CutPlan, field)
    sq = p.sq; Ns = size(sq.X, 2); Ne = size(p.Ye, 2); nt = size(p.targets, 2)
    t0 = time()
    needH = p.grad || p.hess
    V = field_values(field, hcat(sq.X, p.Ye); order = needH ? 2 : 1)
    t1 = time()
    φn = V.pot[1:Ns]; ψn = V.grad[:, 1:Ns]
    φe = V.pot[Ns+1:end]; ψe = V.grad[:, Ns+1:end]
    δa = sq.δa; w = sq.w; N = sq.normal
    # W = Σ δa D[1]: the free term of every formula
    Wv = vec(_layer(p.fD, p.CD, w, δa, 1))
    ce = p.a0 .- Wv
    pot = nothing; grad = nothing; hess = nothing
    if p.pot
        ψn_n = vec(sum(ψn .* N; dims = 1))
        L = vec(_layer(p.fD, p.CD, w, δa .* φn, 1)) .- vec(_layer(p.fS, p.CS, w, δa .* ψn_n, 1))
        pot = (ce .* φe .+ L)[1:nt]
    end
    if needH
        Hn = V.hess[:, 1:Ns]
        # s = H n - tr(H) n at the nodes
        sd = zeros(3, Ns)
        @inbounds for i in 1:Ns
            h = view(Hn, :, i); n1, n2, n3 = N[1, i], N[2, i], N[3, i]; tr = h[1] + h[2] + h[3]
            sd[1, i] = h[1] * n1 + h[4] * n2 + h[5] * n3 - tr * n1
            sd[2, i] = h[4] * n1 + h[2] * n2 + h[6] * n3 - tr * n2
            sd[3, i] = h[5] * n1 + h[6] * n2 + h[3] * n3 - tr * n3
        end
    end
    if p.grad
        g = zeros(3, Ne)
        for i in 1:3
            g[i, :] .= vec(_layer(p.fD, p.CD, w, δa .* ψn[i, :], 1)) .- vec(_layer(p.fS, p.CS, w, δa .* sd[i, :], 1))
        end
        grad = (ce' .* ψe .+ g)[:, 1:nt]
    end
    if p.hess
        He = V.hess[:, Ns+1:end]
        M = zeros(3, 3, Ne)                               # M[j, i, :] = ∂_j ∂_i (layer part)
        for i in 1:3
            # v = n × (H e_i) at the nodes
            hi = Hn[collect(i == 1 ? (1, 4, 5) : i == 2 ? (4, 2, 6) : (5, 6, 3)), :]
            v = zeros(3, Ns)
            @inbounds for q in 1:Ns
                n1, n2, n3 = N[1, q], N[2, q], N[3, q]; a, b, c = hi[1, q], hi[2, q], hi[3, q]
                v[1, q] = n2 * c - n3 * b; v[2, q] = n3 * a - n1 * c; v[3, q] = n1 * b - n2 * a
            end
            Gv = [_layer(p.fG, p.CG, w, δa .* v[l, :], 3) for l in 1:3]          # Gv[l][k, :] = ∂_k S[δa v_l]
            Gs = _layer(p.fG, p.CG, w, δa .* sd[i, :], 3)
            for (j, k, l) in _LEVI
                M[j, i, :] .+= Gv[l][k, :] .- Gv[k][l, :]
            end
            for j in 1:3
                M[j, i, :] .-= Gs[j, :]
            end
        end
        Hh = zeros(6, Ne)
        for (r, (a, b)) in enumerate(((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)))
            Hh[r, :] .= ce .* He[r, :] .+ (M[a, b, :] .+ M[b, a, :]) ./ 2
        end
        hess = Hh[:, 1:nt]
        for (s, j) in enumerate(p.sidx)
            c = nt + 4 * (s - 1)
            hess[:, j] .= 4 .* Hh[:, c + 1] .- 6 .* Hh[:, c + 2] .+ 4 .* Hh[:, c + 3] .- Hh[:, c + 4]
        end
    end
    t2 = time()
    return (; pot, grad, hess, coef = ce[1:nt], timings = (; field = t1 - t0, layers = t2 - t1))
end

"""
    cut_potential(src, interfaces, targets; a0, tol = 1e-10, support = density_support(src), L0, κ = 0.5,
                  order = 12, pot = true, grad = true, hess = false, side = nothing, quad_tol = 1e-12, kw...)

One-shot convenience: [`SmoothField`](@ref) of `src` (accuracy `tol`), [`SurfaceQuadrature`](@ref) of `interfaces`
graded toward `support` (default `L0`: a quarter of the smallest side of the support box, at least 2‖basis‖), a
[`CutPlan`](@ref) at `targets` and [`evaluate`](@ref). For many densities or target batches build the pieces once.
"""
function cut_potential(src::VolumeSource, interfaces::AbstractVector{Interface}, targets::AbstractMatrix; a0::Real,
                       tol::Real = 1e-10, support = density_support(src), L0 = nothing, κ::Real = 0.5,
                       order::Int = 12, pot::Bool = true, grad::Bool = true, hess::Bool = false, side = nothing,
                       quad_tol::Real = 1e-12, max_size::Real = Inf)
    f = SmoothField(src; tol)
    l0 = L0 === nothing ? max(minimum(support[2][d] - support[1][d] for d in 1:3) / 4, 2 * opnorm(src.basis)) : L0
    sq = SurfaceQuadrature(interfaces; support, L0 = l0, κ, order, max_size)
    p = CutPlan(sq, targets; a0, pot, grad, hess, side, tol = quad_tol)
    return evaluate(p, f)
end
