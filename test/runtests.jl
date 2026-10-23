# Run with: julia --project=. -e "using Pkg; Pkg.test()"   (after developing SCTL.jl: see README / AGENT.md)
using CutPotentials
using Test, LinearAlgebra, Printf
using HCubature: hcubature
include("references.jl")

const VERBOSE = get(ENV, "CUTPOT_VERBOSE", "1") == "1"
const ONLY = get(ENV, "CUTPOT_ONLY", "")            # run only the testsets whose name contains this
want(name) = isempty(ONLY) || occursin(ONLY, name)

# max error / max |reference| over the targets, for a vector or a matrix (columns = targets)
relerr(x::AbstractVector, r::AbstractVector) = maximum(abs, x .- r) / maximum(abs, r)
relerr(x::AbstractMatrix, r::AbstractMatrix) =
    maximum(norm(x[:, j] - r[:, j]) for j in axes(r, 2)) / maximum(norm(r[:, j]) for j in axes(r, 2))

# reference values of F (scalar function of y) at the targets; with a side vector the Hessian is the one-sided limit,
# extrapolated from k·1e-6 (k = 1..4) along it (the references are smooth on each side)
function reference(F, Y, side)
    n = size(Y, 2); U = zeros(n); G = zeros(3, n); H = zeros(6, n)
    Threads.@threads for j in 1:n
        y = Y[:, j]; ν = side === nothing ? zeros(3) : side[:, j]
        u, g, h = vgh(F, y)
        if norm(ν) > 0
            ν = ν ./ norm(ν)
            hk = [vgh(F, y .+ (k * 1e-6) .* ν)[3] for k in 1:4]
            h = 4 .* hk[1] .- 6 .* hk[2] .+ 4 .* hk[3] .- hk[4]
        end
        U[j] = u; G[:, j] .= g; H[:, j] .= h
    end
    return U, G, H
end

report(name, e) = VERBOSE && @info @sprintf("%-46s pot %.1e  grad %.1e  hess %.1e", name, e...)

# ---- the density: a Gaussian sampled on a skewed, anisotropic lattice (±7σ) ----
const G1 = Gaussian((0.6, 0.05, 0.1), 0.5)
const BASIS = [0.1 0.02 0.0; 0.0 0.11 0.01; 0.0 0.0 0.09]
function lattice_source(g::Gaussian)
    R = 7 * g.σ
    o = collect(g.c) .- R .- [0.013, -0.021, 0.034] .- BASIS * [3, 3, 3]
    dims = Tuple(ceil.(Int, 2R ./ [0.08, 0.09, 0.08]) .+ 10)
    return VolumeSource(x -> ρ(g, x), o, BASIS, dims)
end
const SRC = lattice_source(G1)
const FIELD = SmoothField(SRC; tol = 1e-12)
const EXACT = ExactField(G1)

@testset "CutPotentials" begin

want("smooth field") && @testset "smooth field (TKM near, FMM far) against the exact Gaussian fields" begin
    Y = reduce(hcat, [collect(G1.c) .+ [1.3cos(k), 1.1sin(k), 0.7cos(3k)] .* (k / 20) for k in 1:40])
    Y = hcat(Y, [10.0, 0, 0], [0.0, 0.6, 7.0])
    V = field_values(FIELD, Y); E = field_values(EXACT, Y)
    e = (relerr(V.pot, E.pot), relerr(V.grad, E.grad), relerr(V.hess, E.hess))
    report("smooth field", e)
    @test all(e .< 1e-10)
    # the window: an explicit width reproduces the default rule's; without the taper the Hessian is wrong
    @test FIELD.win.ns == 20 && FIELD.win.m == 3
    V20 = field_values(SmoothField(SRC; tol = 1e-12, window_ns = 20), Y)
    @test V20.hess ≈ V.hess rtol = 1e-14
    # targets whose stencils end 1e-9 (grid units) before the window's support ends: the untapered window's w″ is
    # singular there, the tapered one's vanishes
    function transition_target(f, y, δ)
        g = f.geom; ns = f.win.ns; y2 = copy(y)
        for d in 1:3
            nf = f.nf[d]; x = g.dks[d] * (y[d] - g.center[d]) / (2π) + 0.5
            j = floor(x); u = (x - j) * nf
            y2[d] = g.center[d] + 2π * (((floor(u - ns / 2) + ns / 2 + δ) / nf + j) - 0.5) / g.dks[d]
        end
        return y2
    end
    F0 = SmoothField(SRC; tol = 1e-12, window_taper = 0)
    Yt = reduce(hcat, [transition_target(FIELD, Y[:, j], 1e-9) for j in axes(Y, 2)])
    Et = field_values(EXACT, Yt)
    @test relerr(field_values(FIELD, Yt).hess, Et.hess) < 1e-10
    @test relerr(field_values(F0, Yt).hess, Et.hess) > 1e-3
    @test_throws ArgumentError SmoothField(SRC; window_ns = 1)
end

want("box regions") && @testset "box regions: erf/heat-kernel reference" begin
    xe = 0.8
    cases = [("one box", [((-2.0, -2.0, -2.0), (xe, 2.0, 0.0))], [0.25]),
             ("four boxes along an edge", [((-2.0, -2.0, -2.0), (xe, 2.0, 0.0)), ((xe, -2.0, -2.0), (3.0, 2.0, 0.0)),
                                           ((-2.0, -2.0, 0.0), (xe, 2.0, 2.0)), ((xe, -2.0, 0.0), (3.0, 2.0, 2.0))],
              [1 / 4, 1 / 12, 1 / 2, 1 / 3]),
             ("three boxes: shared faces, a corner", [((-2.0, -2.0, -2.0), (xe, 2.0, 0.0)), ((xe, -2.0, -2.0), (3.0, 0.3, 0.0)),
                                                      ((-1.0, -1.5, 0.0), (xe, 0.3, 1.5))], [0.2, 0.5, 0.1])]
    a0 = 1.0
    # volume targets (in and out), near and on the faces z = 0 and x = 0.8, near and on their common edge
    Yv = [[0.3, 0.2, -0.5], [1.2, 0.3, 0.4], [0.2, -0.4, 0.6], [1.5, 1.0, -0.4], [3.5, 2.5, 2.5], [-2.6, 0.1, 0.0]]
    Yn = vec([[0.6, 0.05, s * d] for d in (1e-3, 1e-8, 1e-12), s in (1, -1)])
    append!(Yn, vec([[xe + s * d, 0.4, -0.3] for d in (1e-3, 1e-8), s in (1, -1)]))
    append!(Yn, [[xe - 1e-3, 0.05, 1e-4], [xe + 1e-6, 0.05, -1e-6]])
    Ys = [[0.6, 0.05, 0.0], [0.4, 0.6, 0.0], [1.3, -0.3, 0.0], [xe, 0.05, -0.3], [xe, -0.5, 0.4], [xe - 1e-3, 0.05, 0.0],
          [xe, 0.05, 1e-3]]
    nrm = [[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]
    Ye = [[xe - 1e-4, 0.05, 0.0], [xe - 1e-6, 0.05, 0.0], [xe, 0.05, 1e-6], [xe, 0.05, 0.0], [xe, 0.3, 0.0]]
    for (name, boxes, a) in cases
        itf = box_interfaces(boxes, a, a0)
        sq = SurfaceQuadrature(itf; support = (G1.c, G1.c), L0 = G1.σ)
        F(z) = a0 * φ0(G1, z) + sum((a[j] - a0) * Nbox(G1, z, boxes[j]...) for j in eachindex(boxes))
        # off-surface targets; on-surface ones twice (Hessian limits from both sides); pot/grad at edge points
        Yoff = reduce(hcat, vcat(Yv, Yn))
        Yon = reduce(hcat, vcat(Ys, Ys)); Son = reduce(hcat, vcat(.-nrm, nrm))
        Yedge = reduce(hcat, Ye)
        Uo, Go, Ho = reference(F, Yoff, nothing)
        Us, Gs, Hs = reference(F, Yon, Son)
        Ue, Ge, _ = reference(F, Yedge, nothing)
        for (fname, fld) in (("TKM", FIELD), ("exact ψ, H", EXACT))
            ro = evaluate(CutPlan(sq, Yoff; a0, hess = true), fld)
            rs = evaluate(CutPlan(sq, Yon; a0, hess = true, side = Son), fld)
            re = evaluate(CutPlan(sq, Yedge; a0, grad = true, hess = false), fld)
            eo = (relerr(ro.pot, Uo), relerr(ro.grad, Go), relerr(ro.hess, Ho))
            es = (relerr(rs.pot, Us), relerr(rs.grad, Gs), relerr(rs.hess, Hs))
            ee = (relerr(re.pot, Ue), relerr(re.grad, Ge), NaN)
            report("$name, $fname: off-surface", eo)
            report("$name, $fname: on faces (one-sided H)", es)
            report("$name, $fname: at/near the edge", ee)
            @test all(eo .< 1e-9)
            @test all(es .< 1e-9)
            @test ee[1] < 1e-9 && ee[2] < 1e-9
            # the Laplacian: tr H = -a ρ, off the surfaces and one-sided on them (a of the side ν points to)
            coef(y) = (for (j, b) in enumerate(boxes)
                           all(b[1][d] < y[d] < b[2][d] for d in 1:3) && return a[j]
                       end; a0)
            ao = [coef(Yoff[:, j]) for j in axes(Yoff, 2)]
            as = [coef(Yon[:, j] .+ 1e-6 .* Son[:, j]) for j in axes(Yon, 2)]
            hs = maximum(norm, eachcol(Ho))
            @test maximum(abs.(vec(sum(ro.hess[1:3, :]; dims = 1)) .+ ao .* [ρ(G1, y) for y in eachcol(Yoff)])) < 1e-9 * hs
            @test maximum(abs.(vec(sum(rs.hess[1:3, :]; dims = 1)) .+ as .* [ρ(G1, y) for y in eachcol(Yon)])) < 1e-9 * hs
            # the coefficient seen by the formulas: a off the surfaces, the average on a face
            @test maximum(abs, ro.coef .- ao) < 1e-10
            # on a face: the average of the two sides
            aavg = [(coef(Yon[:, j] .+ 1e-6 .* Son[:, j]) + coef(Yon[:, j] .- 1e-6 .* Son[:, j])) / 2 for j in axes(Yon, 2)]
            @test maximum(abs, rs.coef .- aavg) < 1e-10
        end
        VERBOSE && @info "  $name: $(length(itf)) interface rectangles, $(size(sq.X, 2)) surface nodes"
    end
end

want("sphere: noncentral") && @testset "sphere: noncentral-χ heat-kernel reference" begin
    b = (0.1, 0.0, 0.2); R = 0.9; ain = 0.2; a0 = 1.0
    sq = SurfaceQuadrature([sphere_interface(b, R, ain - a0)]; support = (G1.c, G1.c), L0 = G1.σ)
    F(z) = a0 * φ0(G1, z) + (ain - a0) * Nball(G1, z, b, R)
    dirs = normalize.([[0.3, 0.5, 0.8], [-1.0, 0.2, 0.1], [0.2, -0.1, 1.0], [0.9, 0.1, -0.4]])
    # the reference loses ~eps/d in its Hessian at distance d from the sphere: Hessians at d ≥ 1e-6, pot/grad closer
    Yoff = reduce(hcat, [collect(b) .+ (R + o) .* d for d in dirs for o in (-0.5, -1e-3, -1e-6, 1e-6, 1e-3, 0.6)])
    Yclose = reduce(hcat, [collect(b) .+ (R + o) .* d for d in dirs for o in (-1e-9, 1e-9, -1e-13, 1e-13)])
    Yon = reduce(hcat, [collect(b) .+ R .* d for d in dirs for _ in 1:2])
    Son = reduce(hcat, [s .* d for d in dirs for s in (-1, 1)])
    Uo, Go, Ho = reference(F, Yoff, nothing)
    Uc, Gc, _ = reference(F, Yclose, nothing)
    Us, Gs, Hs = reference(F, Yon, Son)
    for (fname, fld) in (("TKM", FIELD), ("exact ψ, H", EXACT))
        ro = evaluate(CutPlan(sq, Yoff; a0, hess = true), fld)
        rc = evaluate(CutPlan(sq, Yclose; a0), fld)
        rs = evaluate(CutPlan(sq, Yon; a0, hess = true, side = Son), fld)
        eo = (relerr(ro.pot, Uo), relerr(ro.grad, Go), relerr(ro.hess, Ho))
        ec = (relerr(rc.pot, Uc), relerr(rc.grad, Gc), NaN)
        es = (relerr(rs.pot, Us), relerr(rs.grad, Gs), relerr(rs.hess, Hs))
        report("sphere, $fname: off-surface", eo)
        report("sphere, $fname: within 1e-9..1e-13", ec)
        report("sphere, $fname: on the surface (one-sided H)", es)
        @test all(eo .< 1e-9)
        @test ec[1] < 1e-9 && ec[2] < 1e-9
        @test all(es .< 1e-9)
        ao = [norm(y .- b) < R ? ain : a0 for y in eachcol(Yoff)]
        as = [dot(Son[:, j], Yon[:, j] .- b) < 0 ? ain : a0 for j in axes(Yon, 2)]
        hs = maximum(norm, eachcol(Ho))
        @test maximum(abs.(vec(sum(ro.hess[1:3, :]; dims = 1)) .+ ao .* [ρ(G1, y) for y in eachcol(Yoff)])) < 1e-9 * hs
        @test maximum(abs.(vec(sum(rs.hess[1:3, :]; dims = 1)) .+ as .* [ρ(G1, y) for y in eachcol(Yon)])) < 1e-9 * hs
        # a point of the exact sphere is within rounding of the discrete surface, on either side or on it: the
        # coefficient seen is a_in, a0 or their average (pot and grad do not depend on which)
        @test all(c -> minimum(abs, c .- (ain, a0, (ain + a0) / 2)) < 1e-10, rs.coef)
    end
    VERBOSE && @info "  sphere: $(size(sq.X, 2)) surface nodes"
end

want("rotated ellipsoid") && @testset "rotated ellipsoid: adaptive cubature of the screened volume integral" begin
    c = (0.4, 0.1, 0.0); ax = (1.0, 0.6, 0.45); a1 = 0.3; a0 = 1.0
    θ = 0.4; Rz = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]; Rx = [1 0 0; 0 cos(0.3) -sin(0.3); 0 sin(0.3) cos(0.3)]
    Rm = Rz * Rx
    sq = SurfaceQuadrature([ellipsoid_interface(c, ax, a1 - a0; R = Rm)]; support = (G1.c, G1.c), L0 = G1.σ)
    # targets at distance ≥ 0.15 from the surface: the volume integrand is smooth. Inside: integrate over the outside
    # (N[χ_E ρ] = φ0 - N[χ_out ρ]), radially out to 8σ beyond the ellipsoid.
    E(r, t, p) = collect(c) .+ Rm * (ax .* (r .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)]))
    jac(r, t) = prod(ax) * r^2 * sin(t)
    function Nell(y, inside)
        f = function (u)
            r, t, p = u[1], u[2], u[3]
            x = E(r, t, p); d = y .- x; dn = norm(d)
            w = jac(r, t) * ρ(G1, x) / (4π)
            vcat(w / dn, -w .* d ./ dn^3, w .* vec((3 * (d * d') - dn^2 * I) / dn^5))
        end
        lo, hi = inside ? ([1.0, 0, 0], [1 + 8G1.σ / minimum(ax) + 2, π, 2π]) : ([0.0, 0, 0], [1.0, π, 2π])
        v = hcubature(f, lo, hi; rtol = 1e-11, atol = 1e-13, maxevals = 5_000_000)[1]
        return v
    end
    Y = [[0.4, 0.1, 0.0], [0.9, 0.3, 0.1], [1.8, 0.4, 0.3], [0.6, 0.05, 0.9]]
    insides = Bool[]; U = Float64[]; Gr = zeros(3, 0); Hh = zeros(6, 0)
    tcub = @elapsed for y in Y
        s = Rm' * (y .- collect(c)) ./ collect(ax); inside = norm(s) < 1
        v = Nell(y, inside)
        if inside      # N[χ_E ρ] = φ0 - N[χ_out ρ]
            v = vcat(φ0(G1, y), ψ0(G1, y), [H0(G1, y)[i] for i in (1, 4, 5, 4, 2, 6, 5, 6, 3)]) .- v
        end
        Hm = reshape(v[5:13], 3, 3)
        push!(insides, inside)
        push!(U, a0 * φ0(G1, y) + (a1 - a0) * v[1])
        Gr = hcat(Gr, a0 .* ψ0(G1, y) .+ (a1 - a0) .* v[2:4])
        Hh = hcat(Hh, a0 .* H0(G1, y) .+ (a1 - a0) .* [Hm[1, 1], Hm[2, 2], Hm[3, 3], Hm[1, 2], Hm[1, 3], Hm[2, 3]])
    end
    @test count(insides) >= 2 && count(!, insides) >= 2
    Ym = reduce(hcat, Y)
    r = evaluate(CutPlan(sq, Ym; a0, hess = true), FIELD)
    e = (relerr(r.pot, U), relerr(r.grad, Gr), relerr(r.hess, Hh))
    report("rotated ellipsoid, TKM (cubature reference)", e)
    @test all(e .< 1e-8)
    @test maximum(abs, r.coef .- [i ? a1 : a0 for i in insides]) < 1e-10
    VERBOSE && @info "  ellipsoid: $(size(sq.X, 2)) surface nodes; cubature reference $(round(tcub, digits = 1)) s"
end

want("sphere inside a box") && @testset "sphere inside a box (mixed interfaces)" begin
    b = (0.5, 0.0, 0.0); R = 0.6; box = ((-0.5, -1.0, -0.8), (1.5, 1.2, 1.0)); a1 = 0.15; a2 = 0.4; a0 = 1.0
    itf = vcat(box_interfaces([box], [a2], a0), [sphere_interface(b, R, a1 - a2)])
    sq = SurfaceQuadrature(itf; support = (G1.c, G1.c), L0 = G1.σ)
    F(z) = a0 * φ0(G1, z) + (a2 - a0) * Nbox(G1, z, box...) + (a1 - a2) * Nball(G1, z, b, R)
    Y = reduce(hcat, [[0.5, 0.1, 0.2], [1.3, 0.1, 0.2], [0.6, 0.05, 1.3], [0.9, 0.3, 1.0 - 1e-7], [0.5, 0.0, 0.6 + 1e-3]])
    Yon = reduce(hcat, [[0.6, 0.05, 1.0], [0.6, 0.05, 1.0], collect(b) .+ R .* normalize([0.2, 0.1, 1.0])])
    Son = reduce(hcat, [[0, 0, 1.0], [0, 0, -1.0], -normalize([0.2, 0.1, 1.0])])
    U, Gr, Hh = reference(F, Y, nothing); Us, Gs, Hs = reference(F, Yon, Son)
    r = evaluate(CutPlan(sq, Y; a0, hess = true), FIELD)
    rs = evaluate(CutPlan(sq, Yon; a0, hess = true, side = Son), FIELD)
    e = (relerr(r.pot, U), relerr(r.grad, Gr), relerr(r.hess, Hh))
    es = (relerr(rs.pot, Us), relerr(rs.grad, Gs), relerr(rs.hess, Hs))
    report("sphere in a box, TKM: off-surface", e)
    report("sphere in a box, TKM: on surfaces", es)
    @test all(e .< 1e-9)
    @test all(es .< 1e-9)
    @test r.coef ≈ [a1, a2, a0, a2, a2] atol = 1e-10
end

want("equal coefficients") && @testset "equal coefficients reduce to a0 times the smooth field" begin
    Y = reduce(hcat, [[0.3, 0.2, -0.5], [0.8, 0.05, 0.0], [0.6, 0.05, 1e-9], [2.0, 1.0, 1.0], [0.8, 0.05, -0.3]])
    a0 = 0.7
    S = field_values(FIELD, Y)
    # every box has a = a0: all faces drop out
    itf = box_interfaces([((-2.0, -2.0, -2.0), (0.8, 2.0, 0.0)), ((0.8, -2.0, -2.0), (3.0, 2.0, 0.0))], [a0, a0], a0)
    @test isempty(itf)
    r = evaluate(CutPlan(SurfaceQuadrature(itf; support = (G1.c, G1.c), L0 = G1.σ), Y; a0, hess = true), FIELD)
    @test r.pot ≈ a0 .* S.pot rtol = 1e-14
    @test r.grad ≈ a0 .* S.grad rtol = 1e-14
    @test r.hess ≈ a0 .* S.hess rtol = 1e-14
    # an interface and its reverse (same sphere, opposite δa): the machinery runs and cancels
    sph = sphere_interface((0.6, 0.0, 0.0), 0.7, 0.5)
    rev = Interface([(u, v) -> p(v, u) for p in sph.patches], 0.5)            # swapped parameters: inward normal
    sq = SurfaceQuadrature([sph, rev]; support = (G1.c, G1.c), L0 = G1.σ)
    r = evaluate(CutPlan(sq, Y; a0, hess = true), FIELD)
    e = (relerr(r.pot, a0 .* S.pot), relerr(r.grad, a0 .* S.grad), relerr(r.hess, a0 .* S.hess))
    report("equal coefficients (cancelling interfaces)", e)
    @test all(e .< 1e-12)
    @test maximum(abs, r.coef .- a0) < 1e-12
    # two boxes with the same a: the merged shared face drops out, equal to the union box and to the
    # region-by-region (unmerged) faces
    ab = 0.25; a0 = 1.0
    b1 = ((-2.0, -2.0, -2.0), (0.8, 2.0, 0.0)); b2 = ((0.8, -2.0, -2.0), (3.0, 2.0, 0.0))
    merged = box_interfaces([b1, b2], [ab, ab], a0)
    union = box_interfaces([((-2.0, -2.0, -2.0), (3.0, 2.0, 0.0))], [ab], a0)
    perregion = vcat(box_interfaces([b1], [ab], a0), box_interfaces([b2], [ab], a0))
    @test !any(i -> i.name == "x1 = 0.8", merged) && any(i -> i.name == "x1 = 0.8", perregion)
    res = [evaluate(CutPlan(SurfaceQuadrature(i; support = (G1.c, G1.c), L0 = G1.σ), Y; a0, hess = true), FIELD)
           for i in (merged, union, perregion)]
    for k in 2:3
        @test relerr(res[k].pot, res[1].pot) < 1e-10
        @test relerr(res[k].grad, res[1].grad) < 1e-10
    end
    # Hessians off the faces (the targets on z = 0 or x = 0.8 are unflagged: two-sided averages)
    off = [1, 3, 4]
    @test relerr(res[2].hess[:, off], res[1].hess[:, off]) < 1e-9
    @test relerr(res[3].hess[:, off], res[1].hess[:, off]) < 1e-9
end

end
