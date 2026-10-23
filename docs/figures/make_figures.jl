# Figures for README.md and docs/src/examples.md, written to docs/src/assets/.
#
# Not run by the docs build (it takes a few minutes and needs a plotting package). Set up an environment with
# CutPotentials, SCTL.jl, CairoMakie, ForwardDiff and SpecialFunctions (see docs/figures/Project.toml), e.g.
#
#     julia --project=docs/figures -e 'using Pkg; Pkg.develop([PackageSpec(path = "."),
#         PackageSpec(url = "https://github.com/xuanzhaogao/SCTL.jl")]); Pkg.build("SCTL"); Pkg.instantiate()'
#     OMP_NUM_THREADS=16 julia -t 16 --project=docs/figures docs/figures/make_figures.jl [sphere errors resolution distance]
#
# The references are those of the tests (test/references.jl): heat-kernel formulas for a Gaussian truncated to boxes
# and balls, differentiated by ForwardDiff.
using CutPotentials
using CairoMakie
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "..", "..", "test", "references.jl"))

const OUT = joinpath(@__DIR__, "..", "src", "assets")
mkpath(OUT)

set_theme!(Theme(fontsize = 17, font = "TeX Gyre Heros Makie",
                 Axis = (xgridvisible = false, ygridvisible = false, xtickalign = 1, ytickalign = 1,
                         titlefont = :regular),
                 Colorbar = (tickalign = 1,)))

# ---- the example: a Gaussian density (σ = 0.5) that straddles a sphere with a = 0.2 inside, a0 = 1 outside ----
const G1 = Gaussian((0.6, 0.05, 0.1), 0.5)
const SPH = (b = (0.1, 0.0, 0.2), R = 0.9, ain = 0.2, a0 = 1.0)
acoef(x) = norm(x .- SPH.b) < SPH.R ? SPH.ain : SPH.a0
Fsphere(z) = SPH.a0 * φ0(G1, z) + (SPH.ain - SPH.a0) * Nball(G1, z, SPH.b, SPH.R)

# samples on a cubic lattice of spacing h covering ±nσ σ around the centre
function lattice_source(f, h; nσ = 8)
    R = nσ * G1.σ; n = ceil(Int, 2R / h) + 1
    return VolumeSource(f, collect(G1.c) .- R, h .* Matrix(1.0I, 3, 3), (n, n, n))
end
const HS = 0.1
const SRC = lattice_source(x -> ρ(G1, x), HS)
const FIELD = SmoothField(SRC; tol = 1e-12)
const ITF = [sphere_interface(SPH.b, SPH.R, SPH.ain - SPH.a0)]
# grade the elements toward the density's centre, as the tests do. The default support of `cut_potential`,
# density_support(src) (samples above 1e-10 max: ±6.8σ here), contains the whole sphere, so it gives uniform elements
# of size κ L0: 672 of them against 177, for the same accuracy
const SQ = SurfaceQuadrature(ITF; support = (G1.c, G1.c), L0 = G1.σ)

function reference_fields(F, Y; hess = false)
    n = size(Y, 2); U = zeros(n); G = zeros(3, n); H = zeros(6, n)
    Threads.@threads for j in 1:n
        if hess
            u, g, h = vgh(F, Y[:, j]); U[j] = u; G[:, j] .= g; H[:, j] .= h
        else
            y = Y[:, j]; U[j] = F(y); G[:, j] .= ForwardDiff.gradient(F, y)
        end
    end
    return U, G, H
end

panel_label!(gp, s) = Label(gp[1, 1, TopLeft()], s; fontsize = 24, font = :bold, padding = (0, 12, 6, 0),
                            halign = :right)

# view direction of an Axis3 with the given azimuth and elevation (orthographic)
eyedir(az, el) = [cos(el) * cos(az), cos(el) * sin(az), sin(el)]

function sphere_mesh(c, R; nθ = 60, nφ = 120)
    V = Point3f[]; T = NTuple{3, Int}[]
    for j in 0:nθ, i in 0:nφ
        θ = π * j / nθ; φ = 2π * i / nφ
        push!(V, Point3f((collect(c) .+ R .* [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)])...))
    end
    id(i, j) = 1 + i + (nφ + 1) * j
    for j in 0:nθ-1, i in 0:nφ-1
        push!(T, (id(i, j), id(i + 1, j), id(i + 1, j + 1))); push!(T, (id(i, j), id(i + 1, j + 1), id(i, j + 1)))
    end
    return V, T
end

# element edges of the surface quadrature, keeping the segments that face the viewer
# does the ray from p toward the viewer pass through the ball (c, r)?
function occluded(p, eye, c, r)
    w = p .- c; β = dot(w, eye); q = dot(w, w) - r^2
    disc = β^2 - q
    return disc > 0 && -β + sqrt(disc) > 0
end

# element edges of the surface quadrature: the parts on the front of the sphere and not behind the ball `hide`
function element_edges(sq, eye; m = 12, hide = nothing)
    P = [p for itf in sq.interfaces for p in itf.patches]
    L = Point3f[]
    vis(p) = dot(p .- collect(SPH.b), eye) > 0 && (hide === nothing || !occluded(p, eye, hide...))
    for e in sq.elems
        f = P[e.patch]
        g(u, v) = CutPotentials._elem_point(f, e, u, v)
        for (a, b) in (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)))
            for t in range(0, 1; length = m)
                p = g(a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
                push!(L, vis(p) ? Point3f(p...) : Point3f(NaN))
            end
            push!(L, Point3f(NaN))
        end
    end
    return L
end

# ---------------------------------------------------------------------------------------------------------------
# README figure: (a) the shape, (b) U on a slice, (c) the normal derivative on the sphere

# Lambert-shaded vertex colours (CairoMakie draws one mesh back to front, so solid and translucent parts mix well)
shade(c::RGBAf, n, light) = (l = 0.50 + 0.50 * max(dot(n, light), 0.0);
                             RGBAf(c.r * l, c.g * l, c.b * l, c.alpha))

# the visible half of the circle z = 0.2 on the sphere (the plane of panel b), NaN-separated where it turns away
front_equator(eye) = [dot([cos(t), sin(t), 0.0], eye) > 0 ?
                      Point3f(SPH.b[1] + 1.003SPH.R * cos(t), SPH.b[2] + 1.003SPH.R * sin(t), SPH.b[3]) : Point3f(NaN)
                      for t in range(0, 2π; length = 400)]

function fig_sphere()
    az, el = 0.30π, 0.17π; eye = eyedir(az, el)
    light = normalize(eye .+ [0.0, 0.0, 0.8] .+ 0.4 .* normalize(cross([0.0, 0, 1], eye)))
    fig = Figure(size = (1560, 520), figure_padding = (10, 25, 5, 5))
    q = 1.01SPH.R
    lims = (SPH.b[1] - q, SPH.b[1] + q, SPH.b[2] - q, SPH.b[2] + q, SPH.b[3] - q, SPH.b[3] + q)
    mk3(g, t) = (ax = Axis3(g[1, 1]; aspect = :data, azimuth = az, elevation = el, protrusions = 0, viewmode = :fit,
                            title = t, titlefont = :regular, limits = lims);
                 hidedecorations!(ax); hidespines!(ax); ax)

    # (a) the sphere, seen as translucent, with its elements (graded toward the density) and the core of the density
    # (ρ > 0.8 max ρ, a ball of radius σ √(2 ln 1.25) inside the sphere). The front of the sphere is drawn opaque,
    # each vertex coloured by blending the sphere's colour with what lies behind it along the view ray (the core, by
    # an analytic ray–ball intersection, or the white background): no translucent triangles, so no seams
    ga = fig[1, 1] = GridLayout()
    ax = mk3(ga, "sphere (a = 0.2 inside, 1 outside), its elements,\nand the density (orange: ρ > 0.8 max ρ)")
    rc = G1.σ * sqrt(2log(1.25)); c = collect(G1.c)
    blue = RGBAf(0.62, 0.75, 0.92, 1.0); orange = RGBAf(0.96, 0.55, 0.12, 1.0); α = 0.42
    function seen(p)
        n = normalize(p .- collect(SPH.b)); fg = shade(blue, n, light)
        w = p .- c; β = -dot(w, eye); disc = β^2 - (dot(w, w) - rc^2)
        bg = if disc > 0 && -β - sqrt(disc) > 0
            hpt = p .- (-β - sqrt(disc)) .* eye; shade(orange, normalize(hpt .- c), light)
        else
            RGBAf(1, 1, 1, 1)
        end
        return RGBAf(α * fg.r + (1 - α) * bg.r, α * fg.g + (1 - α) * bg.g, α * fg.b + (1 - α) * bg.b, 1)
    end
    Vs, Ts = sphere_mesh(SPH.b, SPH.R; nθ = 180, nφ = 360)
    Ts = [t for t in Ts if dot(sum(collect(Vs[k]) for k in t) ./ 3 .- collect(SPH.b), eye) > -0.02]
    mesh!(ax, [Vs[k][d] for k in eachindex(Vs), d in 1:3], [t[d] for t in Ts, d in 1:3];
          color = [seen(collect(Float64, v)) for v in Vs], shading = NoShading)
    lines!(ax, element_edges(SQ, eye); color = (:black, 0.6), linewidth = 0.6)
    lines!(ax, front_equator(eye); color = :black, linewidth = 1.4, linestyle = :dash)
    panel_label!(ga, "a")

    # (b) U on the slice z = 0.2 through the centre
    ring = range(0, 2π; length = 200)
    xs = range(-1.3, 2.1; length = 241); ys = range(-1.5, 1.5; length = 213)
    Y = reduce(hcat, [[x, y, SPH.b[3]] for y in ys for x in xs])
    r = evaluate(CutPlan(SQ, Y; a0 = SPH.a0), FIELD)
    gb = fig[1, 2] = GridLayout()
    ax = Axis(gb[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y",
              title = "potential U on the plane z = 0.2\n(white: the sphere and contours of ρ)")
    hm = heatmap!(ax, xs, ys, reshape(r.pot, length(xs), length(ys)); colormap = :viridis)
    contour!(ax, xs, ys, reshape([ρ(G1, Y[:, j]) for j in axes(Y, 2)], length(xs), length(ys));
             levels = 4, color = (:white, 0.55), linewidth = 0.8)
    lines!(ax, [Point2f(SPH.b[1] + SPH.R * cos(t), SPH.b[2] + SPH.R * sin(t)) for t in ring];
           color = :white, linewidth = 1.6, linestyle = :dash)
    Colorbar(gb[1, 2], hm; height = Relative(0.75))
    panel_label!(gb, "b")

    # (c) ∂U/∂n on the sphere: targets exactly on the surface
    nθ, nφ = 120, 240
    θs = range(0, π; length = nθ); φs = range(0, 2π; length = nφ)
    N = [[sin(t) * cos(p), sin(t) * sin(p), cos(t)] for t in θs, p in φs]
    Ys = reduce(hcat, [collect(SPH.b) .+ SPH.R .* n for n in vec(N)])
    rs = evaluate(CutPlan(SQ, Ys; a0 = SPH.a0, grad = true), FIELD)
    dn = reshape([dot(rs.grad[:, j], vec(N)[j]) for j in axes(Ys, 2)], nθ, nφ)
    gc = fig[1, 3] = GridLayout()
    ax = mk3(gc, "normal derivative ∂U/∂n on the sphere\n(outward normal; targets on the surface)")
    lim = maximum(abs, dn)
    cmap = cgrad(:RdBu; rev = true); cc(v) = cmap[(v + lim) / (2lim)]
    colc = [(c = cc(dn[i, j]); l = 0.62 + 0.38 * max(dot(N[i, j], light), 0.0); RGBf(c.r * l, c.g * l, c.b * l))
            for i in 1:nθ, j in 1:nφ]
    surface!(ax, [SPH.b[1] + SPH.R * n[1] for n in N], [SPH.b[2] + SPH.R * n[2] for n in N],
             [SPH.b[3] + SPH.R * n[3] for n in N]; color = colc, shading = NoShading)
    lines!(ax, front_equator(eye); color = :black, linewidth = 1.2, linestyle = :dash)
    Colorbar(gc[1, 2]; colormap = Reverse(:RdBu), limits = (-lim, lim), height = Relative(0.75))
    panel_label!(gc, "c")
    colgap!(fig.layout, 25)
    save(joinpath(OUT, "sphere.png"), fig; px_per_unit = 1.5)
    @info @sprintf("sphere: ∂U/∂n in [%.3g, %.3g]", extrema(dn)...)
    return fig
end

# ---------------------------------------------------------------------------------------------------------------
# errors on the slice: the naive TKM of aρ against CutPotentials

relmap(x::AbstractVector, r::AbstractVector) = abs.(x .- r) ./ maximum(abs, r)
relmap(x::AbstractMatrix, r::AbstractMatrix) =
    [norm(x[:, j] - r[:, j]) for j in axes(r, 2)] ./ maximum(norm(r[:, j]) for j in axes(r, 2))
centres(a, b, n) = (h = (b - a) / n; range(a + h / 2, b - h / 2; length = n))

function fig_errors()
    xs = centres(-1.3, 2.1, 136); ys = centres(-1.5, 1.5, 120)
    Y = reduce(hcat, [[x, y, SPH.b[3]] for y in ys for x in xs])
    t = @elapsed (U, G, H) = reference_fields(Fsphere, Y; hess = true)
    @info @sprintf("errors: reference at %d targets, %.1f s", size(Y, 2), t)
    r = evaluate(CutPlan(SQ, Y; a0 = SPH.a0, hess = true), FIELD)
    naive = field_values(SmoothField(lattice_source(x -> acoef(x) * ρ(G1, x), HS); tol = 1e-12), Y; order = 1)
    M(v) = reshape(log10.(max.(v, 1e-17)), length(xs), length(ys))
    panels = (("naive: TKM of the screened samples aρ\nerror in U", relmap(naive.pot, U)),
              ("CutPotentials\nerror in U", relmap(r.pot, U)),
              ("CutPotentials\nerror in ∇U", relmap(r.grad, G)),
              ("CutPotentials\nerror in ∇∇U", relmap(r.hess, H)))
    fig = Figure(size = (1600, 470))
    ring = range(0, 2π; length = 200)
    local hm
    for (k, (t, e)) in enumerate(panels)
        ax = Axis(fig[1, k]; aspect = DataAspect(), xlabel = "x", ylabel = k == 1 ? "y" : "",
                  title = t * @sprintf(" (max %.0e)", maximum(e)), yticklabelsvisible = k == 1)
        hm = heatmap!(ax, xs, ys, M(e); colormap = :viridis, colorrange = (-15, -1))
        lines!(ax, [Point2f(SPH.b[1] + SPH.R * cos(t), SPH.b[2] + SPH.R * sin(t)) for t in ring];
               color = :white, linewidth = 1.2, linestyle = :dash)
    end
    Colorbar(fig[1, 5], hm; label = "log₁₀ (error / max |reference|)", height = Relative(0.75))
    save(joinpath(OUT, "errors.png"), fig; px_per_unit = 1.5)
    @info @sprintf("errors: naive %.1e, U %.1e, ∇U %.1e, ∇∇U %.1e", (maximum(p[2]) for p in panels)...)
    return fig
end

# ---------------------------------------------------------------------------------------------------------------
# when the TKM fails: the spectrum of ρ and of aρ, and the error against the sample spacing

# Fourier transform ∫ χ_ball ρ e^{-iκ e·x} dx along e = (c - b)/|c - b|: every plane e·x = s cuts the ball in a disk
# centred on the axis through c, where the plane integral of ρ is closed-form
function ball_spectrum(κs)
    c = collect(G1.c); b = collect(SPH.b); e = normalize(c .- b); σ = G1.σ; R = SPH.R
    sc, sb = dot(e, c), dot(e, b)
    f(s) = (r2 = R^2 - (s - sb)^2; exp(-(s - sc)^2 / (2σ^2)) / sqrt(2π * σ^2) * (1 - exp(-r2 / (2σ^2))))
    x, w = _GL16; np = 400
    edges = range(sb - R, sb + R; length = np + 1)
    S = Float64[]; W = Float64[]
    for p in 1:np
        a, β = edges[p], edges[p + 1]
        append!(S, (a + β) / 2 .+ (β - a) / 2 .* x); append!(W, (β - a) / 2 .* w)
    end
    F = f.(S)
    return [sum(W .* F .* cis.(-κ .* S)) for κ in κs], sc
end

function fig_resolution()
    σ = G1.σ
    κs = range(0, 120; length = 1201)
    ball, sc = ball_spectrum(κs)
    ρhat = exp.(-σ^2 .* κs .^ 2 ./ 2) .* cis.(-κs .* sc)
    aρhat = SPH.a0 .* ρhat .+ (SPH.ain - SPH.a0) .* ball
    fig = Figure(size = (1450, 520))
    ga = fig[1, 1] = GridLayout()
    ax = Axis(ga[1, 1]; yscale = log10, xlabel = "wavenumber κ (along the axis from the sphere's centre to the density's)",
              ylabel = "|ρ̂(κ)| / ρ̂(0)", title = "spectrum of the density before and after screening",
              limits = ((0, 120), (1e-17, 2)))
    lines!(ax, κs, max.(abs.(ρhat), 1e-17); color = :royalblue, linewidth = 2.5, label = "ρ (smooth)")
    lines!(ax, κs, max.(abs.(aρhat), 1e-17); color = :crimson, linewidth = 2.5, label = "aρ (screened)")
    lines!(ax, κs[2:end], 0.35 ./ κs[2:end] .^ 2; color = (:crimson, 0.6), linestyle = :dash, linewidth = 1.5,
           label = "κ⁻²")
    for (h, lab) in ((0.5, "h = σ"), (0.25, "h = σ/2"), (0.1, "h = σ/5"))
        vlines!(ax, [π / h]; color = :gray40, linestyle = :dot, linewidth = 1.5)
        text!(ax, π / h, 3e-17; text = " π/h, " * lab, rotation = π / 2, align = (:left, :top), fontsize = 14,
              color = :gray30)
    end
    axislegend(ax; position = :rt, framevisible = false)
    panel_label!(ga, "a")

    # error against the spacing: the plan (surface quadrature, near corrections) is reused for every h
    Y = let k = 0, pts = Vector{Float64}[]
        while length(pts) < 300
            k += 1
            p = collect(SPH.b) .+ 2.0 .* [sin(1.7k) * cos(2.3k), sin(1.7k) * sin(2.3k), cos(1.7k)] .* cbrt(mod(0.618k, 1))
            abs(norm(p .- SPH.b) - SPH.R) > 1e-2 && push!(pts, p)
        end
        reduce(hcat, pts)
    end
    U, _, _ = reference_fields(Fsphere, Y)
    E0 = field_values(ExactField(G1), Y).pot
    plan = CutPlan(SQ, Y; a0 = SPH.a0, grad = false)
    hs = [0.06, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    eT = Float64[]; eC = Float64[]; eN = Float64[]
    for h in hs
        src = lattice_source(x -> ρ(G1, x), h)
        fld = SmoothField(src; tol = 1e-12)
        push!(eT, maximum(abs, field_values(fld, Y; order = 1).pot .- E0) / maximum(abs, E0))
        push!(eC, maximum(abs, evaluate(plan, fld).pot .- U) / maximum(abs, U))
        nv = field_values(SmoothField(lattice_source(x -> acoef(x) * ρ(G1, x), h); tol = 1e-12), Y; order = 1).pot
        push!(eN, maximum(abs, nv .- U) / maximum(abs, U))
        @info @sprintf("resolution: h = %.3f  TKM(ρ) %.1e  CutPotentials %.1e  naive %.1e", h, eT[end], eC[end], eN[end])
    end
    gb = fig[1, 2] = GridLayout()
    ax = Axis(gb[1, 1]; xscale = log10, yscale = log10, xlabel = "sample spacing h  (σ = 0.5)",
              ylabel = "max error in U / max |U|", title = "accuracy against the sample spacing",
              xticks = ([0.06, 0.1, 0.2, 0.3, 0.5], ["0.06", "0.1", "0.2", "0.3", "0.5"]),
              limits = ((0.055, 0.55), (1e-16, 1)))
    hh = range(0.055, 0.55; length = 200)
    lines!(ax, hh, exp.(-σ^2 .* (π ./ hh) .^ 2 ./ 2); color = (:royalblue, 0.5), linestyle = :dash, linewidth = 1.5,
           label = "|ρ̂(π/h)| = exp(−σ²π²/2h²)")
    scatterlines!(ax, hs, eT; color = :royalblue, marker = :circle, label = "TKM of ρ (the smooth field φ₀)")
    scatterlines!(ax, hs, eC; color = :black, marker = :diamond, label = "CutPotentials: U")
    scatterlines!(ax, hs, eN; color = :crimson, marker = :utriangle, label = "naive: TKM of aρ")
    Legend(gb[1, 2], ax; framevisible = false)
    panel_label!(gb, "b")
    save(joinpath(OUT, "resolution.png"), fig; px_per_unit = 1.5)
    return fig
end

# ---------------------------------------------------------------------------------------------------------------
# error against the distance to a face and an edge of a box

function fig_distance()
    xe = 0.8; a0 = 1.0; box = [((-2.0, -2.0, -2.0), (xe, 2.0, 0.0))]; a = [0.25]
    sq = SurfaceQuadrature(box_interfaces(box, a, a0); support = (G1.c, G1.c), L0 = G1.σ)
    F(z) = a0 * φ0(G1, z) + (a[1] - a0) * Nbox(G1, z, box[1]...)
    ds = 10.0 .^ range(-14, -1; length = 27)
    exact = ExactField(G1)
    fig = Figure(size = (1400, 480))
    ticks = (10.0 .^ (-14:2:-2), ["10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²"])
    yticks = (10.0 .^ (-16:2:-10), ["10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰"])
    for (k, (t, base, dir)) in enumerate((("target approaching the face z = 0 at (0.5, 0.3, 0)", [0.5, 0.3, 0.0], [0, 0, 1.0]),
                                          ("target approaching the edge x = 0.8, z = 0 (diagonally)", [xe, 0.05, 0.0],
                                           normalize([1.0, 0, 1.0]))))
        g = fig[1, k] = GridLayout()
        ax = Axis(g[1, 1]; title = t, xscale = log10, yscale = log10, xlabel = "distance d",
                  ylabel = k == 1 ? "error / max |reference|" : "", xticks = ticks, yticks,
                  limits = (nothing, (1e-16, 1e-10)))
        Y = reduce(hcat, [base .+ d .* dir for d in ds])
        U, G, _ = reference_fields(F, Y)
        sU = maximum(abs, U); sG = maximum(norm, eachcol(G))
        for (fld, sty, fname) in ((FIELD, :solid, "TKM field"), (exact, :dash, "exact smooth field"))
            r = evaluate(CutPlan(sq, Y; a0, grad = true), fld)
            lines!(ax, ds, max.(abs.(r.pot .- U) ./ sU, 1e-17); color = :royalblue, linestyle = sty, linewidth = 2,
                   label = "U, $fname")
            lines!(ax, ds, max.([norm(r.grad[:, j] .- G[:, j]) for j in axes(G, 2)] ./ sG, 1e-17); color = :crimson,
                   linestyle = sty, linewidth = 2, label = "∇U, $fname")
        end
        k == 2 && Legend(fig[1, 3], ax; framevisible = false)
        panel_label!(g, k == 1 ? "a" : "b")
    end
    save(joinpath(OUT, "distance.png"), fig; px_per_unit = 1.5)
    return fig
end

# ---------------------------------------------------------------------------------------------------------------
# the gathering window: its derivatives at the end of the support, and the error against its width

# move the coordinates of y so that each lies δ (in fine-grid units) past a stencil transition of field f: the last
# point of the stencil is then δ from the end of the window's support
function transition_target(f, y, δ)
    g = f.geom; ns = f.win.ns; y2 = copy(y)
    for d in 1:3
        nf = f.nf[d]; x = g.dks[d] * (y[d] - g.center[d]) / (2π) + 0.5
        j = floor(x); u = (x - j) * nf
        u2 = floor(u - ns / 2) + ns / 2 + δ
        y2[d] = g.center[d] + 2π * ((u2 / nf + j) - 0.5) / g.dks[d]
    end
    return y2
end

function fig_window()
    fig = Figure(size = (1450, 520))
    # (a) |w|, |w′|, |w″| against the distance 1 - |z| to the end of the support, ns = 18, σ = 1.5
    ga = fig[1, 1] = GridLayout()
    ax = Axis(ga[1, 1]; xscale = log10, yscale = log10, xlabel = "distance 1 − |z| to the end of the support",
              ylabel = "|w|, |w′|, |w″|  (grid units)", title = "the ES window near its support end (ns = 18, σ = 1.5)",
              limits = ((1e-12, 0.5), (1e-40, 1e4)), xticks = (10.0 .^ (-12:3:-3), ["10⁻¹²", "10⁻⁹", "10⁻⁶", "10⁻³"]),
              yticks = (10.0 .^ (-40:10:0), ["10⁻⁴⁰", "10⁻³⁰", "10⁻²⁰", "10⁻¹⁰", "1"]))
    sz = 10.0 .^ range(-12, log10(0.5); length = 400)
    for (m, sty, lab) in ((0, :dash, "untapered ES"), (3, :solid, "ES · (1 − z²)³"))
        win = CutPotentials._Window(18, 0.97π * 18 * (1 - 1 / 3), m)
        W = [CutPotentials._window_eval(win, (1 - s) * 9.0) for s in sz]
        for (k, (col, d)) in enumerate(((:black, "w"), (:royalblue, "w′"), (:crimson, "w″")))
            lines!(ax, sz, max.(abs.(getindex.(W, k)), 1e-300); color = col, linestyle = sty, linewidth = 2,
                   label = "$d, $lab")
        end
    end
    axislegend(ax; position = :rb, framevisible = false, nbanks = 2, labelsize = 14)
    panel_label!(ga, "a")

    # (b) error of the gathered φ0, ∇φ0, ∇∇φ0 against the width, at random targets and at stencil transitions
    Y0 = reduce(hcat, [collect(G1.c) .+ [1.3cos(k), 1.1sin(k), 0.7cos(3k)] .* (k / 20) for k in 1:60])
    E = field_values(ExactField(G1), Y0)
    rel(x, r) = maximum(abs, x .- r) / maximum(abs, r)
    nss = 8:24
    res = Dict{Tuple{Int, Symbol}, Vector{NTuple{3, Float64}}}()
    for m in (0, 3), ns in nss
        f = SmoothField(SRC; tol = 1e-14, window_ns = ns, window_taper = m)
        V = field_values(f, Y0)
        Yt = reduce(hcat, [transition_target(f, Y0[:, j], 1e-9) for j in axes(Y0, 2)])
        Et = field_values(ExactField(G1), Yt); Vt = field_values(f, Yt)
        push!(get!(res, (m, :random), NTuple{3, Float64}[]), (rel(V.pot, E.pot), rel(V.grad, E.grad), rel(V.hess, E.hess)))
        push!(get!(res, (m, :transition), NTuple{3, Float64}[]),
              (rel(Vt.pot, Et.pot), rel(Vt.grad, Et.grad), rel(Vt.hess, Et.hess)))
        @info @sprintf("window: m = %d ns = %2d  random %.1e %.1e %.1e   transitions %.1e %.1e %.1e", m, ns,
                       res[(m, :random)][end]..., res[(m, :transition)][end]...)
    end
    gb = fig[1, 2] = GridLayout()
    ax = Axis(gb[1, 1]; yscale = log10, xlabel = "window width ns (points per dimension)",
              ylabel = "max error / max |exact|", title = "error of the gathered fields (σ = 1.5)",
              limits = ((7.5, 24.5), (1e-14, 1e2)), xticks = 8:2:24,
              yticks = (10.0 .^ (-14:4:2), ["10⁻¹⁴", "10⁻¹⁰", "10⁻⁶", "10⁻²", "10²"]))
    for (ns, lab) in ((16, "1e-8"), (18, "1e-10"), (20, "1e-12"))
        vlines!(ax, [ns]; color = :gray70, linestyle = :dot)
        text!(ax, ns - 0.15, 2e-14; text = "default at tol " * lab, rotation = π / 2, align = (:left, :bottom),
              fontsize = 13, color = :gray40)
    end
    for (k, col, q) in ((1, :black, "φ₀"), (2, :royalblue, "∇φ₀"), (3, :crimson, "∇∇φ₀"))
        lines!(ax, nss, max.(getindex.(res[(3, :random)], k), 1e-17); color = col, linewidth = 2,
               label = "$q, tapered")
        lines!(ax, nss, max.(getindex.(res[(0, :transition)], k), 1e-17); color = col, linestyle = :dash,
               linewidth = 2, label = "$q, untapered, at stencil transitions")
    end
    scatter!(ax, nss, max.(getindex.(res[(3, :transition)], 3), 1e-17); color = :crimson, marker = :circle,
             markersize = 7, label = "∇∇φ₀, tapered, at stencil transitions")
    Legend(gb[1, 2], ax; framevisible = false, labelsize = 14)
    panel_label!(gb, "b")
    save(joinpath(OUT, "window.png"), fig; px_per_unit = 1.5)
    return fig
end

which = isempty(ARGS) ? ["sphere", "errors", "resolution", "distance", "window"] : ARGS
"sphere" in which && fig_sphere()
"errors" in which && fig_errors()
"resolution" in which && fig_resolution()
"distance" in which && fig_distance()
"window" in which && fig_window()
