# Figures for docs/src/examples.md, written to docs/src/assets/.
#
# Not run by the docs build (it takes a few minutes and needs a plotting package). Set up an environment with
# CutPotentials, SCTL.jl, CairoMakie, ForwardDiff and SpecialFunctions (see docs/figures/Project.toml), e.g.
#
#     julia --project=docs/figures -e 'using Pkg; Pkg.develop([PackageSpec(path = "."),
#         PackageSpec(url = "https://github.com/xuanzhaogao/SCTL.jl")]); Pkg.build("SCTL"); Pkg.instantiate()'
#     OMP_NUM_THREADS=16 julia -t 16 --project=docs/figures docs/figures/make_figures.jl
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

# the density of the tests: a Gaussian sampled on a skewed, anisotropic lattice
const G1 = Gaussian((0.6, 0.05, 0.1), 0.5)
const BASIS = [0.1 0.02 0.0; 0.0 0.11 0.01; 0.0 0.0 0.09]
function lattice_source(f)
    R = 7 * G1.σ
    o = collect(G1.c) .- R .- [0.013, -0.021, 0.034] .- BASIS * [3, 3, 3]
    dims = Tuple(ceil.(Int, 2R ./ [0.08, 0.09, 0.08]) .+ 10)
    return VolumeSource(f, o, BASIS, dims)
end
const SRC = lattice_source(x -> ρ(G1, x))
const FIELD = SmoothField(SRC; tol = 1e-12)

# ---------------------------------------------------------------------------------------------------------------
# geometry: the graded surface elements, coloured by δa, and the density's 2σ ball

function element_mesh(sq::SurfaceQuadrature; m = 4)
    P = [p for itf in sq.interfaces for p in itf.patches]
    pts = Point3f[]; faces = NTuple{3, Int}[]; col = Float64[]
    lines = Point3f[]
    for (e, δa) in zip(sq.elems, sq.elem_δa)
        f = P[e.patch]
        g(u, v) = Point3f(CutPotentials._elem_point(f, e, u, v)...)
        base = length(pts)
        for j in 0:m, i in 0:m
            push!(pts, g(i / m, j / m)); push!(col, δa)
        end
        id(i, j) = base + 1 + i + (m + 1) * j
        for j in 0:m-1, i in 0:m-1
            push!(faces, (id(i, j), id(i + 1, j), id(i + 1, j + 1)))
            push!(faces, (id(i, j), id(i + 1, j + 1), id(i, j + 1)))
        end
        for t in range(0, 1; length = 9); push!(lines, g(t, 0.0)); end
        for t in range(0, 1; length = 9); push!(lines, g(1.0, t)); end
        for t in range(1, 0; length = 9); push!(lines, g(t, 1.0)); end
        for t in range(1, 0; length = 9); push!(lines, g(0.0, t)); end
        push!(lines, Point3f(NaN))
    end
    return pts, faces, col, lines
end

function draw_geometry!(ax, sq; crange = nothing)
    pts, faces, col, lines = element_mesh(sq)
    cr = crange === nothing ? (-maximum(abs, col), maximum(abs, col)) : crange
    V = [pts[k][d] for k in eachindex(pts), d in 1:3]; T = [f[d] for f in faces, d in 1:3]
    # translucent faces, so that interior interfaces and the grading toward the density show
    mesh!(ax, V, T; color = col, colormap = (:RdBu, 0.35), colorrange = cr, shading = NoShading, transparency = true)
    lines!(ax, lines; color = (:black, 0.55), linewidth = 0.5)
    # the density's 2σ ball as a wireframe
    c = collect(G1.c); r = 2G1.σ; ring = range(0, 2π; length = 120)
    for t in range(0, π; length = 7)[2:end-1]
        lines!(ax, [Point3f((c .+ r .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)])...) for p in ring];
               color = :darkorange, linewidth = 1.6)
    end
    for p in range(0, π; length = 7)[1:end-1]
        lines!(ax, [Point3f((c .+ r .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)])...) for t in ring];
               color = :darkorange, linewidth = 1.6)
    end
    return cr
end

const XE = 0.8
const BOXES4 = [((-2.0, -2.0, -2.0), (XE, 2.0, 0.0)), ((XE, -2.0, -2.0), (3.0, 2.0, 0.0)),
                ((-2.0, -2.0, 0.0), (XE, 2.0, 2.0)), ((XE, -2.0, 0.0), (3.0, 2.0, 2.0))]
const A4 = [1 / 4, 1 / 12, 1 / 2, 1 / 3]
const SPH = (b = (0.1, 0.0, 0.2), R = 0.9, ain = 0.2)
const ELL = let θ = 0.4
    Rz = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]; Rx = [1 0 0; 0 cos(0.3) -sin(0.3); 0 sin(0.3) cos(0.3)]
    (c = (0.4, 0.1, 0.0), ax = (1.0, 0.6, 0.45), R = Rz * Rx, a1 = 0.3)
end
const SIB = (box = ((-1.0, -1.2, -1.0), (1.6, 1.2, 1.2)), abox = 0.5, b = (0.5, 0.05, 0.1), R = 0.6, asph = 0.1)

sq_for(itf) = SurfaceQuadrature(itf; support = (G1.c, G1.c), L0 = G1.σ)

function fig_geometry()
    itf4 = box_interfaces(BOXES4, A4, 1.0)
    itfs = [sphere_interface(SPH.b, SPH.R, SPH.ain - 1.0)]
    itfe = [ellipsoid_interface(ELL.c, ELL.ax, ELL.a1 - 1.0; R = ELL.R)]
    itfb = vcat(box_interfaces([SIB.box], [SIB.abox], 1.0), [sphere_interface(SIB.b, SIB.R, SIB.asph - SIB.abox)])
    cases = [("Four boxes along an edge", itf4), ("Sphere", itfs), ("Rotated ellipsoid", itfe),
             ("Sphere inside a box", itfb)]
    fig = Figure(size = (1500, 1250), fontsize = 18)
    cr = (-1.0, 1.0)
    for (k, (title, itf)) in enumerate(cases)
        sq = sq_for(itf)
        i, j = fldmod1(k, 2)
        ax = Axis3(fig[i, j]; title = "$title\n$(length(sq.elems)) elements, $(size(sq.X, 2)) nodes",
                   aspect = :data, azimuth = 0.35π, elevation = 0.12π, protrusions = 30)
        draw_geometry!(ax, sq; crange = cr)
    end
    Colorbar(fig[:, 3]; colormap = :RdBu, limits = cr, label = "δa = a_A − a_B (jump across the interface)")
    Label(fig[0, :], "Interfaces and their graded Gauss–Legendre elements (orange: the density's 2σ ball)";
          fontsize = 22)
    save(joinpath(OUT, "geometry.png"), fig; px_per_unit = 1)
    return fig
end

# ---------------------------------------------------------------------------------------------------------------
# slices: field and log10 errors against the heat-kernel references

function slice_targets(xs, ys, embed)
    Y = zeros(3, length(xs) * length(ys)); k = 0
    for y in ys, x in xs
        k += 1; Y[:, k] .= embed(x, y)
    end
    return Y
end

function reference_fields(F, Y)
    n = size(Y, 2); U = zeros(n); G = zeros(3, n); H = zeros(6, n)
    Threads.@threads for j in 1:n
        u, g, h = vgh(F, Y[:, j]); U[j] = u; G[:, j] .= g; H[:, j] .= h
    end
    return U, G, H
end

# pointwise error over the slice, relative to the largest reference value on it (as in the tests)
relmap(x::AbstractVector, r::AbstractVector) = abs.(x .- r) ./ maximum(abs, r)
relmap(x::AbstractMatrix, r::AbstractMatrix) =
    [norm(x[:, j] - r[:, j]) for j in axes(r, 2)] ./ maximum(norm(r[:, j]) for j in axes(r, 2))

function slice_figure(name, title, itf, F, acoef, xs, ys, embed, labels, outlines!; a0 = 1.0)
    Y = slice_targets(xs, ys, embed)
    nx, ny = length(xs), length(ys)
    t = @elapsed (U, G, H) = reference_fields(F, Y)
    @info @sprintf("%s: reference at %d targets, %.1f s", name, size(Y, 2), t)
    sq = sq_for(itf)
    t = @elapsed r = evaluate(CutPlan(sq, Y; a0, hess = true), FIELD)
    @info @sprintf("%s: CutPotentials, %.1f s", name, t)
    # naive: the smooth-field evaluator applied directly to the screened samples a ρ
    srcn = lattice_source(x -> acoef(x) * ρ(G1, x))
    naive = field_values(SmoothField(srcn; tol = 1e-12), Y; order = 1).pot
    av = [acoef(Y[:, j]) for j in axes(Y, 2)]
    M(v) = reshape(v, nx, ny)
    e_naive = relmap(naive, U); e_pot = relmap(r.pot, U); e_grad = relmap(r.grad, G); e_hess = relmap(r.hess, H)
    @info @sprintf("%s: max errors naive %.1e  pot %.1e  grad %.1e  hess %.1e", name,
                   maximum(e_naive), maximum(e_pot), maximum(e_grad), maximum(e_hess))
    fig = Figure(size = (1700, 830), fontsize = 17)
    xl, yl = labels
    mkax(i, j, t) = Axis(fig[i, j]; title = t, xlabel = xl, ylabel = yl, aspect = DataAspect())
    ax = mkax(1, 1, "coefficient a")
    hm = heatmap!(ax, xs, ys, M(av); colormap = :viridis); outlines!(ax); Colorbar(fig[1, 2], hm)
    ax = mkax(1, 3, "U")
    hm = heatmap!(ax, xs, ys, M(r.pot); colormap = :magma); outlines!(ax); Colorbar(fig[1, 4], hm)
    ax = mkax(1, 5, "|∇U|")
    hm = heatmap!(ax, xs, ys, M([norm(r.grad[:, j]) for j in axes(Y, 2)]); colormap = :magma); outlines!(ax)
    Colorbar(fig[1, 6], hm)
    ax = mkax(1, 7, "tr ∇∇U = −aρ")
    hm = heatmap!(ax, xs, ys, M(vec(sum(r.hess[1:3, :]; dims = 1))); colormap = :magma); outlines!(ax)
    Colorbar(fig[1, 8], hm)
    lim = (-16, 0)
    for (k, (t, e)) in enumerate((("naive: TKM of aρ, U error", e_naive), ("CutPotentials: U error", e_pot),
                                  ("CutPotentials: ∇U error", e_grad), ("CutPotentials: ∇∇U error", e_hess)))
        ax = mkax(2, 2k - 1, t * @sprintf("\nmax %.0e", maximum(e)))
        hm = heatmap!(ax, xs, ys, M(log10.(max.(e, 1e-17))); colormap = :turbo, colorrange = lim); outlines!(ax)
        k == 4 && Colorbar(fig[2, 8], hm; label = "log₁₀ (error / max |reference|)")
    end
    Label(fig[0, :], title; fontsize = 21)
    save(joinpath(OUT, "$name.png"), fig; px_per_unit = 1)
    return fig
end

# cell centres: no target lies exactly on a face of the boxes below
centres(a, b, n) = (h = (b - a) / n; range(a + h / 2, b - h / 2; length = n))

function fig_boxes()
    a0 = 1.0
    itf = box_interfaces(BOXES4, A4, a0)
    F(z) = a0 * φ0(G1, z) + sum((A4[j] - a0) * Nbox(G1, z, BOXES4[j]...) for j in eachindex(BOXES4))
    acoef(x) = (for (j, b) in enumerate(BOXES4)
                    all(b[1][d] < x[d] < b[2][d] for d in 1:3) && return A4[j]
                end; a0)
    xs = centres(-1.0, 2.4, 136); zs = centres(-1.5, 1.7, 128)
    outl!(ax) = (vlines!(ax, [XE]; color = :white, linewidth = 1, linestyle = :dash);
                 hlines!(ax, [0.0]; color = :white, linewidth = 1, linestyle = :dash))
    slice_figure("boxes_slice", "Four boxes along an edge (a = 1/4, 1/12, 1/2, 1/3): the slice y = 0.05 through the density",
                 itf, F, acoef, xs, zs, (x, z) -> [x, 0.05, z], ("x", "z"), outl!)
end

function fig_sphere()
    a0 = 1.0
    itf = [sphere_interface(SPH.b, SPH.R, SPH.ain - a0)]
    F(z) = a0 * φ0(G1, z) + (SPH.ain - a0) * Nball(G1, z, SPH.b, SPH.R)
    acoef(x) = norm(x .- SPH.b) < SPH.R ? SPH.ain : a0
    xs = centres(-1.2, 2.0, 128); ys = centres(-1.5, 1.5, 120)
    outl!(ax) = lines!(ax, [Point2f(SPH.b[1] + SPH.R * cos(t), SPH.b[2] + SPH.R * sin(t)) for t in range(0, 2π; length = 200)];
                       color = :white, linewidth = 1, linestyle = :dash)
    slice_figure("sphere_slice", "Sphere (a = 0.2 inside, 1 outside): the slice z = 0.2 through its centre",
                 itf, F, acoef, xs, ys, (x, y) -> [x, y, SPH.b[3]], ("x", "y"), outl!)
end

# ---------------------------------------------------------------------------------------------------------------
# error against the distance to an interface: no snapping, no loss as targets approach a face or an edge

function fig_distance()
    a0 = 1.0; box = [((-2.0, -2.0, -2.0), (XE, 2.0, 0.0))]; a = [0.25]
    itf = box_interfaces(box, a, a0); sq = sq_for(itf)
    F(z) = a0 * φ0(G1, z) + (a[1] - a0) * Nbox(G1, z, box[1]...)
    ds = 10.0 .^ range(-14, -1; length = 27)
    exact = ExactField(G1)
    fig = Figure(size = (1400, 520), fontsize = 17)
    ticks = (10.0 .^ (-14:2:-2), ["10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²"])
    yticks = (10.0 .^ (-16:2:-8), ["10⁻¹⁶", "10⁻¹⁴", "10⁻¹²", "10⁻¹⁰", "10⁻⁸"])
    for (k, (t, base, dir)) in enumerate((("approaching the face z = 0 at (0.6, 0.05, 0)", [0.6, 0.05, 0.0], [0, 0, 1.0]),
                                          ("approaching the edge x = 0.8, z = 0 (diagonally, at y = 0.05)",
                                           [XE, 0.05, 0.0], normalize([1.0, 0, 1.0]))))
        ax = Axis(fig[1, k]; title = t, xscale = log10, yscale = log10, xlabel = "distance d from the interface",
                  ylabel = "error / max |reference|", limits = (nothing, (1e-17, 1e-8)), xticks = ticks, yticks)
        for (s, mk, side) in ((1, :circle, "outside"), (-1, :xcross, "inside"))
            Y = reduce(hcat, [base .+ (s * d) .* dir for d in ds])
            U, G, _ = reference_fields(F, Y)
            sU = maximum(abs, U); sG = maximum(norm, eachcol(G))
            for (fld, sty, fname) in ((FIELD, :solid, "TKM field"), (exact, :dot, "exact smooth field"))
                r = evaluate(CutPlan(sq, Y; a0, grad = true), fld)
                eU = max.(abs.(r.pot .- U) ./ sU, 1e-17)
                eG = max.([norm(r.grad[:, j] .- G[:, j]) for j in axes(G, 2)] ./ sG, 1e-17)
                scatterlines!(ax, ds, eU; color = :royalblue, linestyle = sty, marker = mk, markersize = 8,
                              label = "U, $fname, $side")
                scatterlines!(ax, ds, eG; color = :crimson, linestyle = sty, marker = mk, markersize = 8,
                              label = "∇U, $fname, $side")
            end
        end
        k == 2 && Legend(fig[1, 3], ax; framevisible = false, merge = true)
    end
    Label(fig[0, :], "One box (a = 1/4 inside, 1 outside): targets at distance d from a face and from an edge, both sides";
          fontsize = 20)
    save(joinpath(OUT, "distance.png"), fig; px_per_unit = 1)
    return fig
end

which = isempty(ARGS) ? ["geometry", "boxes", "sphere", "distance"] : ARGS
"geometry" in which && fig_geometry()
"distance" in which && fig_distance()
"sphere" in which && fig_sphere()
"boxes" in which && fig_boxes()
