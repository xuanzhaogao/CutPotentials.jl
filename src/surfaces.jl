# Interfaces between regions of constant coefficient: parametrized patches, constructors, graded discretization.

"""
    Interface(patches, δa; name = "")

A piece of the interfaces between regions of constant coefficient, with the jump `δa = a_A - a_B` across it.
`patches` is a vector of functions `X(u, v)` (`(u, v) ∈ [0, 1]²`, returning a point as 3 numbers), smooth on the
closed parameter square, whose normal `∂X/∂u × ∂X/∂v` points **from side A to side B**. Each patch is one or more
SCTL Gauss-Legendre elements after refinement ([`SurfaceQuadrature`](@ref)).

The interfaces given to [`SurfaceQuadrature`](@ref) must together form the boundaries of the regions: for each
region `Ω_j` (coefficient `a_j`, background `a_0` outside all of them), its closed boundary `∂Ω_j`, oriented outward, is
covered by pieces whose δa (with orientation) adds up to the region-by-region sum
`Σ_j (a_j - a_0) [∂Ω_j]`. A face shared by two regions is given once, with `δa = a_A - a_B`; faces with
`δa = 0` can be dropped. Constructors: [`box_interfaces`](@ref), [`sphere_interface`](@ref),
[`ellipsoid_interface`](@ref).
"""
struct Interface
    patches::Vector{Any}
    δa::Float64
    name::String
end
Interface(patches::AbstractVector, δa::Real; name::AbstractString = "") = Interface(Vector{Any}(patches), Float64(δa), String(name))
Interface(patch::Function, δa::Real; name::AbstractString = "") = Interface(Any[patch], Float64(δa), String(name))

Base.show(io::IO, i::Interface) = print(io, "Interface(", isempty(i.name) ? "" : "\"$(i.name)\", ",
                                        length(i.patches), " patch(es), δa = ", i.δa, ")")

# ---- boxes ----

# tangent axes (t1, t2) with e_t1 × e_t2 = +e_k
_tangents(k) = k == 1 ? (2, 3) : k == 2 ? (3, 1) : (1, 2)

function _merge_coords(v::Vector{Float64}; rtol = 1e-12)
    v = sort(v); out = [v[1]]
    for x in v[2:end]
        abs(x - out[end]) <= rtol * max(1.0, abs(x)) || push!(out, x)
    end
    return out
end

function _snap(x::Float64, coords::Vector{Float64}; rtol = 1e-12)
    for c in coords
        abs(x - c) <= rtol * max(1.0, abs(c)) && return c
    end
    return x
end

"""
    rectangle_patch(k, c, lo, hi) -> patch

The axis-aligned rectangle `x[k] = c`, `lo[i] ≤ x[t_i] ≤ hi[i]` for the tangent axes `(t1, t2)` with
`e_t1 × e_t2 = +e_k` (k = 1: (2, 3); k = 2: (3, 1); k = 3: (1, 2)); its normal is `+e_k`.
"""
function rectangle_patch(k::Int, c::Real, lo, hi)
    t1, t2 = _tangents(k)
    c = Float64(c); l1, l2 = Float64(lo[1]), Float64(lo[2]); h1, h2 = Float64(hi[1]), Float64(hi[2])
    return function (u, v)
        x = zeros(3); x[k] = c; x[t1] = l1 + u * (h1 - l1); x[t2] = l2 + v * (h2 - l2)
        x
    end
end

"""
    box_interfaces(boxes, a, a0; rtol = 1e-12) -> Vector{Interface}

The interfaces of a union of axis-aligned boxes with disjoint interiors: box `j` is `(lo, hi)` (two 3-tuples) with
coefficient `a[j]`, the background has `a0`. Face coordinates of all boxes are merged within `rtol` (relative), so
neighbouring boxes share faces exactly; the faces are cut into the cells of the grid of all face coordinates, and
each cell where the coefficient jumps becomes one rectangle with normal `+e_k` (from the `x[k] < c` side A to the
`x[k] > c` side B) and `δa = a_A - a_B`. Rectangles with equal coefficients on both sides are dropped.
"""
function box_interfaces(boxes, a::AbstractVector{<:Real}, a0::Real; rtol::Real = 1e-12)
    length(boxes) == length(a) || throw(DimensionMismatch("need one coefficient per box"))
    X = [_merge_coords(Float64[b[s][d] for b in boxes for s in 1:2]; rtol) for d in 1:3]
    B = [(ntuple(d -> _snap(Float64(b[1][d]), X[d]; rtol), 3), ntuple(d -> _snap(Float64(b[2][d]), X[d]; rtol), 3)) for b in boxes]
    for (lo, hi) in B
        all(lo[d] < hi[d] for d in 1:3) || throw(ArgumentError("box with lo ≥ hi: $lo, $hi"))
    end
    coef(x) = begin
        for (j, (lo, hi)) in enumerate(B)
            all(lo[d] < x[d] < hi[d] for d in 1:3) && return Float64(a[j])
        end
        Float64(a0)
    end
    # disjoint interiors: the cell midpoints are in at most one box
    for i in 1:length(X[1])-1, j in 1:length(X[2])-1, k in 1:length(X[3])-1
        m = ((X[1][i] + X[1][i+1]) / 2, (X[2][j] + X[2][j+1]) / 2, (X[3][k] + X[3][k+1]) / 2)
        count(b -> all(b[1][d] < m[d] < b[2][d] for d in 1:3), B) <= 1 ||
            throw(ArgumentError("boxes overlap (at $m)"))
    end
    out = Interface[]
    for k in 1:3
        t1, t2 = _tangents(k)
        off = 1e-3 * minimum(diff(X[k]))           # below the smallest gap between face coordinates
        for c in X[k], i in 1:length(X[t1])-1, j in 1:length(X[t2])-1
            m = zeros(3); m[t1] = (X[t1][i] + X[t1][i+1]) / 2; m[t2] = (X[t2][j] + X[t2][j+1]) / 2
            m[k] = c - off; aA = coef(m)
            m[k] = c + off; aB = coef(m)
            aA == aB && continue
            push!(out, Interface(Any[rectangle_patch(k, c, (X[t1][i], X[t2][j]), (X[t1][i+1], X[t2][j+1]))], aA - aB;
                                 name = "x$k = $c"))
        end
    end
    return out
end

# ---- spheres and ellipsoids (cubed sphere, equiangular) ----

# the 6 cube faces: x[k] = s, (u, v) along (t1, t2) with e_t1 × e_t2 = s e_k (outward)
function _cube_face_point(face::Int, ξ::Float64, η::Float64)
    k = (face + 1) ÷ 2; s = isodd(face) ? 1.0 : -1.0
    t1, t2 = _tangents(k)
    x = zeros(3); x[k] = s
    if s > 0
        x[t1] = ξ; x[t2] = η
    else
        x[t1] = η; x[t2] = ξ
    end
    return x
end

"""
    ellipsoid_interface(center, semiaxes, δa; R = I, nsub = 2, name = "ellipsoid") -> Interface

The ellipsoid `center + R * (semiaxes .* s)`, `s` on the unit sphere, as a cubed sphere: 6 × `nsub`² patches, each
the equiangular map of a square of a cube face, normalized onto the sphere, then scaled and rotated (`R` a 3×3
rotation). Normals point outward: side A is the inside, `δa = a_inside - a_outside`.
"""
function ellipsoid_interface(center, semiaxes, δa::Real; R = Matrix{Float64}(I, 3, 3), nsub::Int = 2,
                             name::AbstractString = "ellipsoid")
    c = collect(Float64, center); ax = collect(Float64, semiaxes); Rm = Matrix{Float64}(R)
    det(Rm) > 0 || throw(ArgumentError("R must be a proper rotation"))
    patches = Any[]
    for face in 1:6, i in 1:nsub, j in 1:nsub
        u0 = -1 + 2(i - 1) / nsub; u1 = -1 + 2i / nsub; v0 = -1 + 2(j - 1) / nsub; v1 = -1 + 2j / nsub
        push!(patches, function (u, v)
            ξ = tan(π / 4 * (u0 + (u1 - u0) * u)); η = tan(π / 4 * (v0 + (v1 - v0) * v))
            s = _cube_face_point(face, ξ, η); s ./= norm(s)
            c .+ Rm * (ax .* s)
        end)
    end
    return Interface(patches, δa; name)
end

"""
    sphere_interface(center, radius, δa; nsub = 2, name = "sphere") -> Interface

The sphere of `radius` about `center` ([`ellipsoid_interface`](@ref) with equal semi-axes); outward normals,
`δa = a_inside - a_outside`.
"""
sphere_interface(center, radius::Real, δa::Real; nsub::Int = 2, name::AbstractString = "sphere") =
    ellipsoid_interface(center, (radius, radius, radius), δa; nsub, name)

# ---- graded discretization ----

struct _Elem
    patch::Int          # global patch index
    u0::Float64
    u1::Float64
    v0::Float64
    v1::Float64
end

_elem_point(P, e::_Elem, u, v) = collect(Float64, P(e.u0 + (e.u1 - e.u0) * u, e.v0 + (e.v1 - e.v0) * v))

function _box_dist(x, lo, hi)
    s = 0.0
    for d in 1:3
        s += max(lo[d] - x[d], 0.0, x[d] - hi[d])^2
    end
    return sqrt(s)
end

# element size (largest distance between sampled points) and distance to the support box (smallest over sampled
# points, less half the size)
function _elem_size_dist(P, e::_Elem, lo, hi)
    pts = [_elem_point(P, e, u, v) for u in (0.0, 0.5, 1.0), v in (0.0, 0.5, 1.0)]
    sz = 0.0
    for a in pts, b in pts
        sz = max(sz, norm(a .- b))
    end
    d = minimum(_box_dist(p, lo, hi) for p in pts)
    return sz, max(d - sz / 2, 0.0)
end

"""
    SurfaceQuadrature(interfaces; support, L0, κ = 0.5, order = 12, max_size = Inf, max_depth = 24)

The source discretization of the surface integrals: every patch of `interfaces` (a vector of [`Interface`](@ref))
is split (quadtree in its parameter square) until each element's size is at most `min(max_size, κ (d + L0))`, `d`
its distance to the box `support = (lo, hi)` of the density (e.g. [`density_support`](@ref)). The integrands vary on
the density's scale near it and like the field's distance farther away; `L0` (a length, e.g. the density's width)
sets the finest size `κ L0`. All elements form one SCTL `QuadElemList` of `order`² Gauss-Legendre nodes (`order` in
4, 8, 12, 16, 20). Stores the nodes `X`, unit normals `normal` (from A to B), weights `w` and the jump `δa` per node.

The discretization depends on the geometry and the support only: reuse it for every density with that support.
"""
struct SurfaceQuadrature
    interfaces::Vector{Interface}
    elems::Vector{_Elem}
    elem_δa::Vector{Float64}
    order::Int
    elist::Union{Nothing, SCTL.QuadElemList}   # nothing when no interface has δa ≠ 0
    X::Matrix{Float64}
    normal::Matrix{Float64}
    w::Vector{Float64}
    δa::Vector{Float64}
    support::Tuple{NTuple{3, Float64}, NTuple{3, Float64}}
    L0::Float64
    κ::Float64
end

function SurfaceQuadrature(interfaces::AbstractVector{Interface}; support, L0::Real, κ::Real = 0.5, order::Int = 12,
                           max_size::Real = Inf, max_depth::Int = 24)
    order in (4, 8, 12, 16, 20) || throw(ArgumentError("order must be one of 4, 8, 12, 16, 20"))
    L0 > 0 || throw(ArgumentError("L0 must be positive"))
    lo = ntuple(d -> Float64(support[1][d]), 3); hi = ntuple(d -> Float64(support[2][d]), 3)
    itf = [i for i in interfaces if i.δa != 0]
    patches = Any[]; pδa = Float64[]
    for i in itf, p in i.patches
        push!(patches, p); push!(pδa, i.δa)
    end
    elems = _Elem[]; eδa = Float64[]
    for (ip, P) in enumerate(patches)
        stack = [(_Elem(ip, 0.0, 1.0, 0.0, 1.0), 0)]
        while !isempty(stack)
            e, depth = pop!(stack)
            sz, d = _elem_size_dist(P, e, lo, hi)
            if depth < max_depth && sz > min(max_size, κ * (d + L0))
                um = (e.u0 + e.u1) / 2; vm = (e.v0 + e.v1) / 2
                push!(stack, (_Elem(ip, e.u0, um, e.v0, vm), depth + 1), (_Elem(ip, um, e.u1, e.v0, vm), depth + 1),
                      (_Elem(ip, e.u0, um, vm, e.v1), depth + 1), (_Elem(ip, um, e.u1, vm, e.v1), depth + 1))
            else
                push!(elems, e); push!(eδa, pδa[ip])
            end
        end
    end
    if isempty(elems)            # all coefficients equal: U = a0 N[ρ]
        return SurfaceQuadrature(itf, elems, eδa, order, nothing, zeros(3, 0), zeros(3, 0), Float64[], Float64[],
                                 (lo, hi), Float64(L0), Float64(κ))
    end
    el = SCTL.QuadElemList(order, length(elems)) do e, u, v
        _elem_point(patches[elems[e].patch], elems[e], u, v)
    end
    X, N = SCTL.node_coord(el)
    w = SCTL.node_weights(el)
    δa = repeat(eδa, inner = order^2)
    return SurfaceQuadrature(itf, elems, eδa, order, el, X, N, w, δa, (lo, hi), Float64(L0), Float64(κ))
end

Base.show(io::IO, s::SurfaceQuadrature) =
    print(io, "SurfaceQuadrature(", length(s.interfaces), " interface(s), ", length(s.elems), " elements of order ",
          s.order, ", ", size(s.X, 2), " nodes)")
