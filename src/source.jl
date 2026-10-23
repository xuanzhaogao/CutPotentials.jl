# Volume sources: samples of a smooth density on a (possibly anisotropic or skewed) lattice.

"""
    VolumeSource(positions, weights, density, basis)
    VolumeSource(values::AbstractArray{<:Real,3}, origin, basis)
    VolumeSource(f::Function, origin, basis, dims)

Samples `density` of a smooth density ρ at the points `positions` (`3 × N`), with quadrature `weights` (cell
volumes), on a lattice with primitive step vectors `basis` (a 3×3 matrix whose column `j` is the step along lattice
direction `j`, or a tuple of three 3-vectors). The point-charge (and spectral) representation of ρ is
`Σ_s weights[s] density[s] δ(x - positions[s])`.

The lattice forms build the samples `x_ijk = origin + basis * (i-1, j-1, k-1)` with weights `|det basis|`, from an
array of values or from a function `f(x)` sampled on `dims = (n1, n2, n3)` points.

The basis is used for the near/far geometry of the TKM field ([`SmoothField`](@ref)): the characteristic spacing
`h = ‖basis‖₂`, the source box (the bounding box of the samples inflated by half a cell) and the spectral cutoff
`kmax = π / h_min` with `h_min` the shortest lattice vector.
"""
struct VolumeSource
    positions::Matrix{Float64}
    weights::Vector{Float64}
    density::Vector{Float64}
    basis::Matrix{Float64}
    function VolumeSource(positions::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real},
                          density::AbstractVector{<:Real}, basis)
        size(positions, 1) == 3 || throw(DimensionMismatch("positions must be 3 × N"))
        n = size(positions, 2)
        (length(weights) == n && length(density) == n) ||
            throw(DimensionMismatch("need one weight and one density value per position"))
        n >= 1 || throw(ArgumentError("need at least one sample"))
        B = _basis_matrix(basis)
        abs(det(B)) > 0 || throw(ArgumentError("the lattice basis is singular"))
        return new(Matrix{Float64}(positions), Vector{Float64}(weights), Vector{Float64}(density), B)
    end
end

_basis_matrix(B::AbstractMatrix) = (size(B) == (3, 3) || throw(DimensionMismatch("basis must be 3×3")); Matrix{Float64}(B))
_basis_matrix(B::Union{Tuple, AbstractVector}) =
    (length(B) == 3 || throw(DimensionMismatch("need three basis vectors")); Float64[B[j][i] for i in 1:3, j in 1:3])

function VolumeSource(values::AbstractArray{<:Real, 3}, origin, basis)
    B = _basis_matrix(basis)
    o = collect(Float64, origin)
    n1, n2, n3 = size(values)
    X = Matrix{Float64}(undef, 3, length(values))
    c = 0
    for k in 1:n3, j in 1:n2, i in 1:n1
        c += 1
        @views X[:, c] .= o .+ B[:, 1] .* (i - 1) .+ B[:, 2] .* (j - 1) .+ B[:, 3] .* (k - 1)
    end
    return VolumeSource(X, fill(abs(det(B)), length(values)), vec(Float64.(values)), B)
end

function VolumeSource(f::Function, origin, basis, dims)
    B = _basis_matrix(basis)
    o = collect(Float64, origin)
    vals = Array{Float64, 3}(undef, dims...)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        vals[i, j, k] = f(o .+ B[:, 1] .* (i - 1) .+ B[:, 2] .* (j - 1) .+ B[:, 3] .* (k - 1))
    end
    return VolumeSource(vals, o, B)
end

Base.length(s::VolumeSource) = length(s.density)

# the shortest nonzero lattice vector (search over small integer combinations of the basis)
function _shortest_lattice_vector(B::Matrix{Float64})
    h = Inf
    for i in -3:3, j in -3:3, k in -3:3
        (i, j, k) == (0, 0, 0) && continue
        h = min(h, norm(B * [i, j, k]))
    end
    return h
end

"""
    density_support(src; rtol = 1e-10) -> (lo, hi)

The bounding box of the samples of `src` whose |density| exceeds `rtol` times the largest one: the default support
toward which the surface elements are graded ([`SurfaceQuadrature`](@ref)).
"""
function density_support(src::VolumeSource; rtol::Real = 1e-10)
    m = maximum(abs, src.density)
    idx = findall(v -> abs(v) > rtol * m, src.density)
    isempty(idx) && (idx = collect(eachindex(src.density)))
    P = src.positions[:, idx]
    return (ntuple(d -> minimum(view(P, d, :)), 3), ntuple(d -> maximum(view(P, d, :)), 3))
end
