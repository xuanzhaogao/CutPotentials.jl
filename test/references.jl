# Independent references for an isotropic Gaussian density ρ = N(c, σ² I) (unit mass):
#   - the exact smooth fields φ0, ψ, H (radial formulas);
#   - heat-kernel formulas for its truncation to a box or a ball: 1/(4π r) = ∫_0^∞ (4π t)^(-3/2) exp(-r²/4t) dt, and
#     K_t(y - x) ρ(x) = N(y; c, (σ² + 2t) I) N(x; m, s² I), so N[χ_Ω ρ](y) = ∫_0^∞ N(y; c, (σ²+2t) I) P_Ω(m, s) dt with
#     P_Ω the probability of Ω under N(m, s² I): a product of erf differences for a box, a noncentral-χ formula for
#     a ball. Derivatives in y by ForwardDiff.
using SpecialFunctions: erf
using ForwardDiff
using LinearAlgebra

struct Gaussian
    c::NTuple{3, Float64}
    σ::Float64
end

ρ(g::Gaussian, x) = exp(-sum(abs2, x .- g.c) / (2g.σ^2)) / (2π * g.σ^2)^1.5

function _fp(g::Gaussian, r)          # φ0'(r)
    σ = g.σ
    return (sqrt(2 / π) / σ * exp(-r^2 / (2σ^2)) * r - erf(r / (sqrt(2) * σ))) / (4π * r^2)
end
φ0(g::Gaussian, x) = (r = norm(x .- g.c); erf(r / (sqrt(2) * g.σ)) / (4π * r))
ψ0(g::Gaussian, x) = (d = collect(x .- g.c); r = norm(d); _fp(g, r) .* d ./ r)
function H0(g::Gaussian, x)          # 6-vector xx, yy, zz, xy, xz, yz
    d = collect(x .- g.c); r = norm(d); u = d ./ r; fp = _fp(g, r)
    H = (-ρ(g, x) - 2fp / r) .* (u * u') .+ (fp / r) .* (I - u * u')
    return [H[1, 1], H[2, 2], H[3, 3], H[1, 2], H[1, 3], H[2, 3]]
end

# exact fields as a "field" for CutPotentials.evaluate (field_values protocol)
struct ExactField
    g::Gaussian
end
function CutPotentials.field_values(f::ExactField, Y::AbstractMatrix; order::Int = 2)
    n = size(Y, 2); P = zeros(n); G = zeros(3, n); H = zeros(6, n)
    Threads.@threads for j in 1:n
        P[j] = φ0(f.g, Y[:, j]); G[:, j] .= ψ0(f.g, Y[:, j]); order >= 2 && (H[:, j] .= H0(f.g, Y[:, j]))
    end
    return (; pot = P, grad = G, hess = order >= 2 ? H : nothing)
end

const _GL16 = let
    # 16-point Gauss-Legendre on [-1, 1] (Golub-Welsch)
    n = 16; β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    (E.values, 2 .* E.vectors[1, :] .^ 2)
end

# ∫_0^∞ F(t) dt with t = e^p, p ∈ [-90, 80), 16-point Gauss-Legendre per unit interval
function _tint(F)
    x, w = _GL16
    s = zero(F(1.0))
    for p in -90:79, (xi, wi) in zip(x, w)
        t = exp(p + 0.5 + 0.5xi)
        s += 0.5wi * t * F(t)
    end
    return s
end

# 1D factor for the box: N(y; c, v) times the probability of [lo, hi] under N(m, s²)
function _E1(t, y, c, σ, lo, hi)
    v = σ^2 + 2t; s2 = 1 / (1 / (2t) + 1 / σ^2); s = sqrt(s2); m = s2 * (y / (2t) + c / σ^2)
    return exp(-(y - c)^2 / (2v)) / sqrt(2π * v) * (erf((hi - m) / (sqrt(2) * s)) - erf((lo - m) / (sqrt(2) * s))) / 2
end
Nbox(g::Gaussian, y, lo, hi) = _tint(t -> prod(_E1(t, y[d], g.c[d], g.σ, lo[d], hi[d]) for d in 1:3))

# probability of the ball |x - b| ≤ R under N(m, s² I), μ = |m - b|
function _Pball(μ, s, R)
    a = (erf((R - μ) / (sqrt(2) * s)) + erf((R + μ) / (sqrt(2) * s))) / 2
    x = R * μ / s^2
    b = x < 1 ? -(2s / sqrt(2π)) * exp(-(R^2 + μ^2) / (2s^2)) * sinh(x) / μ :
                (s / (μ * sqrt(2π))) * (exp(-(R + μ)^2 / (2s^2)) - exp(-(R - μ)^2 / (2s^2)))
    return a + b
end
function Nball(g::Gaussian, y, b, R)
    σ = g.σ
    return _tint() do t
        v = σ^2 + 2t; s2 = 1 / (1 / (2t) + 1 / σ^2)
        m = s2 .* (y ./ (2t) .+ collect(g.c) ./ σ^2)
        exp(-sum(abs2, y .- collect(g.c)) / (2v)) / (2π * v)^1.5 * _Pball(norm(m .- collect(b)), sqrt(s2), R)
    end
end

# value, gradient, Hessian (6-vector) of a scalar function of y
function vgh(F, y)
    y = collect(Float64, y)
    H = ForwardDiff.hessian(F, y)
    return F(y), ForwardDiff.gradient(F, y), [H[1, 1], H[2, 2], H[3, 3], H[1, 2], H[1, 3], H[2, 3]]
end
