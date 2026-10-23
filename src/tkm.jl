# The smooth fields φ0 = N[ρ], ψ = ∇φ0 and H = ∇ψ of the full (unscreened) density: the truncated-kernel method
# (TKM) near the samples, with one real scalar fine grid and window-derivative gathering, and point charges by SCTL's
# FMM elsewhere. Adapted from DielectricBIE.jl (src/shape/volume_field.jl, src/tkm3d/, src/core/source_geometry.jl;
# MIT, same authors).

# ---- near/far geometry (Section 3 of the four-index BIE paper) ----

struct NearGeometry
    lo::NTuple{3, Float64}        # padded source box B_pad: TKM targets
    hi::NTuple{3, Float64}
    center::NTuple{3, Float64}    # centre of the source box B
    L::Float64                    # truncation radius of the kernel
    dks::NTuple{3, Float64}       # Fourier spacings
    h::Float64                    # ‖basis‖₂
end

function _near_geometry(src::VolumeSource, c_pad::Real)
    c_pad >= 0 || throw(ArgumentError("c_pad must be >= 0"))
    B = src.basis
    half = ntuple(a -> (abs(B[a, 1]) + abs(B[a, 2]) + abs(B[a, 3])) / 2, 3)
    P = src.positions
    loB = ntuple(a -> minimum(view(P, a, :)) - half[a], 3)
    hiB = ntuple(a -> maximum(view(P, a, :)) + half[a], 3)
    l = ntuple(a -> hiB[a] - loB[a], 3)
    h = opnorm(B, 2)
    hn = c_pad * h
    lo = ntuple(d -> loB[d] - hn, 3); hi = ntuple(d -> hiB[d] + hn, 3)
    center = ntuple(d -> (loB[d] + hiB[d]) / 2, 3)
    L = sqrt((l[1] + hn)^2 + (l[2] + hn)^2 + (l[3] + hn)^2)
    dks = ntuple(d -> prevfloat(2π / (l[d] + hn + L)), 3)
    return NearGeometry(lo, hi, center, L, dks, h)
end

@inline _in_box(g::NearGeometry, Y::AbstractMatrix, i::Integer) =
    (g.lo[1] <= Y[1, i] <= g.hi[1]) && (g.lo[2] <= Y[2, i] <= g.hi[2]) && (g.lo[3] <= Y[3, i] <= g.hi[3])

# ---- truncated Laplace kernel ----

function _centered_mode_axis(dk::Float64, kmax::Float64)
    m = ceil(Int, kmax / dk)
    return [dk * (i - m - 1) for i in 1:(2m + 1)]
end

@inline _truncated_laplace_hat(k::Float64, L::Float64) = iszero(k) ? L^2 / 2 : 2 * (sin(L * k / 2) / k)^2

function _truncated_kernel_multiply!(c::Array{ComplexF64, 3}, kx, ky, kz, kmax::Float64, L::Float64)
    Threads.@threads for iz in eachindex(kz)
        @inbounds for iy in eachindex(ky), ix in eachindex(kx)
            k = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
            c[ix, iy, iz] = k <= kmax ? c[ix, iy, iz] * _truncated_laplace_hat(k, L) : zero(ComplexF64)
        end
    end
    return c
end

# ---- the window: ES tapered by (1 - z²)^m, analytic with two derivatives ----

struct _Window
    ns::Int
    beta::Float64
    m::Int
end

function _window_params(tolw::Real, sigma::Real; nsmax::Int = 24, taper::Int = 3)
    ns = clamp(ceil(Int, log(1 / max(Float64(tolw), eps())) / (π * sqrt(1 - 1 / sigma))), 2, nsmax)
    return _Window(ns, 0.97 * π * ns * (1 - 1 / (2 * sigma)), taper)
end

# w, dw/dt, d²w/dt² at t (grid units), w(t) = exp(β(q - 1)) q^(2m), q = sqrt(1 - z²), z = t/(ns/2)
@inline function _window_eval(win::_Window, t::Float64)
    h = win.ns / 2; z = t / h; r = 1 - z * z
    r <= 0 && return 0.0, 0.0, 0.0
    q = sqrt(r); β = win.beta; m2 = 2 * win.m
    w = exp(β * (q - 1)) * q^m2
    dq = -z / q / h; d2q = -1 / (q * r) / h^2
    a = β + m2 / q
    g1 = a * dq; g2 = a * d2q - m2 / r * dq^2
    return w, w * g1, w * (g2 + g1^2)
end

function _legendre_pd(n::Int, x::Float64)
    n == 0 && return 1.0, 0.0
    n == 1 && return x, 1.0
    p0 = 1.0; p1 = x
    for i in 1:(n - 1)
        p0, p1 = p1, ((2i + 1) * x * p1 - i * p0) / (i + 1)
    end
    return p1, n * (x * p1 - p0) / (x^2 - 1)
end

function _gauss_legendre(n::Int)
    x = zeros(n); w = zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        for _ in 1:100
            p, dp = _legendre_pd(n, z)
            dz = p / dp; z -= dz
            abs(dz) < 1e-16 && break
        end
        _, dp = _legendre_pd(n, z)
        x[n + 1 - i] = z; w[n + 1 - i] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

# Fourier factors of the window on a fine grid of nf points (Gauss-Legendre quadrature over its support, with the
# (-1)^k of FINUFFT's half-period fold)
function _window_fseries(nf::Int, win::_Window)
    J2 = win.ns / 2.0
    q = 2 * floor(Int, 2 + 3.0 * J2)
    z, w = _gauss_legendre(2q)
    f = zeros(q); a = zeros(ComplexF64, q)
    # nodes in (0, 1] of the symmetric rule (FINUFFT uses the positive half)
    zp = z[(q + 1):end]; wp = w[(q + 1):end]
    for n in 1:q
        zn = zp[n] * J2
        f[n] = J2 * wp[n] * first(_window_eval(win, zn))
        a[n] = -exp(2π * im * zn / nf)
    end
    aj = ones(ComplexF64, q); out = zeros(nf ÷ 2 + 1)
    for j in 0:(nf ÷ 2)
        s = 0.0
        for n in 1:q
            s += f[n] * 2.0 * real(aj[n])
        end
        out[j + 1] = s
        for n in 1:q
            aj[n] *= a[n]
        end
    end
    return out
end

function _next235even(n::Integer)
    n <= 2 && return 2
    m = isodd(n) ? n + 1 : n
    while true
        k = m
        for p in (2, 3, 5)
            while k % p == 0
                k ÷= p
            end
        end
        k == 1 && return m
        m += 2
    end
end

# The real scalar fine grid: coefficients (centered modes) / window factors, placed in FFT order on the fine grid,
# bfft'd, real part.
function _window_grid(coeff::Array{ComplexF64, 3}, win::_Window, sigma::Float64)
    ms = size(coeff)
    nf = ntuple(d -> _next235even(max(ceil(Int, sigma * ms[d]), 2 * win.ns)), 3)
    phi = ntuple(d -> _window_fseries(nf[d], win), 3)
    fw = zeros(ComplexF64, nf)
    kmin = ntuple(d -> -(ms[d] ÷ 2), 3); kmaxs = ntuple(d -> (ms[d] - 1) ÷ 2, 3)
    fwi(k, n) = k >= 0 ? k + 1 : n + k + 1
    @inbounds for k3 in kmin[3]:kmaxs[3], k2 in kmin[2]:kmaxs[2], k1 in kmin[1]:kmaxs[1]
        fw[fwi(k1, nf[1]), fwi(k2, nf[2]), fwi(k3, nf[3])] =
            coeff[k1 + ms[1] ÷ 2 + 1, k2 + ms[2] ÷ 2 + 1, k3 + ms[3] ÷ 2 + 1] /
            (phi[1][abs(k1) + 1] * phi[2][abs(k2) + 1] * phi[3][abs(k3) + 1])
    end
    FFTW.bfft!(fw)
    grid = Array{Float64, 3}(undef, nf)
    Threads.@threads for i in eachindex(grid)
        @inbounds grid[i] = real(fw[i])
    end
    return grid, nf
end

"""
    SmoothField(src::VolumeSource; tol = 1e-10, c_pad = 5.0, kmax = nothing, window_sigma = 1.5,
                window_tol = tol * 1e-4, far_digits = nothing, nthreads = min(16, Threads.nthreads()))

The free-space Newton potential φ0 = N[ρ] (kernel 1/(4π|y - x|)) of the full, unscreened density of `src`, with its
gradient ψ and Hessian H, ready for evaluation at any points ([`field_values`](@ref)).

- **Near the samples** (inside the source box padded by `c_pad · ‖basis‖₂`): the truncated-kernel method. A type-1
  NUFFT (FINUFFT, accuracy `tol`) of the charges, times the truncated Laplace kernel 2 (sin(Lk/2)/k)² for |k| ≤
  `kmax` (default `π/h_min`), on one real scalar fine grid (oversampling `window_sigma`). Potential, gradient and
  Hessian are gathered from that grid with the window w, w′ and w″ of an analytic ES window tapered by (1 - z²)³
  (`window_tol` sets its width).
- **Elsewhere**: point charges `weights .* density` by SCTL's FMM (`Laplace3D_FxU`, `_FxdU`, `_Fxd2U`) at
  `far_digits` digits (default `-log10(tol) + 5`, at most 15).

Accuracy is limited by `tol` and by the sampling of ρ (spectrally accurate for a smooth, well-resolved density).
"""
struct SmoothField
    src::VolumeSource
    geom::NearGeometry
    kmax::Float64
    tol::Float64
    grid::Array{Float64, 3}
    nf::NTuple{3, Int}
    win::_Window
    alpha::NTuple{3, Float64}
    prefactor::Float64
    far_digits::Int
end

function SmoothField(src::VolumeSource; tol::Real = 1e-10, c_pad::Real = 5.0, kmax::Union{Nothing, Real} = nothing,
                     window_sigma::Real = 1.5, window_tol::Union{Nothing, Real} = nothing,
                     far_digits::Union{Nothing, Integer} = nothing, nthreads::Integer = min(16, Threads.nthreads()))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    window_sigma > 1 || throw(ArgumentError("window_sigma must be > 1"))
    g = _near_geometry(src, c_pad)
    km = kmax === nothing ? π / _shortest_lattice_vector(src.basis) : Float64(kmax)
    ks = ntuple(d -> _centered_mode_axis(g.dks[d], km), 3)
    nmodes = length.(ks)
    P = src.positions
    xs = ntuple(d -> g.dks[d] .* (P[d, :] .- g.center[d]), 3)
    charges = complex.(src.weights .* src.density)
    c = FINUFFT.nufft3d1(xs[1], xs[2], xs[3], charges, -1, Float64(tol), nmodes...; nthreads = Int(nthreads))
    c = ndims(c) == 4 ? dropdims(c; dims = 4) : c
    _truncated_kernel_multiply!(c, ks..., km, g.L)
    win = _window_params(window_tol === nothing ? tol * 1e-4 : window_tol, window_sigma)
    grid, nf = _window_grid(c, win, Float64(window_sigma))
    alpha = ntuple(d -> nf[d] * g.dks[d] / (2π), 3)
    pref = g.dks[1] * g.dks[2] * g.dks[3] / (2π)^3
    fd = far_digits === nothing ? clamp(ceil(Int, -log10(tol)) + 5, 1, 15) : Int(far_digits)
    return SmoothField(src, g, km, Float64(tol), grid, nf, win, alpha, pref, fd)
end

Base.show(io::IO, f::SmoothField) =
    print(io, "SmoothField(", length(f.src), " samples, fine grid ", join(f.nf, "×"), ", window ns ", f.win.ns, ")")

@inline function _stencil(x::Float64, nf::Int, ns::Int)
    u = mod(x / (2π) + 0.5, 1.0) * nf
    istart = ceil(Int, u - ns / 2)
    return istart, istart - u
end

# fused gathering of φ, ∇φ, ∇∇φ (rows xx, yy, zz, xy, xz, yz) at the points idx of Y; order 0, 1 or 2
function _gather!(P, G, H, f::SmoothField, Y::AbstractMatrix, idx::Vector{Int}, order::Int)
    grid = f.grid; nf1, nf2, nf3 = f.nf; ns = f.win.ns; win = f.win
    α = f.alpha; C = f.prefactor; g = f.geom
    n = length(idx)
    nch = max(1, min(Threads.nthreads(), n))
    Threads.@threads for ch in 1:nch
        a0 = zeros(ns); a1 = zeros(ns); a2 = zeros(ns)
        b0 = zeros(ns); b1 = zeros(ns); b2 = zeros(ns)
        c0 = zeros(ns); c1 = zeros(ns); c2 = zeros(ns)
        i1 = zeros(Int, ns)
        @inbounds for jj in (div((ch - 1) * n, nch) + 1):div(ch * n, nch)
            j = idx[jj]
            s1, x1 = _stencil(g.dks[1] * (Y[1, j] - g.center[1]), nf1, ns)
            s2, x2 = _stencil(g.dks[2] * (Y[2, j] - g.center[2]), nf2, ns)
            s3, x3 = _stencil(g.dks[3] * (Y[3, j] - g.center[3]), nf3, ns)
            for l in 1:ns
                a0[l], a1[l], a2[l] = _window_eval(win, x1 + (l - 1))
                b0[l], b1[l], b2[l] = _window_eval(win, x2 + (l - 1))
                c0[l], c1[l], c2[l] = _window_eval(win, x3 + (l - 1))
                i1[l] = mod(s1 + l - 1, nf1) + 1
            end
            u000 = u100 = u010 = u001 = u200 = u020 = u002 = u110 = u101 = u011 = 0.0
            for l3 in 1:ns
                iz = mod(s3 + l3 - 1, nf3) + 1
                t00 = t10 = t01 = t20 = t02 = t11 = 0.0
                for l2 in 1:ns
                    iy = mod(s2 + l2 - 1, nf2) + 1
                    q0 = 0.0; q1 = 0.0; q2 = 0.0
                    @simd for l1 in 1:ns
                        v = grid[i1[l1], iy, iz]
                        q0 = muladd(v, a0[l1], q0); q1 = muladd(v, a1[l1], q1); q2 = muladd(v, a2[l1], q2)
                    end
                    t00 = muladd(q0, b0[l2], t00)
                    t10 = muladd(q1, b0[l2], t10); t01 = muladd(q0, b1[l2], t01)
                    t20 = muladd(q2, b0[l2], t20); t02 = muladd(q0, b2[l2], t02); t11 = muladd(q1, b1[l2], t11)
                end
                u000 = muladd(t00, c0[l3], u000)
                u100 = muladd(t10, c0[l3], u100); u010 = muladd(t01, c0[l3], u010); u001 = muladd(t00, c1[l3], u001)
                u200 = muladd(t20, c0[l3], u200); u020 = muladd(t02, c0[l3], u020); u002 = muladd(t00, c2[l3], u002)
                u110 = muladd(t11, c0[l3], u110); u101 = muladd(t10, c1[l3], u101); u011 = muladd(t01, c1[l3], u011)
            end
            P[j] = C * u000
            if order >= 1
                G[1, j] = -C * α[1] * u100; G[2, j] = -C * α[2] * u010; G[3, j] = -C * α[3] * u001
            end
            if order >= 2
                H[1, j] = C * α[1]^2 * u200; H[2, j] = C * α[2]^2 * u020; H[3, j] = C * α[3]^2 * u002
                H[4, j] = C * α[1] * α[2] * u110; H[5, j] = C * α[1] * α[3] * u101; H[6, j] = C * α[2] * α[3] * u011
            end
        end
    end
    return nothing
end

"""
    field_values(f::SmoothField, Y; order = 2) -> (; pot, grad, hess)

φ0 (vector), ψ = ∇φ0 (`3 × n`) and, for `order = 2`, H = ∇∇φ0 (`6 × n`, rows ∂xx, ∂yy, ∂zz, ∂xy, ∂xz, ∂yz) at
the points `Y` (`3 × n`). `order = 1` skips the Hessian (`hess = nothing`).
"""
function field_values(f::SmoothField, Y::AbstractMatrix{<:Real}; order::Int = 2)
    size(Y, 1) == 3 || throw(DimensionMismatch("points must be 3 × n"))
    Yf = Matrix{Float64}(Y); n = size(Yf, 2)
    P = zeros(n); G = zeros(3, n); H = order >= 2 ? zeros(6, n) : zeros(6, 0)
    inb = [_in_box(f.geom, Yf, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    isempty(bidx) || _gather!(P, G, H, f, Yf, bidx, max(order, 1))
    if !isempty(oidx)
        src = f.src; Xs = src.positions; q = reshape(src.weights .* src.density, 1, :); Yo = Yf[:, oidx]
        P[oidx] .= vec(SCTL.fmm_eval(SCTL.Laplace3D_FxU(), Xs, q, Yo; digits = f.far_digits))
        G[:, oidx] .= SCTL.fmm_eval(SCTL.Laplace3D_FxdU(), Xs, q, Yo; digits = f.far_digits)
        if order >= 2
            U = SCTL.fmm_eval(SCTL.Laplace3D_Fxd2U(), Xs, q, Yo; digits = f.far_digits)
            H[:, oidx] .= U[[1, 5, 9, 2, 3, 6], :]
        end
    end
    return (; pot = P, grad = G, hess = order >= 2 ? H : nothing)
end
