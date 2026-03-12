# Batched CTC loss: forward-backward on device, ChainRulesCore.rrule for gradients.
# Blank index defaults to last class (size(logits, 1)); override with keyword `blank`.

@inline function logaddexp(a::T, b::T) where T <: AbstractFloat
    m = max(a, b)
    m == T(-Inf) && return T(-Inf)
    m + log(exp(a - m) + exp(b - m))
end

"""
    expand_ctc_labels(targets, blank) → (labels, exp_lens, skip)

Interleave blanks: `[a, b, c]` → `[blank, a, blank, b, blank, c, blank]`.
Returns padded `(max_S, B)` Int32 matrix, lengths `Vector{Int32}`, and
`(max_S, B)` Bool skip-allowed mask.
"""
function expand_ctc_labels(targets::Vector{Vector{Int}}, blank::Int)
    B = length(targets)
    exp_lens = Int32[2 * length(t) + 1 for t in targets]
    S = Int(maximum(exp_lens; init = Int32(1)))
    labels = fill(Int32(blank), S, B)
    for b in 1:B, (i, tok) in enumerate(targets[b])
        labels[2i, b] = Int32(tok)
    end
    skip = fill(false, S, B)
    for b in 1:B, s in 3:exp_lens[b]
        skip[s, b] = labels[s, b] != blank && labels[s, b] != labels[s - 2, b]
    end
    labels, exp_lens, skip
end

# ─── KA Kernels ───────────────────────────────────────────────────────────────

@kernel function ctc_gather_emit_kernel!(em::AbstractArray{T, 3},
        @Const(lp::AbstractArray{T, 3}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where T
    s, t, b = @index(Global, NTuple)
    @inbounds if s <= Sl[b] && t <= Tl[b]
        em[s, t, b] = lp[lab[s, b], t, b]
    else
        em[s, t, b] = T(-Inf)
    end
end

@kernel function ctc_fwd_init_kernel!(α::AbstractArray{T, 3},
        @Const(em::AbstractArray{T, 3}), @Const(Sl::AbstractVector{Int32})) where T
    b = @index(Global)
    @inbounds begin
        Sl[b] >= Int32(1) && (α[1, 1, b] = em[1, 1, b])
        Sl[b] >= Int32(2) && (α[2, 1, b] = em[2, 1, b])
    end
end

@kernel function ctc_fwd_step_kernel!(α::AbstractArray{T, 3},
        @Const(em::AbstractArray{T, 3}), @Const(skip::AbstractMatrix{Bool}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        t::Int32) where T
    s, b = @index(Global, NTuple)
    @inbounds if s <= Sl[b] && t <= Tl[b]
        tp = t - Int32(1)
        v = α[s, tp, b]
        s > Int32(1) && (v = logaddexp(v, α[s - Int32(1), tp, b]))
        (s > Int32(2) && skip[s, b]) && (v = logaddexp(v, α[s - Int32(2), tp, b]))
        α[s, t, b] = v + em[s, t, b]
    end
end

@kernel function ctc_bwd_init_kernel!(β::AbstractArray{T, 3},
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where T
    b = @index(Global)
    @inbounds begin
        Sb = Sl[b]; Tb = Tl[b]
        Sb >= Int32(1) && (β[Sb, Tb, b] = T(0))
        Sb >= Int32(2) && (β[Sb - Int32(1), Tb, b] = T(0))
    end
end

@kernel function ctc_bwd_step_kernel!(β::AbstractArray{T, 3},
        @Const(em::AbstractArray{T, 3}), @Const(skip::AbstractMatrix{Bool}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        t::Int32) where T
    s, b = @index(Global, NTuple)
    @inbounds begin
        Sb = Sl[b]
        if s <= Sb && t < Tl[b]
            t1 = t + Int32(1)
            v = β[s, t1, b] + em[s, t1, b]
            if s + Int32(1) <= Sb
                v = logaddexp(v, β[s + Int32(1), t1, b] + em[s + Int32(1), t1, b])
            end
            if s + Int32(2) <= Sb && skip[s + Int32(2), b]
                v = logaddexp(v, β[s + Int32(2), t1, b] + em[s + Int32(2), t1, b])
            end
            β[s, t, b] = v
        end
    end
end

@kernel function ctc_loss_kernel!(losses::AbstractVector{T}, @Const(α::AbstractArray{T, 3}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where T
    b = @index(Global)
    @inbounds begin
        Sb = Sl[b]; Tb = Tl[b]
        v = α[Sb, Tb, b]
        Sb >= Int32(2) && (v = logaddexp(v, α[Sb - Int32(1), Tb, b]))
        losses[b] = -v
    end
end

@kernel function ctc_grad_kernel!(grad::AbstractArray{T, 3},
        @Const(α::AbstractArray{T, 3}), @Const(β::AbstractArray{T, 3}),
        @Const(lp::AbstractArray{T, 3}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(nll::AbstractVector{T})) where T
    k, t, b = @index(Global, NTuple)
    @inbounds if t > Tl[b]
        grad[k, t, b] = T(0)
    else
        ab = T(-Inf)
        Sb = Sl[b]
        for s in Int32(1):Sb
            lab[s, b] == k && (ab = logaddexp(ab, α[s, t, b] + β[s, t, b]))
        end
        grad[k, t, b] = exp(lp[k, t, b]) - exp(ab + nll[b])
    end
end

# ─── Forward-backward ────────────────────────────────────────────────────────

function ctc_forward_backward(logits::AbstractArray{T, 3},
                              labels::Matrix{Int32}, exp_lens::Vector{Int32},
                              skip::Matrix{Bool}, input_lengths::Vector{Int}) where T
    V, Tmax, B = size(logits)
    Smax = size(labels, 1)
    backend = KernelAbstractions.get_backend(logits)

    lab_d  = copyto!(similar(logits, Int32, Smax, B), labels)
    Sl_d   = copyto!(similar(logits, Int32, B), exp_lens)
    Tl_d   = copyto!(similar(logits, Int32, B), Int32.(input_lengths))
    skip_d = copyto!(similar(logits, Bool,  Smax, B), skip)

    log_probs = logsoftmax(logits; dims = 1)

    em = similar(logits, T, Smax, Tmax, B)
    ctc_gather_emit_kernel!(backend)(em, log_probs, lab_d, Sl_d, Tl_d;
                                     ndrange = (Smax, Tmax, B))

    α = fill!(similar(logits, T, Smax, Tmax, B), T(-Inf))
    ctc_fwd_init_kernel!(backend)(α, em, Sl_d; ndrange = B)
    for t in Int32(2):Int32(Tmax)
        ctc_fwd_step_kernel!(backend)(α, em, skip_d, Sl_d, Tl_d, t;
                                      ndrange = (Smax, B))
    end

    nll = similar(logits, T, B)
    ctc_loss_kernel!(backend)(nll, α, Sl_d, Tl_d; ndrange = B)

    β = fill!(similar(logits, T, Smax, Tmax, B), T(-Inf))
    ctc_bwd_init_kernel!(backend)(β, Sl_d, Tl_d; ndrange = B)
    for t in Int32(Tmax - 1):-Int32(1):Int32(1)
        ctc_bwd_step_kernel!(backend)(β, em, skip_d, Sl_d, Tl_d, t;
                                      ndrange = (Smax, B))
    end

    grad = similar(logits, T, V, Tmax, B)
    ctc_grad_kernel!(backend)(grad, α, β, log_probs, lab_d, Sl_d, Tl_d, nll;
                              ndrange = (V, Tmax, B))

    KernelAbstractions.synchronize(backend)

    loss = sum(nll) / T(B)
    grad ./= T(B)
    loss, grad
end

# ─── Public API ───────────────────────────────────────────────────────────────

"""
    ctc_loss_batched(logits, targets, input_lengths [; blank])

Batched CTC loss (mean over batch). `logits`: `(V, time, batch)` raw logits.
`targets`: `Vector{Vector{Int}}` label sequences. `input_lengths`: valid frames per sample.
`blank` defaults to `size(logits, 1)` (blank = last class). For AD with custom blank,
use the 4-arg form `ctc_loss_batched(logits, targets, input_lengths, blank)`.
"""
function ctc_loss_batched(logits::AbstractArray{T, 3},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int};
                          blank::Int = size(logits, 1)) where T
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, _ = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    loss
end

function ctc_loss_batched(logits::AbstractArray{T, 3},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int},
                          blank::Int) where T
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, _ = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    loss
end

function ChainRulesCore.rrule(::typeof(ctc_loss_batched),
                              logits::AbstractArray{T, 3},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int}) where T
    blank = size(logits, 1)
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, grad = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    ctc_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent())
    loss, ctc_pullback
end

function ChainRulesCore.rrule(::typeof(ctc_loss_batched),
                              logits::AbstractArray{T, 3},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              blank::Int) where T
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, grad = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    ctc_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(), NoTangent())
    loss, ctc_pullback
end

"""
    ctc_greedy_decode(logits, input_lengths; blank)

Greedy CTC decode: argmax per frame, collapse repeats, drop blanks.
Returns `Vector{Vector{Int}}` (one sequence per batch element).
`blank` defaults to `size(logits, 1)`.
"""
function ctc_greedy_decode(logits::AbstractArray{<:Real, 3},
                           input_lengths::Vector{Int};
                           blank::Int = size(logits, 1))
    ctc_greedy_decode(Array(logits), input_lengths, blank)
end

function ctc_greedy_decode(logits_cpu::Array{<:Real, 3},
                           input_lengths::Vector{Int},
                           blank::Int)
    _, _, batch = size(logits_cpu)
    results = Vector{Vector{Int}}(undef, batch)
    for b in 1:batch
        ids = Int[]
        prev = -1
        for t in 1:input_lengths[b]
            c = argmax(@view logits_cpu[:, t, b])
            if c != prev
                c != blank && push!(ids, c)
            end
            prev = c
        end
        results[b] = ids
    end
    results
end
