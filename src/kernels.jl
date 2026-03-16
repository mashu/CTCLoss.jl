# KernelAbstractions kernels for batched CTC forward-backward (device-agnostic).

@kernel function ctc_gather_emit_kernel!(em::AbstractArray{T,3},
        @Const(lp::AbstractArray{T,3}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where {T}
    s, t, b = @index(Global, NTuple)
    @inbounds if s <= Sl[b] && t <= Tl[b]
        em[s, t, b] = lp[lab[s, b], t, b]
    else
        em[s, t, b] = T(-Inf)
    end
end

@kernel function ctc_fwd_init_kernel!(α::AbstractArray{T,3},
        @Const(em::AbstractArray{T,3}), @Const(Sl::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds begin
        # FIX (Bug 2): Guard against Tl[b] == 0 — skip init entirely
        if Tl[b] >= Int32(1)
            Sl[b] >= Int32(1) && (α[1, 1, b] = em[1, 1, b])
            Sl[b] >= Int32(2) && (α[2, 1, b] = em[2, 1, b])
        end
    end
end

@kernel function ctc_fwd_step_kernel!(α::AbstractArray{T,3},
        @Const(em::AbstractArray{T,3}), @Const(skip::AbstractMatrix{Bool}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        t::Int32) where {T}
    s, b = @index(Global, NTuple)
    @inbounds if s <= Sl[b] && t <= Tl[b]
        tp = t - Int32(1)
        v = α[s, tp, b]
        s > Int32(1) && (v = logaddexp(v, α[s - Int32(1), tp, b]))
        (s > Int32(2) && skip[s, b]) && (v = logaddexp(v, α[s - Int32(2), tp, b]))
        α[s, t, b] = v + em[s, t, b]
    end
end

@kernel function ctc_bwd_init_kernel!(β::AbstractArray{T,3},
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds begin
        Sb = Sl[b]; Tb = Tl[b]
        # FIX (Bug 2): Guard against Tb == 0
        if Tb >= Int32(1)
            Sb >= Int32(1) && (β[Sb, Tb, b] = T(0))
            Sb >= Int32(2) && (β[Sb - Int32(1), Tb, b] = T(0))
        end
    end
end

@kernel function ctc_bwd_step_kernel!(β::AbstractArray{T,3},
        @Const(em::AbstractArray{T,3}), @Const(skip::AbstractMatrix{Bool}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        t::Int32) where {T}
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

@kernel function ctc_loss_kernel!(losses::AbstractVector{T}, @Const(α::AbstractArray{T,3}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds begin
        Sb = Sl[b]; Tb = Tl[b]
        # FIX (Bug 2): When Tb == 0 there is no valid alignment at all
        if Tb < Int32(1) || Sb < Int32(1)
            losses[b] = T(Inf)
        else
            v = α[Sb, Tb, b]
            Sb >= Int32(2) && (v = logaddexp(v, α[Sb - Int32(1), Tb, b]))
            losses[b] = -v
        end
    end
end

@kernel function ctc_grad_kernel!(grad::AbstractArray{T,3},
        @Const(α::AbstractArray{T,3}), @Const(β::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,3}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Sl::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(nll::AbstractVector{T})) where {T}
    k, t, b = @index(Global, NTuple)
    @inbounds if t > Tl[b]
        grad[k, t, b] = T(0)
    else
        # FIX (Bug 1): When the alignment is impossible nll[b] == Inf,
        # so ab + nll[b] = -Inf + Inf = NaN.  Emit zero gradient instead —
        # there is no probability mass to redistribute.
        nll_b = nll[b]
        if isinf(nll_b) && nll_b > T(0)
            grad[k, t, b] = T(0)
        else
            ab = T(-Inf)
            Sb = Sl[b]
            for s in Int32(1):Sb
                lab[s, b] == k && (ab = logaddexp(ab, α[s, t, b] + β[s, t, b]))
            end
            grad[k, t, b] = exp(lp[k, t, b]) - exp(ab + nll_b)
        end
    end
end
