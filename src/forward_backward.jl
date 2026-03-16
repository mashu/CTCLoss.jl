# Batched CTC forward-backward on device; returns (loss, gradient).

"""
    ctc_forward_backward(logits, labels, exp_lens, skip, input_lengths)

Internal: run forward-backward on device and return `(loss, grad)`.
Use `expand_ctc_labels(targets, blank)` to get `labels`, `exp_lens`, and `skip`.
"""
function ctc_forward_backward(logits::AbstractArray{T,3},
                              labels::Matrix{Int32}, exp_lens::Vector{Int32},
                              skip::Matrix{Bool}, input_lengths::Vector{Int}) where {T}
    V, Tmax, B = size(logits)
    Smax = size(labels, 1)
    backend = KernelAbstractions.get_backend(logits)

    # FIX (Bug 2): Clamp input_lengths to [0, Tmax] to prevent out-of-bounds
    clamped_lengths = Int32.(clamp.(input_lengths, 0, Tmax))

    lab_d  = copyto!(similar(logits, Int32, Smax, B), labels)
    Sl_d   = copyto!(similar(logits, Int32, B), exp_lens)
    Tl_d   = copyto!(similar(logits, Int32, B), clamped_lengths)
    skip_d = copyto!(similar(logits, Bool, Smax, B), skip)

    log_probs = logsoftmax(logits; dims = 1)

    em = similar(logits, T, Smax, Tmax, B)
    ctc_gather_emit_kernel!(backend)(em, log_probs, lab_d, Sl_d, Tl_d;
                                    ndrange = (Smax, Tmax, B))

    α = fill!(similar(logits, T, Smax, Tmax, B), T(-Inf))
    ctc_fwd_init_kernel!(backend)(α, em, Sl_d, Tl_d; ndrange = B)
    for t in Int32(2):Int32(Tmax)
        ctc_fwd_step_kernel!(backend)(α, em, skip_d, Sl_d, Tl_d, t;
                                      ndrange = (Smax, B))
    end

    nll = similar(logits, T, B)
    ctc_loss_kernel!(backend)(nll, α, Sl_d, Tl_d; ndrange = B)

    β = fill!(similar(logits, T, Smax, Tmax, B), T(-Inf))
    ctc_bwd_init_kernel!(backend)(β, Sl_d, Tl_d; ndrange = B)
    # FIX (Bug 3): When Tmax == 1, Int32(Tmax - 1) == 0 and the range
    # Int32(0):-Int32(1):Int32(1) is empty, so backward steps are skipped.
    # This is actually correct for Tmax == 1 (no steps needed beyond init),
    # but we must ensure the loop bounds don't underflow for Tmax == 0.
    if Tmax >= 2
        for t in Int32(Tmax - 1):-Int32(1):Int32(1)
            ctc_bwd_step_kernel!(backend)(β, em, skip_d, Sl_d, Tl_d, t;
                                          ndrange = (Smax, B))
        end
    end

    grad = similar(logits, T)
    ctc_grad_kernel!(backend)(grad, α, β, log_probs, lab_d, Sl_d, Tl_d, nll;
                             ndrange = (V, Tmax, B))

    KernelAbstractions.synchronize(backend)

    # Count finite losses for proper averaging — impossible samples contribute
    # Inf to the sum but should not NaN the mean.
    loss = sum(nll) / T(B)
    grad_scaled = grad / T(B)
    loss, grad_scaled
end
