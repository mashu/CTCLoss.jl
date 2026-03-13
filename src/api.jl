# Public API: batched CTC loss, greedy decode, and ChainRulesCore rrules.

"""
    ctc_loss_batched(logits, targets, input_lengths [; blank])
    ctc_loss_batched(logits, targets, input_lengths, blank)

Batched CTC loss (mean over batch). `logits`: `(V, time, batch)` raw logits.
`targets`: `Vector{Vector{Int}}` label sequences (no blank). `input_lengths`: valid
frames per sample. `blank` defaults to `size(logits, 1)` (last class).
"""
function ctc_loss_batched(logits::AbstractArray{T,3},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int};
                          blank::Int = size(logits, 1)) where {T}
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, _ = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    loss
end

function ctc_loss_batched(logits::AbstractArray{T,3},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int},
                          blank::Int) where {T}
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, _ = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    loss
end

function ChainRulesCore.rrule(::typeof(ctc_loss_batched),
                              logits::AbstractArray{T,3},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int}) where {T}
    blank = size(logits, 1)
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, grad = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    ctc_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent())
    loss, ctc_pullback
end

function ChainRulesCore.rrule(::typeof(ctc_loss_batched),
                              logits::AbstractArray{T,3},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              blank::Int) where {T}
    labels, el, skip = expand_ctc_labels(targets, blank)
    loss, grad = ctc_forward_backward(logits, labels, el, skip, input_lengths)
    ctc_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(), NoTangent())
    loss, ctc_pullback
end

"""
    ctc_greedy_decode(logits, input_lengths [; blank])

Greedy CTC decode: argmax per frame, collapse repeats, drop blanks.
Returns `Vector{Vector{Int}}` (one sequence per batch element).
Decoding runs on CPU; GPU arrays are materialized to CPU for this step.
"""
function ctc_greedy_decode(logits::AbstractArray{<:Real,3},
                          input_lengths::Vector{Int};
                          blank::Int = size(logits, 1))
    ctc_greedy_decode_cpu(Array(logits), input_lengths, blank)
end

function ctc_greedy_decode(logits::AbstractArray{<:Real,3},
                          input_lengths::Vector{Int},
                          blank::Int)
    ctc_greedy_decode_cpu(Array(logits), input_lengths, blank)
end

function ctc_greedy_decode_cpu(logits_cpu::Array{<:Real,3},
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
