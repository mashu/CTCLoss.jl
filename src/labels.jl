# CTC label expansion: interleave blanks for the forward-backward lattice (CPU-side).

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
