# Numerically stable log(exp(a) + exp(b)); used in CTC forward-backward.

"""
    logaddexp(a, b)

Numerically stable `log(exp(a) + exp(b))`. Internal use only; not exported to avoid
clashing with `LogExpFunctions.logaddexp`. Use `CTCLoss.logaddexp` if needed.
"""
function logaddexp(a::T, b::T) where {T<:AbstractFloat}
    m = max(a, b)
    m == T(-Inf) && return T(-Inf)
    m + log(exp(a - m) + exp(b - m))
end
