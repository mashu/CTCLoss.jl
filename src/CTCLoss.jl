"""
    CTCLoss

Batched Connectionist Temporal Classification (CTC) loss and greedy decoding for Julia,
with GPU support via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl).
Gradients are provided through [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
`rrule`s, so [Zygote](https://github.com/FluxML/Zygote.jl) and other AD systems work
without tracing through the kernels.

Convention: blank token index defaults to last class, i.e. `size(logits, 1)`.
All API functions accept a `blank` keyword or argument to override.
"""
module CTCLoss

using KernelAbstractions
using ChainRulesCore: ChainRulesCore, NoTangent
using NNlib: logsoftmax

export expand_ctc_labels,
       ctc_forward_backward,
       ctc_loss_batched,
       ctc_greedy_decode

include("utils.jl")
include("labels.jl")
include("kernels.jl")
include("forward_backward.jl")
include("api.jl")

end
