"""
    CTCLoss

Batched Connectionist Temporal Classification (CTC) loss and greedy decoding,
GPU-native via KernelAbstractions. Forward-backward runs on-device; gradients
provided via ChainRulesCore.rrule (Zygote-compatible).

Convention: blank token index defaults to last class, i.e. `size(logits, 1)`.
All API functions accept `blank` keyword to override.
"""
module CTCLoss

using KernelAbstractions
using ChainRulesCore: ChainRulesCore, NoTangent
using NNlib: logsoftmax

export logaddexp,
       expand_ctc_labels,
       ctc_forward_backward,
       ctc_loss_batched,
       ctc_greedy_decode

include("ctc.jl")

end
