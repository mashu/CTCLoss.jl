# CTCLoss.jl

[![CI](https://github.com/mashu/CTCLoss.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/CTCLoss.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/CTCLoss.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/CTCLoss.jl)
[![Documentation](https://img.shields.io/badge/docs-blue.svg)](https://mashu.github.io/CTCLoss.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Batched **Connectionist Temporal Classification (CTC)** loss and greedy decoding for Julia, with GPU support via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). Differentiable via [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) (Zygote-compatible).

```julia
using CTCLoss, Zygote

logits = randn(Float32, 51, 100, 4)   # (vocab+blank, time, batch)
targets = [[1, 2, 3], [4, 5], [1], [2, 3, 4, 5]]
input_lengths = fill(100, 4)

loss = CTCLoss.ctc_loss_batched(logits, targets, input_lengths)
grad = Zygote.gradient(l -> CTCLoss.ctc_loss_batched(l, targets, input_lengths), logits)[1]
decoded = CTCLoss.ctc_greedy_decode(logits, input_lengths)
```
