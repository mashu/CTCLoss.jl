# CTCLoss.jl

Batched **Connectionist Temporal Classification (CTC)** loss and greedy decoding for Julia, with GPU support via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). Gradients are provided through a custom [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule`, so [Zygote](https://github.com/FluxML/Zygote.jl) and other AD systems work without tracing through the kernels.

## Features

- **Batched** CTC loss (no per-sample loops; one forward-backward over the batch).
- **Device-agnostic**: same code runs on CPU arrays and CUDA arrays (KernelAbstractions backend).
- **Differentiable**: use with `Zygote.gradient` or any ChainRules-compatible AD.
- **Stable**: log-space forward-backward; blank token index configurable (default: last class).

## Installation

```julia
using Pkg
Pkg.add("CTCLoss")
```

From the development repo:

```julia
Pkg.develop(url = "https://github.com/mashu/CTCLoss.jl")
```

## Quick start

- **Logits**: `(vocab_size_plus_blank, time, batch)` — raw logits before softmax. The blank token is the last class by default.
- **Targets**: `Vector{Vector{Int}}` — label indices (no blank in the sequences).
- **Input lengths**: `Vector{Int}` — number of valid time steps per batch element.

```julia
using CTCLoss
using Zygote

V = 51   # e.g. 50 labels + 1 blank
T = 100
B = 4
logits = randn(Float32, V, T, B)
targets = [[1, 2, 3], [4, 5], [1], [2, 3, 4, 5]]
input_lengths = fill(T, B)

loss = CTCLoss.ctc_loss_batched(logits, targets, input_lengths)
grad = Zygote.gradient(logits -> CTCLoss.ctc_loss_batched(logits, targets, input_lengths), logits)[1]
```

Custom blank index (e.g. blank = 51 when vocab size is 50):

```julia
blank = 51
loss = CTCLoss.ctc_loss_batched(logits, targets, input_lengths, blank)
# or
loss = CTCLoss.ctc_loss_batched(logits, targets, input_lengths; blank = blank)
```

Greedy decode:

```julia
decoded = CTCLoss.ctc_greedy_decode(logits, input_lengths; blank = 51)
# => Vector{Vector{Int}}, one sequence per batch element
```

## Dependencies

- [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [NNlib](https://github.com/FluxML/NNlib.jl) (for `logsoftmax`)

## License

MIT.
