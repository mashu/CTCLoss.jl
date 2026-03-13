#!/usr/bin/env julia
# Optional parity check vs PyTorch CTC loss. Not part of CI; run locally when needed.
#
# Fair comparison: same RNG seed, same logits and targets for both. Log-probs are computed
# once in Julia (logsoftmax(logits)) and the same array is passed to PyTorch (permuted to
# (T,N,C)); no separate random data in PyTorch. Blank: Julia blank=1 ↔ PyTorch blank=0.
#
# We check:
# 1. Loss with our reduction (mean over batch of NLL) vs PyTorch reduction='none' then .mean().
# 2. Loss with PyTorch reduction (mean over batch of NLL_i/target_lengths_i) vs same in Julia.
# 3. Gradient d(mean_loss)/d(log_probs): same scaling (1/B), layout permuted (T,B,C)→(C,T,B).
#
# Run from repo root: julia --project=dev dev/check_pytorch_parity.jl
# First time: julia --project=dev -e 'using Pkg; Pkg.instantiate()'

using LinearAlgebra
using Random

# Load CTCLoss from parent repo (dev env only has PyCall)
push!(LOAD_PATH, dirname(@__DIR__))
using CTCLoss
using NNlib: logsoftmax
using PyCall

try
    global torch = pyimport("torch")
    global F = pyimport("torch.nn.functional")
catch e
    @error "Could not import torch. Run from repo root: julia --project=dev dev/check_pytorch_parity.jl" exception = e
    exit(1)
end

function run_parity(; C = 20, T = 30, B = 4, seed = 42)
    Random.seed!(seed)
    torch.manual_seed(seed)

    # Shared data: same logits for both. Julia uses (C, T, B), PyTorch (T, B, C).
    logits_jl = randn(Float64, C, T, B)

    # Targets: 1-based for Julia (no blank index in labels). Blank = 1 in Julia (= PyTorch 0).
    # So valid labels in Julia are 2:C; in PyTorch 1:(C-1).
    targets_jl = [rand(2:C, rand(1:min(5, T ÷ 2))) for _ in 1:B]
    input_lengths = fill(T, B)

    # Julia: blank = 1 (first class) to match PyTorch blank = 0
    blank_jl = 1
    loss_jl = CTCLoss.ctc_loss_batched(logits_jl, targets_jl, input_lengths, blank_jl)
    _, grad_jl = CTCLoss.ctc_forward_backward(
        logits_jl,
        CTCLoss.expand_ctc_labels(targets_jl, blank_jl)...,
        input_lengths,
    )

    # PyTorch: log_probs (T, N, C). Use precomputed log_probs so the tensor is a leaf and .grad is populated.
    log_probs_jl = logsoftmax(logits_jl; dims = 1)
    log_probs_pt = torch.tensor(
        permutedims(log_probs_jl, (2, 3, 1)),
        dtype = torch.float64,
        requires_grad = true,
    )

    # Targets: 0-based, padded. PyTorch expects (N, S) with S = max target length.
    target_lengths_list = [length(t) for t in targets_jl]
    S = maximum(target_lengths_list)
    targets_pt = torch.zeros((B, S), dtype = torch.int64)
    for b in 1:B
        t = targets_jl[b]
        for (i, lbl) in enumerate(t)
            targets_pt[b, i] = lbl - 1   # 1-based indices in Julia; label 0-based for PyTorch
        end
    end
    input_lengths_pt = torch.full((B,), T, dtype = torch.int64)
    target_lengths_pt = torch.tensor(target_lengths_list, dtype = torch.int64)

    # --- Same reduction as ours: mean over batch (no target-length norm) ---
    loss_per_item = F.ctc_loss(
        log_probs_pt,
        targets_pt,
        input_lengths_pt,
        target_lengths_pt;
        blank = 0,
        reduction = "none",
    )
    loss_pt_simple_mean = loss_per_item.mean().item()
    # PyTorch default reduction (mean over batch of loss_i / target_lengths_i) for parity check
    loss_pt_default = F.ctc_loss(
        log_probs_pt,
        targets_pt,
        input_lengths_pt,
        target_lengths_pt;
        blank = 0,
        reduction = "mean",
    ).item()
    loss_per_item.mean().backward()
    grad_pt = log_probs_pt.grad.numpy()

    # Same layout + same reduction => numerical parity. Julia: per-sample NLL then (1/B)*sum(nll_i / target_lengths_i)
    loss_per_sample_jl = [
        CTCLoss.ctc_loss_batched(
            logits_jl[:, :, b:b],
            [targets_jl[b]],
            [input_lengths[b]];
            blank = blank_jl,
        ) for b in 1:B
    ]
    target_lengths_jl = [length(t) for t in targets_jl]
    loss_jl_pytorch_reduction = sum(loss_per_sample_jl[b] / target_lengths_jl[b] for b in 1:B) / B

    # Both gradients are d(mean_loss)/d(log_probs). Julia (C,T,B), PyTorch (T,B,C). Permute PyTorch -> (C,T,B).
    grad_pt_jl_layout = permutedims(grad_pt, (3, 1, 2))

    loss_ok = isapprox(loss_jl, loss_pt_simple_mean; rtol = 1e-9, atol = 1e-8)
    reduction_ok = isapprox(loss_jl_pytorch_reduction, loss_pt_default; rtol = 1e-9, atol = 1e-8)
    grad_ok = isapprox(grad_jl, grad_pt_jl_layout; rtol = 1e-8, atol = 1e-7)

    println("Seed = ", seed, ", C = ", C, ", T = ", T, ", B = ", B)
    println("  Loss (our reduction):  Julia ", loss_jl, "  PyTorch ", loss_pt_simple_mean, "  match: ", loss_ok)
    println("  Loss (PyTorch reduction): Julia ", loss_jl_pytorch_reduction, "  PyTorch ", loss_pt_default, "  match: ", reduction_ok)
    println("  Grad  Julia norm: ", norm(grad_jl), "  PyTorch norm: ", norm(grad_pt_jl_layout), "  match: ", grad_ok)
    if !loss_ok || !reduction_ok || !grad_ok
        println("  Loss diff (simple): ", abs(loss_jl - loss_pt_simple_mean))
        println("  Loss diff (PyTorch reduction): ", abs(loss_jl_pytorch_reduction - loss_pt_default))
        println("  Grad max diff: ", maximum(abs.(grad_jl .- grad_pt_jl_layout)))
    end
    return loss_ok && reduction_ok && grad_ok
end

function main()
    println("CTCLoss.jl vs PyTorch CTC parity check (optional, not in CI)\n")
    ok = true
    for seed in [0, 1, 123]
        ok &= run_parity(; seed = seed)
    end
    println(ok ? "\nParity check passed." : "\nParity check failed.")
    exit(ok ? 0 : 1)
end

main()
