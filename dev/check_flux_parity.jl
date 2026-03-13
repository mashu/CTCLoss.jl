#!/usr/bin/env julia
# Optional parity check vs Flux.ctc_loss. Not part of CI; run locally when needed.
#
# Flux.ctc_loss(ŷ, y): ŷ = (classes, time) = (C, T), y = 1D labels. Applies logsoftmax to ŷ;
# blank = last class = size(ŷ, 1). Single sample only.
# We compare: same (C, T) logits and same labels; Flux.ctc_loss(ŷ, y) vs
# CTCLoss.ctc_loss_batched(ŷ with batch dim, [y], [T]; blank=C). Both use blank=last.
#
# Fair: one RNG seed, same logits and labels for both. With B=1 we should get identical loss
# (and gradient w.r.t. logits, if we check).
#
# Run from repo root: julia --project=dev dev/check_flux_parity.jl
# First time: julia --project=dev -e 'using Pkg; Pkg.instantiate()'

using LinearAlgebra
using Random

push!(LOAD_PATH, dirname(@__DIR__))
using CTCLoss
using Flux
using Zygote

function run_parity(; C = 20, T = 40, seed = 42)
    Random.seed!(seed)

    # Single sample: (C, T) for Flux, (C, T, 1) for us. Same numeric data.
    logits_CT = randn(Float64, C, T)
    labels_1d = rand(1:(C - 1), rand(1:min(8, T ÷ 2)))  # no blank in labels; 1-based
    blank = C  # last class in both Flux and us

    # Flux: (C, T) and 1D labels; applies logsoftmax internally
    loss_flux = Flux.ctc_loss(logits_CT, labels_1d)

    # Ours: (C, T, 1), [labels], [T], blank=C
    logits_batched = reshape(logits_CT, C, T, 1)
    targets_batched = [copy(labels_1d)]
    input_lengths = [T]
    loss_ours = CTCLoss.ctc_loss_batched(logits_batched, targets_batched, input_lengths, blank)

    loss_ok = isapprox(loss_flux, loss_ours; rtol = 1e-9, atol = 1e-8)

    # Gradient: Flux via Zygote, ours via rrule
    grad_flux = Zygote.gradient(x -> Flux.ctc_loss(x, labels_1d), logits_CT)[1]
    grad_ours_all = Zygote.gradient(
        x -> CTCLoss.ctc_loss_batched(x, targets_batched, input_lengths, blank),
        logits_batched,
    )[1]
    grad_ours = grad_ours_all[:, :, 1]
    grad_ok = isapprox(grad_flux, grad_ours; rtol = 1e-8, atol = 1e-7)

    println("Seed = ", seed, ", C = ", C, ", T = ", T)
    println("  Loss  Flux ", loss_flux, "  CTCLoss ", loss_ours, "  match: ", loss_ok)
    println("  Grad  Flux norm ", norm(grad_flux), "  CTCLoss norm ", norm(grad_ours), "  match: ", grad_ok)
    if !loss_ok || !grad_ok
        println("  Loss diff: ", abs(loss_flux - loss_ours))
        println("  Grad max diff: ", maximum(abs.(grad_flux .- grad_ours)))
    end
    return loss_ok && grad_ok
end

function main()
    println("CTCLoss.jl vs Flux.ctc_loss parity check (single sample, optional, not in CI)\n")
    ok = true
    for seed in [0, 1, 123]
        ok &= run_parity(; seed = seed)
    end
    println(ok ? "\nParity check passed." : "\nParity check failed.")
    exit(ok ? 0 : 1)
end

main()
