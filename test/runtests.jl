using Test
using CTCLoss
using ChainRulesCore
import Zygote

const V = 51
const BLANK = 51

@testset "CTCLoss" begin

    @testset "logaddexp" begin
        @test CTCLoss.logaddexp(0.0f0, 0.0f0) ≈ log(2.0f0)
        @test CTCLoss.logaddexp(Float32(-Inf), 1.0f0) ≈ 1.0f0
        @test CTCLoss.logaddexp(1.0f0, Float32(-Inf)) ≈ 1.0f0
        @test CTCLoss.logaddexp(Float32(-Inf), Float32(-Inf)) == Float32(-Inf)
        @test CTCLoss.logaddexp(10.0f0, 10.0f0) ≈ 10.0f0 + log(2.0f0)
        @test CTCLoss.logaddexp(0.0, 0.0) ≈ log(2.0)
        @test typeof(CTCLoss.logaddexp(1.0f0, 2.0f0)) === Float32
        @test typeof(CTCLoss.logaddexp(1.0, 2.0)) === Float64
    end

    @testset "expand_ctc_labels" begin
        targets = [[1, 2, 3]]
        labels, el, skip = CTCLoss.expand_ctc_labels(targets, BLANK)
        @test el == Int32[7]
        @test labels[:, 1] == Int32[BLANK, 1, BLANK, 2, BLANK, 3, BLANK]
        @test !skip[1, 1]
        @test skip[4, 1]
        @test !skip[5, 1]
        @test skip[6, 1]

        targets_rep = [[1, 1]]
        labels_r, el_r, skip_r = CTCLoss.expand_ctc_labels(targets_rep, BLANK)
        @test el_r == Int32[5]
        @test labels_r[1:5, 1] == Int32[BLANK, 1, BLANK, 1, BLANK]
        @test !skip_r[4, 1]

        targets_batch = [[1, 2], [3, 4, 5]]
        labels_b, el_b, skip_b = CTCLoss.expand_ctc_labels(targets_batch, BLANK)
        @test el_b == Int32[5, 7]
        @test size(labels_b, 1) == 7
    end

    @testset "ctc_loss_batched forward" begin
        T_frames = 20
        B = 2
        logits = randn(Float32, V, T_frames, B)
        targets = [[1, 2, 3], [4, 5]]
        input_lengths = [T_frames, T_frames]

        loss = CTCLoss.ctc_loss_batched(logits, targets, input_lengths; blank = BLANK)
        @test isfinite(loss)
        @test loss > 0
        @test typeof(loss) <: Float32
    end

    @testset "ctc_loss_batched — impossible alignment" begin
        targets = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        T_frames = 3
        logits = randn(Float32, V, T_frames, 1)
        loss = CTCLoss.ctc_loss_batched(logits, targets, [T_frames]; blank = BLANK)
        @test loss == Inf32 || loss > 1e10
    end

    @testset "ctc_loss_batched — single label" begin
        targets = [[5]]
        logits = randn(Float32, V, 10, 1)
        loss = CTCLoss.ctc_loss_batched(logits, targets, [10]; blank = BLANK)
        @test isfinite(loss)
        @test loss > 0
    end

    @testset "ctc_loss_batched — loss decreases for peaked logits" begin
        targets = [[3]]
        T_frames = 10
        logits_random = zeros(Float32, V, T_frames, 1)
        loss_random = CTCLoss.ctc_loss_batched(logits_random, targets, [T_frames]; blank = BLANK)

        logits_peaked = fill(Float32(-10), V, T_frames, 1)
        for t in 1:T_frames
            if t == 5
                logits_peaked[3, t, 1] = 10.0f0
            else
                logits_peaked[BLANK, t, 1] = 10.0f0
            end
        end
        loss_peaked = CTCLoss.ctc_loss_batched(logits_peaked, targets, [T_frames]; blank = BLANK)
        @test loss_peaked < loss_random
    end

    @testset "ctc_loss_batched rrule" begin
        T_frames = 15
        B = 2
        logits = randn(Float32, V, T_frames, B)
        targets = [[1, 2], [3]]
        input_lengths = [T_frames, T_frames]

        loss_val, pullback = ChainRulesCore.rrule(CTCLoss.ctc_loss_batched, logits, targets, input_lengths)
        @test isfinite(loss_val)

        nt, grad_logits, nt2, nt3 = pullback(1.0f0)
        @test nt === NoTangent()
        @test nt2 === NoTangent()
        @test nt3 === NoTangent()
        @test size(grad_logits) == size(logits)
        @test all(isfinite, grad_logits)
    end

    @testset "ctc_loss_batched Zygote gradient" begin
        T_frames = 15
        B = 2
        logits = randn(Float32, V, T_frames, B)
        targets = [[1, 2], [3]]
        input_lengths = [T_frames, T_frames]

        grad = Zygote.gradient(logits) do x
            CTCLoss.ctc_loss_batched(x, targets, input_lengths, BLANK)
        end
        g = grad[1]
        @test g !== nothing
        @test size(g) == size(logits)
        @test all(isfinite, g)
        @test sum(abs, g) > 0
    end

    @testset "ctc_loss_batched gradient numerical check" begin
        T_frames = 8
        targets = [[2, 4]]
        input_lengths = [T_frames]
        logits = randn(Float64, V, T_frames, 1)

        _, pullback = ChainRulesCore.rrule(CTCLoss.ctc_loss_batched, logits, targets, input_lengths)
        _, analytic_grad, _, _ = pullback(1.0)

        ε = 1e-5
        n_checks = 20
        rng_idx = [(rand(1:V), rand(1:T_frames), 1) for _ in 1:n_checks]
        for (i, j, k) in rng_idx
            logits_plus = copy(logits)
            logits_plus[i, j, k] += ε
            logits_minus = copy(logits)
            logits_minus[i, j, k] -= ε
            fd = (CTCLoss.ctc_loss_batched(logits_plus, targets, input_lengths) -
                  CTCLoss.ctc_loss_batched(logits_minus, targets, input_lengths)) / (2ε)
            @test analytic_grad[i, j, k] ≈ fd atol = 1e-4 rtol = 1e-3
        end
    end

    @testset "ctc_forward_backward == ctc_loss_batched" begin
        T_frames = 12
        B = 3
        logits = randn(Float32, V, T_frames, B)
        targets = [[1], [2, 3], [4, 5, 6]]
        input_lengths = fill(T_frames, B)

        loss1 = CTCLoss.ctc_loss_batched(logits, targets, input_lengths; blank = BLANK)
        labels, el, skip = CTCLoss.expand_ctc_labels(targets, BLANK)
        loss2, grad = CTCLoss.ctc_forward_backward(logits, labels, el, skip, input_lengths)
        @test loss1 ≈ loss2
        @test size(grad) == size(logits)
    end

    @testset "ctc_greedy_decode" begin
        T_frames = 10
        logits = fill(Float32(-100), V, T_frames, 1)
        for t in [1, 2, 6, 9, 10]
            logits[BLANK, t, 1] = 100.0f0
        end
        for t in [3, 4, 5]
            logits[3, t, 1] = 100.0f0
        end
        for t in [7, 8]
            logits[5, t, 1] = 100.0f0
        end
        result = CTCLoss.ctc_greedy_decode(logits, [T_frames]; blank = BLANK)
        @test result[1] == [3, 5]
    end

    @testset "ctc_greedy_decode respects input_lengths" begin
        logits = fill(Float32(-100), V, 10, 1)
        for t in 1:3
            logits[2, t, 1] = 100.0f0
        end
        for t in 4:10
            logits[8, t, 1] = 100.0f0
        end
        result = CTCLoss.ctc_greedy_decode(logits, [5]; blank = BLANK)
        @test result[1] == [2, 8]
        result2 = CTCLoss.ctc_greedy_decode(logits, [3]; blank = BLANK)
        @test result2[1] == [2]
    end

    @testset "ctc_loss_batched Float64" begin
        logits = randn(Float64, V, 12, 1)
        targets = [[1, 2]]
        loss = CTCLoss.ctc_loss_batched(logits, targets, [12]; blank = BLANK)
        @test isfinite(loss)
        @test typeof(loss) <: Float64
    end

    @testset "ctc_loss_batched batch vs individual" begin
        T_frames = 15
        logits1 = randn(Float32, V, T_frames, 1)
        logits2 = randn(Float32, V, T_frames, 1)
        logits_both = cat(logits1, logits2; dims = 3)
        targets1 = [[1, 3]]
        targets2 = [[2, 4, 5]]
        loss1 = CTCLoss.ctc_loss_batched(logits1, targets1, [T_frames]; blank = BLANK)
        loss2 = CTCLoss.ctc_loss_batched(logits2, targets2, [T_frames]; blank = BLANK)
        loss_both = CTCLoss.ctc_loss_batched(logits_both, [targets1[1], targets2[1]], [T_frames, T_frames]; blank = BLANK)
        @test loss_both ≈ (loss1 + loss2) / 2 atol = 1e-5
    end

    @testset "default blank = last class" begin
        logits = randn(Float32, V, 10, 1)
        targets = [[1, 2]]
        loss_kw = CTCLoss.ctc_loss_batched(logits, targets, [10]; blank = V)
        loss_default = CTCLoss.ctc_loss_batched(logits, targets, [10])
        @test loss_kw ≈ loss_default
    end
end
