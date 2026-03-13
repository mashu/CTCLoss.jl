## PyTorch parity check

`check_pytorch_parity.jl` compares CTCLoss.jl with PyTorch’s `torch.nn.functional.ctc_loss` on **the same inputs** (same logits, same targets, same lengths) and checks that:

1. **Loss (our reduction)** — mean over batch of per-sample NLL: Julia and PyTorch (`reduction="none"` then `.mean()`) agree.
2. **Loss (PyTorch’s default reduction)** — mean over batch of `loss_i / target_lengths_i`: we compute the same in Julia (per-sample loss then that formula) and compare to PyTorch `reduction="mean"`.
3. **Gradient** — gradient of the mean loss w.r.t. log-probabilities: same layout (after permuting PyTorch’s (T,B,C) to our (C,T,B)) and same scaling, so we compare element-wise.

The comparison: one shared RNG seed, one set of logits and targets; both sides get the same numeric inputs (log-probs are derived from the same logits via the same `logsoftmax`). Blank index is aligned (Julia blank=1 ↔ PyTorch blank=0).

### How to run

From the **repo folder**:

```bash
# First time only: install dev environment (adds PyCall)
julia --project=dev -e 'using Pkg; Pkg.instantiate()'

# Run the parity check
julia --project=dev dev/check_pytorch_parity.jl
```

**Requirements:** Python with `torch` available to PyCall (e.g. `using Conda; Conda.add("pytorch")` then rebuild PyCall if needed). Not required for normal use of CTCLoss.jl.

### Last check

```
CTCLoss.jl vs PyTorch CTC parity check (optional, not in CI)

Seed = 0, C = 20, T = 30, B = 4
  Loss (our reduction):  Julia 90.74824989755635  PyTorch 90.74824989755635  match: true
  Loss (PyTorch reduction): Julia 49.64327226322134  PyTorch 49.64327226322134  match: true
  Grad  Julia norm: 2.093328086197603  PyTorch norm: 2.09332808619759  match: true
Seed = 1, C = 20, T = 30, B = 4
  Loss (our reduction):  Julia 84.79283077402579  PyTorch 84.79283077402579  match: true
  Loss (PyTorch reduction): Julia 55.22955852772782  PyTorch 55.22955852772782  match: true
  Grad  Julia norm: 2.2627857961094624  PyTorch norm: 2.2627857961094664  match: true
Seed = 123, C = 20, T = 30, B = 4
  Loss (our reduction):  Julia 82.65334110971182  PyTorch 82.65334110971182  match: true
  Loss (PyTorch reduction): Julia 53.059622937939416  PyTorch 53.059622937939416  match: true
  Grad  Julia norm: 2.1488481497732037  PyTorch norm: 2.1488481497731975  match: true

Parity check passed.
```
