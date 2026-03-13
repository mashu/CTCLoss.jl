# API

```@docs
CTCLoss
```

## Loss and decoding

```@docs
CTCLoss.ctc_loss_batched
CTCLoss.ctc_greedy_decode
```

## Helpers

```@docs
CTCLoss.expand_ctc_labels
```

## Internal (advanced)

```@docs
CTCLoss.ctc_forward_backward
CTCLoss.logaddexp
```

## Summary

| Function | Description |
|----------|-------------|
| `ctc_loss_batched(logits, targets, input_lengths [; blank])` | Scalar CTC loss (mean over batch). |
| `ctc_greedy_decode(logits, input_lengths [; blank])` | Argmax per frame, collapse repeats, drop blanks. |
| `expand_ctc_labels(targets, blank)` | Expand label sequences with blanks for the CTC lattice. |
| `ctc_forward_backward(logits, labels, exp_lens, skip, input_lengths)` | Internal: returns `(loss, grad)` (use labels from `expand_ctc_labels`). |
