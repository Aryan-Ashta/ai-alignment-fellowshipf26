# Gate 0.3 throughput

Best of 5 after 3 warm-up calls. bf16.

- 4096×4096×4096: torch 30.05 TFLOPS (4.6 ms), MLX 30.54 TFLOPS (4.5 ms), MLX/torch 1.02
- 8192×4096×4096: torch 29.31 TFLOPS (9.4 ms), MLX 31.01 TFLOPS (8.9 ms), MLX/torch 1.06
- 7B forward+backward batch 8×128, grads w.r.t. inputs only: 2.00 s

## Decision

The fitter stays torch-MPS. MLX is used only for CoT generation.
MLX matmul ≥ 2× torch on every shape: False.
A fitter port to MLX is not in this plan. Gate 3.4 still stops if a checkpoint fit projects past 12 h.
