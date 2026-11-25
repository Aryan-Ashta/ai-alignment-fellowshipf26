# Gate 0.4 MPS attention

Reference: CPU fp32 `attn_implementation='sdpa'`.
Pass: every final-block-output cosine ≥ 0.995 and next-token argmax agreement ≥ 99%.
Chunk size 1024.

Eager single-forward skip log:
- 4096: eager single forward not in the candidate list; score bytes 1073741824 ≤ 24 GB
- 8192: eager single forward not in the candidate list; score bytes 4294967296 ≤ 24 GB
- 16384: eager single forward not in the candidate list; score bytes 17179869184 ≤ 24 GB

- path a 4096: min_cos=0.99247 argmax=0.9880 pass=False
- path b 4096: min_cos=0.99247 argmax=0.9880 pass=False
- path c 4096: min_cos=0.99009 argmax=0.9883 pass=False

STOP: no path passed at 8192.
512-token spot check path a: min_cos=0.99892 argmax=0.9941 pass=True
