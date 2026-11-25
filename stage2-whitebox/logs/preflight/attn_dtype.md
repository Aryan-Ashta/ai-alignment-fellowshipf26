# Gate 0.4 dtype split

Same WikiText prefix as `00_attn.py`. SDPA only. Median cosine stays ≥ 0.99992 in every cell. The every-position 0.995 bar fails for CPU bf16 versus CPU fp32, so the miss is not an MPS-only collapse.

| comparison | length | min cosine | median cosine | fraction < 0.995 | argmax agreement |
|---|---:|---:|---:|---:|---:|
| MPS bf16 vs CPU fp32 | 4096 | 0.99247 | 0.99995 | 0.0005 | 0.9880 |
| CPU bf16 vs CPU fp32 | 4096 | 0.98934 | 0.99995 | 0.0002 | 0.9873 |
| MPS bf16 vs CPU bf16 | 4096 | 0.97980 | 0.99992 | 0.0010 | 0.9905 |
| MPS bf16 vs CPU fp32 | 8192 | 0.97157 | 0.99994 | 0.0007 | 0.9877 |
| CPU bf16 vs CPU fp32 | 8192 | 0.95573 | 0.99995 | 0.0005 | 0.9874 |
| MPS bf16 vs CPU bf16 | 8192 | 0.93364 | 0.99992 | 0.0012 | 0.9893 |

512-token spot check (MPS bf16 sdpa vs CPU fp32) passed: min cosine 0.99892, argmax 0.9941.

No candidate passed the written CPU-fp32 bar at 8192. Path (a) and path (b) missed at 4096 by the same amount. Stop and ask before freezing `configs/gen.yaml: long_seq_attn`.
