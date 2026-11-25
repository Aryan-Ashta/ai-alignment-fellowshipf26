# 32B cost estimate

Throughput ratio H200/M5 = 990/30.0 = 33.0.
H200 bf16 TFLOPS is the published dense peak, not a measurement on this code.
Fit scale = (32/7) * (5120/4096) / 33.0.
Generation uses 1050 tok/s.

- 7B fit hours: 4.0
- 32B fit hours (one checkpoint): 0.6926406926406925
- 32B generation hours: 0.26455026455026454

Dollars omitted: pass --usd-per-gpu-hour.
