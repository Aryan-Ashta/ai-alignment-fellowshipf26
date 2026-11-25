# Gate 0.5 memory

Peaks are `torch.mps.driver_allocated_memory()` minus the reading before load.

- model load: 15.61 GB
- forward+backward batch 8×128: 24.24 GB
- forward 1×16384: 29.64 GB
