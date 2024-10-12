# Simple Euler Equation JAX benchmarking

Philip Mocz (2024)

Flatiron Institute

Benchmarking on `macbook` (Apple M3 Max) and `rusty` (Nvidia A100)

## Files

* `euler.py` simple JAX version on single node
* `euler_distributed.py` simple numpy version
* `euler_numpy.py` JAX version for distributed systems

## Strong Scaling on `macbook`:

![strong scaling](results/scaling_strong.png)

## Weak Scaling on `rusty`:

![weak scaling](results/scaling_weak.png)

## Final Simulation Result

16384^2 resolution JAX (single-precision) simulation after 277300 iterations on 16 GPUs in 64.1 minutes

(for reference, my macbook run (single-precision) at 1024^2 resolution after 15426 iterations took 4.6 minutes)

The GPU calculations had a throughput (mcups) 335x more!

![final snapshot](results/result_16384_single.png)
