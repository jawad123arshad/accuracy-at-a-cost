## Practical guidance (auto-generated from this run's results)

- **[default_hparams] Best average positive transfer under multitask training:** architecture `Transformer` (mean Delta = 0.0080).
- **[param_matched] Best average positive transfer under multitask training:** architecture `Transformer` (mean Delta = 0.0099).
- **[default_hparams] Best accuracy-per-GFLOP on `problem_cls`:** `Transformer`.
- **[default_hparams] Best accuracy-per-GFLOP on `verdict`:** `Transformer`.
- **[default_hparams] Best accuracy-per-GFLOP on `runtime_bucket`:** `Transformer`.
- **[param_matched] Best accuracy-per-GFLOP on `problem_cls`:** `Transformer`.
- **[param_matched] Best accuracy-per-GFLOP on `verdict`:** `Transformer`.
- **[param_matched] Best accuracy-per-GFLOP on `runtime_bucket`:** `Transformer`.
- **Cheapest architecture at seq_len=32:** `Transformer`.
- **Cheapest architecture at seq_len=64:** `Transformer`.
- **Cheapest architecture at seq_len=128:** `Transformer`.
- **Cheapest architecture at seq_len=256:** `Transformer`.
- **Cheapest architecture at seq_len=512:** `Transformer`.

- Results aggregated over 3 seeds per (architecture, mode, task); see `c1_aggregated.csv` for 95% CIs and `c1_raw_runs.csv` for the full audit trail.

- These conclusions are drawn from a pilot of 200 problems — re-run at larger scale (more problems, more tasks) before treating as a final publication result.