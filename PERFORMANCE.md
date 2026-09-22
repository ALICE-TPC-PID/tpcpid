# Training memory and efficiency audit

Changes are uncommitted in `/lustre/alice/users/csonnab/TPC/o2-tpc-pid` on Hydra. Session 3 was used through its MCP `summarize_text` tool to summarize the training and surrounding pipeline; findings were checked against source and tests.

## Main causes and fixes

- **Per-row Python/tensor objects:** `list(zip(X, y))` created two tensor views and a tuple for every observation. Replaced with two tensor arrays and vectorized batch indexing. Sequential batches use views; shuffling keeps one int64 permutation rather than millions of Python indices. Memory reporting now includes every feature/output element.
- **Accidental CUDA allocation:** CPU loading still invoked scalers that moved full arrays to CUDA before moving them back. Scaling now stays on CPU during preparation; float32 NumPy arrays can share storage with tensors. GPU-resident loading remains an explicit option. CUDA device selection happens before optional GPU data loading.
- **Unbounded validation:** validation previously used the entire validation dataset in one batch with autograd enabled. It now uses `validation_batch_size=65536`, no gradients, proper evaluation mode, and sample-weighted metrics. Training metrics are detached immediately, so epoch accumulation cannot retain backward graph structures.
- **Unbounded target generation and QA:** SIGMA/FULL teacher predictions and ONNX QA now run in bounded batches. Float32 training data, early release of source/split arrays, and preallocated prediction outputs reduce simultaneous copies. QA respects allocated CPU threads.
- **ROOT loading:** selected scalar branches are read in chunks directly into a preallocated array, avoiding intermediate DataFrames and concatenation for the normal single-file path. Entry limits apply during reading; TTree and RNTuple output are supported. Files close deterministically and array caching is disabled. Requested feature order is honored and missing columns fail clearly.
- **Distributed correctness:** every process uses the same train/test split (default seed 42, explicit configured seed preserved). Validation shards contain each observation once; losses are reduced across ranks before scheduling. Removed per-batch barriers and repeated loader creation. Failures are no longer silently converted into independent single-GPU training jobs.
- **Job launch:** respect configured CPUs per GPU task, resolve the master host outside containers for multi-node jobs, bind the correct visible GPU, stop peer tasks on failure, pass `--rocm` for AMD containers, reject unsupported partial >8-GPU node counts.
- **Other verified bugs:** standard `MSELoss` works without a `weights` argument; integer weights and multi-output weighted MSE work while preserving the existing squared-weight objective; `nsamples` is now an exact batch limit; batch schedule validation catches malformed schedules; checkpoint and ONNX export do not move the live model off its device; ONNX validation failures propagate; export explicitly retains opset 14 behavior; TorchScript method no longer shadows itself. QA accepts boolean configuration values and no longer mutates shared fit defaults. Fixed an existing indentation error in the legacy conversion script.

## Measurements on Hydra

PyTorch 2.9.0+cu130 and uproot 5.7.2 in the configured CUDA container. Synthetic inputs; these numbers are not a production end-to-end benchmark.

| 200,000 rows, 7 float32 features + 1 target | Previous dataset/collation | New dataset/collation |
|---|---:|---:|
| Construction | 1.154 s | 0.00762 s |
| Iterate all rows, batches of 512 | 0.202 s | 0.0605 s |
| Increase in peak process RSS | 276.4 MiB | 6.55 MiB |

Separate one-CPU Slurm debug allocations; the old path reproduces the original tensor copies, row list, and DataLoader collation. Construction was approximately 151x faster and iteration 3.3x faster in these runs. These factors do not predict total training speedups.

**20,000,000 synthetic rows, two H200 GPUs, two epochs:** a 7-input, 12-wide, 10-hidden-layer ReLU model with the configured initial batch size of 262144 completed successfully. Each rank stores the full tensor dataset and trains its shard. Epochs took 1.97 s and 1.06 s; peak allocated GPU tensor memory was 218 MiB per rank; peak host RSS was 2490/2477 MiB per rank. Both ranks reported identical losses and parameter sums. This excludes ROOT reading, train/test copying, fitting scalers, and physical-data convergence. Slurm job 32498.

## Validation and reproduction

- `python3 -m unittest discover -s tests -v`: **10 tests passed**, covering tensor storage and CPU flag, shuffling and padded/unpadded rank coverage, scaler fit isolation, validation batch limits and train/eval modes, sample-weighted metrics, batch limits, weighted multi-output loss, worker loading, bounded ONNX inference, ROOT formats/order/limits/wildcards, dtype promotion and latest cycles.
- `python3 tests/smoke_training_pipeline.py`: isolated synthetic ROOT -> MEAN -> SIGMA -> FULL; checkpoint reloads, ONNX opset 14 and variable-batch inference, generated Slurm script syntax. Uses temporary outputs, not production networks.
- Two GPU tasks running `python3 tests/smoke_distributed_training.py`: unequal validation batch counts against an independently computed full-set metric, plus non-mutating checkpoint/ONNX export.
- `python3 tests/benchmark_training_memory.py --rows 200000 [--legacy]`: small loader comparison. `--rows 20000000 --train` runs the large synthetic training check under an appropriate GPU allocation.
- Python compilation and `git diff --check`.

## Operational notes and limits

Regenerate `TRAIN.sh` through the existing job creation flow to pick up launch fixes. Retrain the full MEAN/SIGMA/FULL chain together: feature ordering, repeatable data splits, epoch shuffling, and corrected validation metrics can change training trajectories. Existing checkpoints contain no feature-order metadata, so a checkpoint trained with a different ROOT branch order cannot be automatically reconciled.

Keep `DATA_LOADER.copy_to_device=False` and start with `num_workers=0`; vectorized tensor batching removes the usual need for many loader workers. `validation_batch_size` is an independent `NET_TRAINING` option. No mixed precision or model/physics changes were enabled. CPU thread counts and additional GPUs still need workload-specific profiling; eight GPUs are not automatically more efficient for such small networks.

The dataset and shuffle indices still scale linearly with row count, and each distributed rank reads/stores its own copy. This is substantially smaller than the previous object-heavy representation, but it is not a fully streaming/memory-mapped pipeline. Optional sklearn scaler fitting can still allocate full-size temporary arrays. Multiple wildcard ROOT files are concatenated once and can temporarily use roughly twice their combined output storage. Multi-node and AMD execution were not available in this validation; post-training plotting and full production data preparation were not run. No historical OOM job log was identified, so causes are established from code and controlled reproduction rather than attribution to one past job.
