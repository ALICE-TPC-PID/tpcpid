# O2Physics integration CI

## Local CCDB fixture

Generate and commit `run/ci/o2physics/ccdb/ccdb.tar.gz` before running CI. In an authenticated
local O2Physics shell, run `python3 run/ci/o2physics/ccdb/fetch_ccdb.py` from the repository
root. The script runs the workflow, captures its CCDB objects and headers, and
packages them with checksums and input provenance. See [CCDB instructions](o2physics/ccdb/README.md)
for input overrides, regeneration, GitHub upload and snapshot limitations.

## Pipeline

The `CI` GitHub Actions workflow trains MEAN, SIGMA, and FULL networks for **20 epochs each**. It exports the networks from that exact run as the `ci-networks` artifact. A dependent GitHub-hosted Ubuntu job mounts `alice.cern.ch` with CVMFS, selects the newest `daily-YYYYMMDD-HHMM-revision` O2Physics build, and runs it in the current CVMFS EL9 container. CVMFS fetches the container and package files on demand; no private image or login is required.

The pipeline in `o2physics/run.sh` is adapted from `/scratch/alice/csonnab/MyO2/misc/test-mlfix/master/run.sh`. `o2physics/AO2D.2dfs.root` and `o2physics/configuration.json` are copies of the supplied inputs. The ROOT fixture is approximately 85 MiB and is included directly so GitHub runners do not depend on hydra or grid access. The original files are not changed.

Each inference run gets a generated configuration with automatic network fetching disabled, the network timestamp set to zero, and an explicit local CI model path. The MEAN model tests mean correction; the FULL model tests mean and sigma correction. The SIGMA model is an intermediate model used to train FULL, not a standalone O2 PID correction.

The runner passes only an allowlisted environment, preserves HTTP(S) proxy settings, and uses isolated home, temporary, and IPC namespaces. It does not source or mount AliEn tokens. The workflow script contains no proxy unsets. All pipeline stages are covered by `pipefail`. The shared-memory segment is 2 GB for the small fixture (instead of the reference script's 7.5 GB).

Failures retain logs, generated configurations, resolved container/package details, and any ROOT outputs as the `o2physics-ci` artifact. Success requires a zero pipeline status, a log confirmation that the local model loaded, and nonempty ROOT output. This is an integration smoke test, not a physics-quality criterion.

To reproduce from the repository root in an environment with the training dependencies:

```bash
python3 run/ci/ci_test.py --artifact-dir output/my-ci-networks
python3 run/ci/o2physics/o2physics_pid_test.py \
  --artifact-dir output/my-ci-networks \
  --output-dir output/my-o2physics-test
```

Both output directories must be new. Training uses a unique run directory to avoid overwriting previous results. The O2Physics runner needs Apptainer or Singularity and mounted ALICE CVMFS. Use `--package O2Physics/daily-YYYYMMDD-HHMM-revision` to reproduce a specific build recorded in `environment.json`.

Lightweight checks:

```bash
python3 -m unittest discover -s tests -p test_o2physics_ci.py -v
bash -n run/ci/o2physics/run.sh
```

Setup references: [CVMFS GitHub action](https://github.com/cvmfs-contrib/github-action-cvmfs), [Apptainer Ubuntu installation](https://apptainer.org/docs/admin/latest/installation.html#install-ubuntu-packages).
