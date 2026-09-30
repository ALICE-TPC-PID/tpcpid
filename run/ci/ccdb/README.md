# Generate the CI CCDB snapshot

Enter a local **O2Physics** environment with Python/PyROOT and your normal valid
AliEn token. From the repository root run:

```bash
python3 run/ci/ccdb/fetch_ccdb.py
```

This runs the complete `../o2physics/run.sh` pipeline on the repository AO2D with
the supplied configuration, capturing CCDB downloads in a fresh cache. There is
no fixed object list, run number, timestamp, or manual log parsing. The default
configuration fetches its discovery ONNX model from CCDB. If that model cannot
be fetched, provide an existing compatible model with `--network /path/model.onnx`.
CI always uses its own freshly trained models regardless of discovery's model.

Outputs, relative to this directory:

- `ccdb.tar.gz`: the GitHub-ready archive, containing `ccdb/manifest.json`, ROOT
  snapshots and cached HEAD responses (including run information when requested).
- `work/`: ignored scratch directory with `workflow.log`, generated configuration,
  analysis outputs, and the uncompressed `ccdb/` tree. It is never archived wholesale.

The script inherits local authentication to run the workflow; it does not copy
token files, environment dumps, logs, or analysis outputs into the archive.
Only captured `snapshot.root` and `header.json` files and the generated manifest
are packaged. A failing workflow or incomplete/invalid capture produces no final
archive. The default timeout is 1800 seconds (`--timeout` overrides it).

## Different input or output location

```bash
python3 run/ci/ccdb/fetch_ccdb.py /tmp/new-ccdb-capture \
  --aod /path/to/AO2D.root \
  --archive /tmp/new-ccdb.tar.gz
cp /tmp/new-ccdb.tar.gz run/ci/ccdb/ccdb.tar.gz
```

`--config` and `--workflow` override the repository defaults; custom workflows
must read `configuration.json` in their working directory, accept `O2_AOD_FILE`,
and produce the same two AnalysisResults ROOT outputs as the CI pipeline.
`--host` overrides the production CCDB URL used for configured CCDB endpoints.
O2 alone cannot execute these O2Physics tasks.

For a changed CI AO2D, replace `run/ci/o2physics/AO2D.2dfs.root` first, then generate
the archive from that file. The archive records SHA-256 hashes of the AO2D,
source configuration and workflow script; CI rejects mismatched inputs with a
regeneration message. Do not change the configuration or workflow after capture
without regenerating. Work directories and archive paths must be new: move old
ones aside or use the overrides when refreshing. The historical
`ccdb-manifest.json` is retained as a reference from `mltest.log`, not used as input.

## Publish to GitHub

The tools and `ccdb.tar.gz` are not ignored. Include them in your normal commit
and push together with the CI changes:

```bash
git add run/ci/ccdb/fetch_ccdb.py run/ci/ccdb/README.md run/ci/ccdb/ccdb.tar.gz
git add run/ci/o2physics/run.sh run/ci/o2physics_pid_test.py run/ci/README.md
git add .gitignore .github/workflows/ci.yml
```

For an archive of 100 MiB or larger, use Git LFS before adding it:

```bash
git lfs install
git lfs track 'run/ci/ccdb/ccdb.tar.gz'
git add .gitattributes run/ci/ccdb/ccdb.tar.gz
```

The O2Physics GitHub Actions checkout fetches LFS content. CI reads
`run/ci/ccdb/ccdb.tar.gz` by default (`--ccdb-archive` overrides it), validates
file checksums and input hashes, and gives each model run a separate writable
cache. Configured CCDB URLs become `file:///ccdb`; O2's local cache variables
are passed into the isolated container. CI does not receive your AliEn token.

## Scope and limitations

The capture covers the executed workflow, configuration and data. The current
O2 cache stores one version per object path and may ignore validity on reuse.
The generator therefore checks every recorded object request against the saved
validity interval and refuses to package a cache spanning incompatible intervals
or requests with unresolved nonpositive timestamps. Use a smaller AO2D fixture
in that case; this is not a multi-version CCDB server. Optional missing paths are
recorded in the manifest when the workflow completes successfully.

A newer O2Physics build can request new objects or change selection behavior;
regenerate using the corresponding local environment if CI reports missing
objects. No workflow execution or tests were performed when preparing these
scripts; the first authenticated capture and CI run still need to complete.
