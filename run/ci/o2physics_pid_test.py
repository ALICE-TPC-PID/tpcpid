"""Exercise the current CI's mean and full ONNX models in the latest CVMFS O2Physics."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

CVMFS = Path('/cvmfs/alice.cern.ch')
CONTAINER = CVMFS / 'containers/fs/singularity/el9'
PACKAGES = CVMFS / 'el9-x86_64/Packages/O2Physics'
PROXIES = ('http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')


def latest_package(directory=PACKAGES):
    candidates = []
    for path in directory.iterdir():
        match = re.fullmatch(r'daily-(\d{8})-(\d{4})-(\d+)', path.name)
        if match and path.is_dir():
            candidates.append((tuple(map(int, match.groups())), path.name))
    if not candidates:
        raise RuntimeError(f'No O2Physics daily build found in {directory}')
    return 'O2Physics/' + max(candidates)[1]


def prepare_config(source, network):
    config = json.loads(source.read_text())
    pid = config['pid-tpc-service']
    pid.update({
        'pidTPC.autofetchNetworks': '0',
        'pidTPC.ccdb-timestamp': '0',
        'pidTPC.useNetworkCorrection': '1',
        'pidTPC.networkPathLocally': network,
    })
    return config


def clean_environment(home):
    # Allowlist, rather than inheriting tokens or APPTAINERENV/SINGULARITYENV overrides.
    env = {'PATH': os.defpath, 'HOME': str(home), 'TMPDIR': str(home),
           'LANG': 'C.UTF-8'}
    env.update({key: os.environ[key] for key in PROXIES if key in os.environ})
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).parent / 'o2physics')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for configurations, logs, and ROOT outputs')
    parser.add_argument('--package', help='Override automatic latest daily selection')
    args = parser.parse_args()
    artifacts = args.artifact_dir.resolve()
    fixtures = args.fixtures.resolve()
    output = args.output_dir.resolve()
    runtime = shutil.which('apptainer') or shutil.which('singularity')
    if not runtime:
        parser.error('Install Apptainer (or Singularity) first')
    for path in (CONTAINER, fixtures / 'AO2D.2dfs.root', fixtures / 'configuration.json',
                 fixtures / 'run.sh'):
        if not path.exists():
            parser.error(f'Missing required input: {path}')
    for mode in ('mean', 'full'):
        model = artifacts / 'networks' / f'network_{mode}' / f'net_onnx_{mode}.onnx'
        if not model.is_file():
            parser.error(f'Missing CI model: {model}')
    package = args.package or latest_package()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'environment.json').write_text(json.dumps({
        'package': package, 'container': str(CONTAINER.resolve()),
        'models': ['mean', 'full'],
    }, indent=2) + '\n')
    print(f'Using {package} in {CONTAINER.resolve()}', flush=True)
    for mode in ('mean', 'full'):
        work = output / mode
        work.mkdir()
        network = f'/models/network_{mode}/net_onnx_{mode}.onnx'
        config = prepare_config(fixtures / 'configuration.json', network)
        (work / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
        # Private /tmp and home prevent discovery of host AliEn credentials.
        with tempfile.TemporaryDirectory(prefix='o2physics-ci-') as scratch:
            command = [runtime, 'exec', '--cleanenv', '--containall',
                       '--no-mount', 'hostfs,cwd,bind-paths',
                       '--bind', '/cvmfs:/cvmfs:ro',
                       '--bind', f'{fixtures}:/fixtures:ro',
                       '--bind', f'{artifacts / "networks"}:/models:ro',
                       '--bind', f'{work}:/work', '--pwd', '/work']
            for key in PROXIES:
                if key in os.environ:
                    command.extend(['--env', f'{key}={os.environ[key]}'])
            command.extend([str(CONTAINER), '/bin/bash', '--noprofile', '--norc', '-c',
                            'set -eo pipefail; '
                            'environment=$(/cvmfs/alice.cern.ch/bin/alienv printenv "$1"); '
                            'eval "$environment"; exec bash /fixtures/run.sh',
                            'o2physics-ci', package])
            print(f'Running {mode} model; log: {work / "stdout.log"}', flush=True)
            with (work / 'stdout.log').open('w') as log:
                result = subprocess.run(command, env=clean_environment(scratch),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1200, cwd=scratch)
            if result.returncode:
                tail = (work / 'stdout.log').read_text(errors='replace').splitlines()[-60:]
                print('\n'.join(tail), flush=True)
                raise RuntimeError(f'O2Physics failed for {mode}: exit {result.returncode}')
        log_text = (work / 'stdout.log').read_text(errors='replace')
        if f'Using local file [{network}]' not in log_text:
            raise RuntimeError(f'{mode}: no confirmation that O2Physics loaded the CI model')
        results = [work / 'AnalysisResults.root', work / 'AnalysisResults_trees.root']
        if not all(path.is_file() and path.stat().st_size > 0 for path in results):
            raise RuntimeError(f'{mode}: O2Physics did not produce nonempty ROOT output')
        print(f'O2Physics {mode} model passed.', flush=True)


if __name__ == '__main__':
    main()
