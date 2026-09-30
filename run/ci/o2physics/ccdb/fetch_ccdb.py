"""Run the CI workflow in an authenticated O2Physics shell and bundle its CCDB cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile

# Example running:
# cd /lustre/alice/users/csonnab/TPC/o2-tpc-pid
# python3 /lustre/alice/users/csonnab/TPC/o2-tpc-pid/run/ci/o2physics/ccdb/fetch_ccdb.py --token-dir /lustre/alice/users/csonnab/token_dir --token-id 9898 --unset-proxies

def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    base = Path(__file__).resolve().parent
    fixtures = base.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=base / 'work',
                        help='Working directory, cleared on startup (default: ccdb/work)')
    parser.add_argument('--aod', type=Path, default=fixtures / 'AO2D.2dfs.root')
    parser.add_argument('--workflow', type=Path, default=fixtures / 'run.sh')
    parser.add_argument('--config', type=Path, default=fixtures / 'configuration.json')
    parser.add_argument('--archive', type=Path, default=base / 'ccdb.tar.gz')
    parser.add_argument('--network', type=Path,
                        help='Optional local ONNX; otherwise use configured CCDB network')
    parser.add_argument('--host', default='http://alice-ccdb.cern.ch')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--unset-proxies', action='store_true',
                        help='Remove HTTP, HTTPS and ALL proxy variables from the workflow environment')
    parser.add_argument('--token-dir', type=Path,
                        help='Directory containing tokencert_ID.pem and tokenkey_ID.pem')
    parser.add_argument('--token-id', type=int,
                        help='Select a token pair when the directory contains multiple IDs')
    parser.add_argument('--remove-archive-on-startup', default=1, help='Removes the tar file on startup')
    args = parser.parse_args()
    env = os.environ.copy()
    if args.unset_proxies:
        for variable in ('http_proxy', 'https_proxy', 'all_proxy',
                         'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
            env.pop(variable, None)
    if args.token_id is not None and args.token_dir is None:
        parser.error('--token-id requires --token-dir')
    if args.token_dir is not None:
        token_dir = args.token_dir.expanduser().resolve()
        if not token_dir.is_dir():
            parser.error(f'Token directory does not exist: {token_dir}')
        if args.token_id is None:
            token_ids = []
            for certificate in sorted(token_dir.glob('tokencert_*.pem')):
                match = re.fullmatch(r'tokencert_(\d+)\.pem', certificate.name)
                if (match and certificate.is_file()
                        and (token_dir / f'tokenkey_{match.group(1)}.pem').is_file()):
                    token_ids.append(match.group(1))
            if len(token_ids) != 1:
                parser.error('Expected one matching token pair in --token-dir; '
                             'use --token-id to select a pair')
            token_id = token_ids[0]
        else:
            token_id = str(args.token_id)
        for variable, prefix in (('JALIEN_TOKEN_CERT', 'tokencert'),
                                 ('JALIEN_TOKEN_KEY', 'tokenkey')):
            token_path = token_dir / f'{prefix}_{token_id}.pem'
            if not token_path.is_file() or not os.access(token_path, os.R_OK):
                parser.error(f'Token file is missing or unreadable: {token_path}')
            env[variable] = str(token_path)
    output, archive = args.output.resolve(), args.archive.resolve()
    inputs = {'aod': args.aod.resolve(), 'workflow': args.workflow.resolve(),
              'config': args.config.resolve()}
    for path in [*inputs.values(), *([args.network] if args.network else [])]:
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    if archive.exists():
        subprocess.call(['rm', '-f',  str(output), + "/../*.tar.gz"])
    if output == archive or output in archive.parents:
        parser.error('Archive must be outside the work directory')
    if args.output.is_symlink() or (output.exists() and not output.is_dir()):
        parser.error('Work directory must be a real directory, not a symlink or file')
    protected = [*inputs.values(), base, Path.cwd().resolve(), Path.home().resolve()]
    if args.network:
        protected.append(args.network.resolve())
    if args.token_dir:
        protected.append(args.token_dir.expanduser().resolve())
    for variable in ('JALIEN_TOKEN_CERT', 'JALIEN_TOKEN_KEY'):
        if env.get(variable):
            protected.append(Path(env[variable]).expanduser().resolve())
    if any(output == path or output in path.parents for path in protected):
        parser.error('Work directory must not contain inputs, tokens, code, home or current directory')
    workflow = inputs['workflow'].read_text()
    if 'O2_AOD_FILE' not in workflow:
        parser.error('Workflow must accept O2_AOD_FILE (as the repository run.sh does)')
    commands = set(re.findall(r'\bo2-analysis-[a-zA-Z0-9_-]+', workflow))
    if not commands:
        parser.error('No O2Physics commands found in workflow')
    missing = sorted(command for command in commands if not shutil.which(command))
    if missing:
        parser.error('Load an O2Physics environment; missing: ' + ', '.join(missing))
    import ROOT
    if ROOT.gSystem.Load('libO2CCDB') < 0:
        parser.error('Cannot load libO2CCDB in this Python/ROOT environment')
    config = json.loads(inputs['config'].read_text())
    for section in config.values():
        if isinstance(section, dict):
            for key in section:
                if 'ccdb' in key.lower() and 'url' in key.lower():
                    section[key] = args.host
    if args.network:
        config['pid-tpc-service'].update({
            'pidTPC.autofetchNetworks': '0', 'pidTPC.ccdb-timestamp': '0',
            'pidTPC.useNetworkCorrection': '1',
            'pidTPC.networkPathLocally': str(args.network.resolve()),
        })
    if output.exists():
        print(f'Removing existing work directory: {output}', flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    cache = output / 'ccdb'
    cache.mkdir()
    (output / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    provenance = {name: {'name': path.name, 'sha256': digest(path)}
                  for name, path in inputs.items()}
    env['O2_AOD_FILE'] = str(inputs['aod'])
    env['ALICEO2_CCDB_LOCALCACHE'] = str(cache)
    env.pop('IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE', None)
    print(f'Running workflow; log: {output / "workflow.log"}', flush=True)
    with (output / 'workflow.log').open('w') as log:
        process = subprocess.Popen(['bash', str(inputs['workflow'])], cwd=output, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if status:
            raise RuntimeError(f'Workflow failed ({status}); inspect {output / "workflow.log"}')
    for name, path in inputs.items():
        if digest(path) != provenance[name]['sha256']:
            raise RuntimeError(f'Input changed during capture: {path}')
    for name in ('AnalysisResults.root', 'AnalysisResults_trees.root'):
        if not (output / name).is_file() or not (output / name).stat().st_size:
            raise RuntimeError(f'Workflow did not produce {name}; inspect workflow.log')
    access_log = cache / 'log'
    if not access_log.is_file():
        raise RuntimeError('O2 produced no CCDB access log; cannot audit this snapshot')
    queries = {}
    for path, timestamp in re.findall(r' to (\S+) timestamp (-?\d+)', access_log.read_text()):
        queries.setdefault(path, set()).add(int(timestamp))
    files, objects = [], []
    for snapshot in sorted(cache.rglob('snapshot.root')):
        if snapshot.is_symlink():
            raise RuntimeError(f'Unexpected cache symlink: {snapshot}')
        path = snapshot.parent.relative_to(cache).as_posix()
        root_file = ROOT.TFile.Open(str(snapshot))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f'Invalid ROOT snapshot: {snapshot}')
        metadata = root_file.Get('ccdb_meta')
        if not metadata:
            raise RuntimeError(f'Missing CCDB metadata: {snapshot}')
        headers = {str(item.first): str(item.second) for item in metadata}
        root_file.Close()
        start, end = int(headers['Valid-From']), int(headers['Valid-Until'])
        timestamps = sorted(queries.get(path, set()))
        if not timestamps:
            raise RuntimeError(f'No recorded requests for {path}; cannot verify validity')
        if any(timestamp <= 0 or not start <= timestamp < end for timestamp in timestamps):
            raise RuntimeError(
                f'{path}: requests {timestamps} exceed snapshot validity [{start}, {end}). '
                'Use a fixture within one validity interval; no archive was produced.')
        objects.append({'path': path, 'uuid': headers['ETag'].strip('"'),
                        'valid_from': start, 'valid_until': end, 'timestamps': timestamps})
        files.append(snapshot)
    if not objects:
        raise RuntimeError('No CCDB ROOT objects captured')
    for header in sorted(cache.rglob('header.json')):
        if header.is_symlink():
            raise RuntimeError(f'Unexpected cache symlink: {header}')
        json.loads(header.read_text())
        files.append(header)
    manifest = {
        'format': 1, 'inputs': provenance, 'objects': objects,
        'files': [{'path': path.relative_to(cache).as_posix(), 'sha256': digest(path)}
                  for path in files],
        'missing_object_paths': sorted(set(queries) - {entry['path'] for entry in objects}
                                       - {path.parent.parent.relative_to(cache).as_posix()
                                          for path in files if path.name == 'header.json'}),
    }
    manifest_path = cache / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.partial')
    with tarfile.open(temporary, 'w:gz') as bundle:
        for path in [*files, manifest_path]:
            bundle.add(path, arcname='ccdb/' + path.relative_to(cache).as_posix())
    temporary.replace(archive)
    print(f'Created {archive}: {len(objects)} objects, {archive.stat().st_size:,} bytes')
    if archive.stat().st_size >= 100 * 1024 * 1024:
        print('Archive exceeds 100 MiB; store it with Git LFS before pushing to GitHub.')
    subprocess.call(['rm', '-rf', str(output)])

if __name__ == '__main__':
    main()
