"""Checks for CI model selection and credential isolation (no O2 installation needed)."""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('o2physics_pid_test', ROOT / 'run/ci/o2physics_pid_test.py')
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class O2PhysicsCITest(unittest.TestCase):
    def test_latest_daily_uses_numeric_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('daily-20260927-0000-99', 'daily-20260928-0000-2',
                         'daily-20260928-0000-10', 'MC-prod-2026-v99'):
                (root / name).mkdir()
            self.assertEqual(ci.latest_package(root), 'O2Physics/daily-20260928-0000-10')

    def test_missing_daily_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ci.latest_package(Path(tmp))

    def test_local_model_overrides_both_ccdb_download_modes(self):
        source = ROOT / 'run/ci/o2physics/configuration.json'
        original = json.loads(source.read_text())
        config = ci.prepare_config(source, '/models/network_full/net_onnx_full.onnx')
        pid = config.pop('pid-tpc-service')
        original.pop('pid-tpc-service')
        self.assertEqual(config, original)
        self.assertEqual(pid['pidTPC.autofetchNetworks'], '0')
        self.assertEqual(pid['pidTPC.ccdb-timestamp'], '0')
        self.assertEqual(pid['pidTPC.useNetworkCorrection'], '1')
        self.assertEqual(pid['pidTPC.networkPathLocally'], '/models/network_full/net_onnx_full.onnx')

    def test_pipeline_reports_failure_in_an_early_stage(self):
        script = ROOT / 'run/ci/o2physics/run.sh'
        commands = re.findall(r'^o2-analysis-[a-z0-9-]+', script.read_text(), re.MULTILINE)
        with tempfile.TemporaryDirectory() as tmp:
            for command in commands:
                fake = Path(tmp) / command
                fake.write_text('#!/bin/sh\nexit ' + ('17' if command == commands[0] else '0') + '\n')
                fake.chmod(0o755)
            result = subprocess.run(['bash', str(script)], env={'PATH': tmp + os.pathsep + os.defpath})
        self.assertEqual(result.returncode, 17)

    def test_environment_drops_credentials_and_runtime_injection_but_keeps_proxies(self):
        host_env = {'JALIEN_TOKEN_CERT': '/private/cert', 'JALIEN_TOKEN_KEY': '/private/key',
                    'X509_USER_PROXY': '/private/proxy', 'HOME': '/private/home',
                    'APPTAINER_BIND': '/private:/private',
                    'APPTAINERENV_JALIEN_TOKEN_CERT': '/private/cert',
                    'SINGULARITYENV_JALIEN_TOKEN_KEY': '/private/key',
                    'http_proxy': 'http://proxy:3128', 'no_proxy': 'localhost'}
        with patch.dict(os.environ, host_env, clear=True):
            env = ci.clean_environment('/isolated')
        self.assertEqual(env['HOME'], '/isolated')
        self.assertEqual(env['http_proxy'], host_env['http_proxy'])
        self.assertEqual(env['no_proxy'], 'localhost')
        self.assertFalse(set(host_env).intersection(env) - {'HOME', 'http_proxy', 'no_proxy'})


if __name__ == '__main__':
    unittest.main()
