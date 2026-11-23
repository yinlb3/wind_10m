"""End-to-end checks for the unified command dispatcher."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

import yaml

from wind10m.config import load_config

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class CliTest(unittest.TestCase):
    """Run the real module entry point with a custom dataset config."""

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        """Run one CLI command and capture text output."""
        return subprocess.run(
            [sys.executable, '-m', 'wind10m', *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_help(self) -> None:
        """The dispatcher exposes every supported workflow command."""
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('estimate-cache', result.stdout)
        self.assertIn('predict-10day', result.stdout)

    def test_custom_config_and_local_override(self) -> None:
        """Verify global options reach compatibility scripts via environment."""
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = load_config()
            cfg['cache']['mode'] = 'full'
            dataset_path = root / 'custom.yaml'
            local_path = root / 'custom.local.yaml'
            with open(dataset_path, 'w', encoding='utf-8') as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            with open(local_path, 'w', encoding='utf-8') as handle:
                yaml.safe_dump({'cache': {'mode': 'obs'}}, handle)
            result = self.run_cli(
                '--config', str(dataset_path),
                '--local', str(local_path),
                'estimate-cache',
                '--stations', '2', '--samples', '3',
                '--hours', '4', '--features', '5',
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            '2 stations x 3 samples x 4 hours; 5 features', result.stdout
        )
        self.assertIn('Configured cache.mode: obs', result.stdout)


if __name__ == '__main__':
    unittest.main()
