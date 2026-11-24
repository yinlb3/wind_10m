#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified command dispatcher for the configurable wind workflow.

Founded in 2026-11-23
Modified in 2026-11-24
@author: yinlb
"""

import argparse
import os
import pathlib
import subprocess
import sys
import typing

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMANDS = {
    'estimate-cache': 'cache_size.py',
    'cache': 'build_cache.py',
    'qc': 'qc_stations.py',
    'train': 'train.py',
    'verify': 'verify.py',
    'mi': 'compute_mi.py',
    'plot': 'plot.py',
    'stations': 'plot_stations.py',
    'benchmark': 'benchmark_schemes.py',
    'product': 'predict_product.py',
    'download': 'download_obs.py',
}


def parse_args(
    argv: typing.Optional[typing.Sequence[str]]
) -> argparse.Namespace:
    """Parse global configuration options and one workflow command."""
    parser = argparse.ArgumentParser(
        prog='python -m wind10m',
        description='Run the configurable 10 m wind correction workflow',
    )
    parser.add_argument(
        '--config', help='Dataset YAML path; defaults to the reference dataset'
    )
    parser.add_argument(
        '--local', help='Optional machine-specific YAML override path'
    )
    parser.add_argument('command', choices=sorted(COMMANDS))
    parser.add_argument(
        'command_args', nargs=argparse.REMAINDER,
        help='Arguments passed unchanged to the selected workflow command',
    )
    return parser.parse_args(argv)


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    """Run one compatibility entry point with the selected configuration."""
    args = parse_args(argv)
    env = os.environ.copy()
    if args.config:
        env['WIND10M_CONFIG'] = str(pathlib.Path(args.config).resolve())
    if args.local:
        env['WIND10M_LOCAL'] = str(pathlib.Path(args.local).resolve())
    script = PROJECT_ROOT / COMMANDS[args.command]
    command = [sys.executable, str(script), *args.command_args]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, env=env, check=False
    )
    return completed.returncode
