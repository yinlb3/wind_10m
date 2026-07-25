#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Utility functions module

Common utility functions for configuration loading, time formatting, etc.

Founded in 2026-04-03
Modified in 2026-07-22
@author: yinlb
"""

import os
import pathlib
import platform
import typing

import yaml

# Global configuration cache to avoid repeated YAML reads
_CONFIG: typing.Optional[dict] = None


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries; override takes precedence.

    Args:
        base: The base dictionary to be merged into.
        override: The overriding dictionary with higher priority.

    Returns:
        The merged dictionary (base is modified in place).
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base=base[key], override=value)
        else:
            base[key] = value
    return base


def load_config(local_name: typing.Optional[str] = None) -> dict:
    """
    Load and merge the base and local configuration files.

    The base configuration is located at config/config.yaml. The local
    configuration is selected automatically based on the operating system:
    - Windows: config/config.local.windows.yaml
    - Linux: config/config.local.linux.yaml

    It can also be specified via the parameter or the environment variable
    WIND_10M_LOCAL_CONFIG.

    Args:
        local_name: Name of the local configuration file, default None
            (automatically selected by platform).

    Returns:
        The merged configuration dictionary.
    """
    global _CONFIG
    # 1. Load the base config once and cache it globally for repeated calls.
    if _CONFIG is None:
        config_dir = pathlib.Path(__file__).parent.parent / 'config'
        with open(
            config_dir / 'config.yaml', 'r', encoding='utf-8'
        ) as file_handle:
            _CONFIG = yaml.safe_load(file_handle)

        # 2. Determine the local override:
        #    explicit argument > env variable > platform default.
        if local_name is None:
            local_name = os.environ.get('WIND_10M_LOCAL_CONFIG')
        if local_name is None:
            sys_platform = platform.system().lower()
            if sys_platform == 'windows':
                local_name = 'config.local.windows.yaml'
            elif sys_platform == 'linux':
                local_name = 'config.local.linux.yaml'
            else:
                # Fallback for unknown platforms; keeps the three-file
                # convention.
                local_name = 'config.local.yaml'

        # 3. Deep-merge the local override into the base config if it exists.
        local_path = config_dir / local_name
        if local_path.exists():
            with open(
                local_path, 'r', encoding='utf-8'
            ) as file_handle:
                local_cfg = yaml.safe_load(file_handle)
            _deep_merge(base=_CONFIG, override=local_cfg)

    return _CONFIG


def format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    Format seconds as a human-readable time string.

    Used for program running-time statistics, etc. Supports both abbreviated
    and full formats.

    Args:
        second: Number of seconds, must be non-negative.
        is_abbreviation: Whether to use the abbreviated format, default False.
            - True: Returns '45.5s', '5m', '2h', etc.
            - False: Returns '45.5 seconds', '5 minutes', '2 hours', etc.

    Returns:
        The formatted time string.

    Raises:
        ValueError: If the input seconds is negative.

    Examples:
        >>> format_time(45.5)
        '45.5 seconds'
        >>> format_time(45.5, is_abbreviation=True)
        '45.5s'
        >>> format_time(3600)
        '60.0 minutes'
        >>> format_time(7200, is_abbreviation=True)
        '2.0h'
    """
    # Negative durations are not supported by this simple formatter.
    if second < 0:
        raise ValueError('Input seconds cannot be negative')

    # Choose the format first, then pick the largest natural unit (s, m, h).
    if is_abbreviation:
        if second <= 60:
            time_str = str(second) + 's'
        elif second <= 3600:
            time_str = str(second / 60) + 'm'
        else:
            time_str = str(second / 3600) + 'h'
    else:
        if second <= 1:
            time_str = str(second) + ' second'
        elif second <= 60:
            time_str = str(second) + ' seconds'
        elif second <= 3600:
            time_str = str(second / 60) + ' minutes'
        else:
            time_str = str(second / 3600) + ' hours'

    return time_str
