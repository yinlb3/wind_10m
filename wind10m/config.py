"""Layered YAML configuration for replaceable datasets."""

import os
import pathlib
import typing

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
BASE_CONFIG = CONFIG_DIR / 'base.yaml'
DEFAULT_DATASET = CONFIG_DIR / 'datasets' / '2411.yaml'
DEFAULT_LOCAL = CONFIG_DIR / 'config.2411.local.yaml'
CONFIG_ENV = 'WIND10M_CONFIG'
LOCAL_ENV = 'WIND10M_LOCAL'


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries; override wins.

    Args:
        base: Base dictionary, modified in place.
        override: Dictionary whose values take precedence.

    Returns:
        dict: The merged dictionary.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: pathlib.Path) -> dict:
    """Read one YAML mapping or raise a clear configuration error."""
    if not path.exists():
        raise FileNotFoundError('Missing configuration: {0}'.format(path))
    with open(path, 'r', encoding='utf-8') as file_handle:
        content = yaml.safe_load(file_handle)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError('Configuration must be a mapping: {0}'.format(path))
    return content


def _resolve_path(value: typing.Union[str, pathlib.Path]) -> pathlib.Path:
    """Resolve a user-supplied path against the current directory."""
    return pathlib.Path(value).expanduser().resolve()


def _implicit_local(
    dataset_path: pathlib.Path
) -> typing.Optional[pathlib.Path]:
    """Return an optional local override for the selected dataset."""
    if dataset_path == DEFAULT_DATASET and DEFAULT_LOCAL.exists():
        return DEFAULT_LOCAL
    candidate = dataset_path.with_name(
        '{0}.local.yaml'.format(dataset_path.stem)
    )
    return candidate if candidate.exists() else None


def load_config(
    config_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    local_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
) -> dict:
    """
    Load generic defaults, a dataset configuration, and local overrides.

    Environment variables WIND10M_CONFIG and WIND10M_LOCAL provide the same
    overrides for compatibility entry points and the unified CLI.

    Args:
        config_path: Dataset YAML path. Defaults to the reference dataset.
        local_path: Optional machine-specific YAML path.

    Returns:
        dict: Merged configuration dictionary.
    """
    selected = config_path or os.environ.get(CONFIG_ENV)
    dataset_path = (
        _resolve_path(selected) if selected else DEFAULT_DATASET.resolve()
    )
    cfg = _read_yaml(BASE_CONFIG)
    deep_merge(cfg, _read_yaml(dataset_path))

    selected_local = local_path or os.environ.get(LOCAL_ENV)
    if selected_local:
        override_path = _resolve_path(selected_local)
        deep_merge(cfg, _read_yaml(override_path))
    else:
        override_path = _implicit_local(dataset_path)
        if override_path is not None:
            deep_merge(cfg, _read_yaml(override_path))
    return cfg
