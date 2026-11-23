"""Dataset adapter selection."""

from wind10m.dataset.base import DatasetAdapter
from wind10m.dataset.npy_csv import NpyCsvAdapter

ADAPTERS = {'npy_csv': NpyCsvAdapter}


def create_adapter(cfg: dict) -> DatasetAdapter:
    """
    Create the adapter selected by dataset.type.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        DatasetAdapter: Configured dataset adapter.

    Raises:
        ValueError: If the requested adapter is unknown.
    """
    name = cfg.get('dataset', {}).get('type', 'npy_csv')
    if name not in ADAPTERS:
        raise ValueError(
            'Unknown dataset.type: {0!r}; available: {1}'.format(
                name, sorted(ADAPTERS)
            )
        )
    return ADAPTERS[name](cfg)


__all__ = ['DatasetAdapter', 'NpyCsvAdapter', 'create_adapter']
