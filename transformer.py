#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Transformer model for 10m wind speed correction.

Transformer encoder architecture for correcting ECMWF 10m wind speed
forecasts.

Founded in 2025-09-09
Modified in 2026-09-15
@author: yinlb
"""

import copy
import math
import os
import pathlib
import sys
import typing

import arrow
import numpy as np
import torch
from torch import nn
from torch.utils import data as Data

from models import base_torch
from src import utils


LR_EPOCHS = (1e-2, 1e-4, 100)
WEIGHT_DECAY = 1e-1
BATCH_SIZE = 256


class PositionalEncoding(nn.Module):
    """Inject positional information with sine and cosine encodings."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        device: typing.Union[int, str, None] = None,
   ) -> None:
        """
        Initialize PositionalEncoding.

        Args:
            d_model: Embedding dimension.
            dropout: Dropout probability.
            max_len: Maximum sequence length.
            device: Target device.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def copy(self) -> 'PositionalEncoding':
        """Return a deep copy of the module."""
        return copy.deepcopy(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to inputs."""
        out = inputs + self.pe[:inputs.size(0), :].transpose(0, 1)
        return self.dropout(out)


class TransformerModel(nn.Transformer):
    """Transformer model for wind speed correction."""

    def __init__(
        self,
        nfeat: int,
        nlabel: int,
        ninp: int = 512,
        nhead: int = 8,
        nhid: int = 2048,
        nlayers: int = 6,
        dropout: float = 0.5,
        device: typing.Union[int, str, None] = None,
   ) -> None:
        """
        Initialize TransformerModel.

        Args:
            nfeat: Number of input feature channels.
            nlabel: Number of output channels.
            ninp: Transformer embedding dimension.
            nhead: Number of attention heads.
            nhid: Feed-forward dimension.
            nlayers: Number of encoder layers.
            dropout: Dropout probability.
            device: Target device.
        """
        super(TransformerModel, self).__init__(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            num_encoder_layers=nlayers,
            batch_first=True,
            device=device,
        )
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(
            d_model=ninp,
            dropout=dropout,
            device=device,
        )
        self.input_emb = nn.Sequential(
            nn.Conv1d(
                in_channels=nfeat,
                out_channels=ninp,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.LayerNorm([ninp, 24], device=device),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=ninp,
                out_channels=nlabel,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.LayerNorm([nlabel, 24], device=device),
            nn.ReLU(),
        )
        self.ninp = ninp
        self.init_weights()

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for transformer."""
        return torch.log(torch.tril(torch.ones(size, size)))

    def init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                nn.init.zeros_(module.bias)

    def copy(self) -> 'TransformerModel':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.input_emb(src) * math.sqrt(self.ninp)
        out = self.pos_encoder(out)
        out = out.transpose(0, 1)
        out = self.encoder(out)
        out = out.transpose(0, 1)
        out = self.decoder(out)
        return out


def _reshape_for_transformer(
    nwp: np.ndarray,
    obs: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Reshape NWP and observation arrays for transformer input.

    Args:
        nwp: NWP array of shape (samples, lead, stations, features).
        obs: Observation array of shape (samples, lead, stations).

    Returns:
        Reshaped (features, labels) arrays.
    """
    # (samples, lead, stations, features) -> (samples*stations, features, lead)
    nwp = np.reshape(
        np.transpose(nwp, (0, 3, 1, 2)),
        (-1, nwp.shape[3], nwp.shape[1]),
    )
    # (samples, lead, stations) -> (samples*stations, 1, lead)
    obs = np.reshape(
        np.transpose(obs, (0, 2, 1)),
        (-1, 1, obs.shape[1]),
    )
    return nwp, obs


def _build_valid_mask(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return mask of samples without missing values."""
    mask = np.ones(labels.shape[0], dtype=np.bool_)
    for time_idx in range(labels.shape[-1]):
        mask &= ~np.isnan(labels[:, 0, time_idx])
        for feat_idx in range(features.shape[1]):
            mask &= ~np.isnan(features[:, feat_idx, time_idx])
    return mask


def _train_one_target(
    data_dir: pathlib.Path,
    train_years: typing.List[int],
    val_year: int,
    target_idx: int,
    thres: typing.Tuple[float, ...],
    output_dir: pathlib.Path,
) -> None:
    """
    Train a Transformer model for one target variable and save predictions.

    Args:
        data_dir: Directory containing nwp{year}.npy and ob{year}.npy.
        train_years: Years used for training.
        val_year: Year used for validation and prediction output.
        target_idx: Index of target variable in observation data.
        thres: Thresholds for dice loss.
        output_dir: Directory to save predictions.
    """
    data_list = []
    label_list = []
    for year in train_years:
        nwp, obs = base_torch.prepare_data(
            data_dir / f'nwp{year}.npy',
            data_dir / f'ob{year}.npy',
            target_idx,
            is_0_84=False,
        )
        data_list.append(nwp)
        label_list.append(obs)
    train_data = np.concatenate(data_list, axis=0)
    train_label = np.concatenate(label_list, axis=0)
    val_nwp, val_obs = base_torch.prepare_data(
        data_dir / f'nwp{val_year}.npy',
        data_dir / f'ob{val_year}.npy',
        target_idx,
        is_0_84=False,
    )

    train_data, train_label = _reshape_for_transformer(train_data, train_label)
    val_data, val_label = _reshape_for_transformer(val_nwp, val_obs)

    train_mask = _build_valid_mask(train_data, train_label)
    val_mask = _build_valid_mask(val_data, val_label)
    train_x = train_data[train_mask]
    val_x = val_data[val_mask]
    train_y = train_label[train_mask]
    val_y = val_label[val_mask]

    train_x, val_x, _, _ = base_torch.normalize(train_x, val_x)

    train_dataset = Data.TensorDataset(
        torch.from_numpy(train_x).to(base_torch.DEVICE),
        torch.from_numpy(train_y).to(base_torch.DEVICE),
    )
    val_dataset = Data.TensorDataset(
        torch.from_numpy(val_x).to(base_torch.DEVICE),
        torch.from_numpy(val_y).to(base_torch.DEVICE),
    )
    train_loader = Data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = Data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = TransformerModel(
        nfeat=train_x.shape[1],
        nlabel=1,
        device=base_torch.DEVICE,
    )
    weights = np.ones(len(thres), dtype=np.float32)
    weights = weights / np.sum(weights)
    loss_function = base_torch.DiceMSELoss2(thres, weights=weights)
    model = base_torch.train_model(
        model=model,
        loss_function=loss_function,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_start=LR_EPOCHS[0],
        lr_end=LR_EPOCHS[1],
        n_epochs=LR_EPOCHS[2],
        weight_decay=WEIGHT_DECAY,
        thres=thres,
    )

    pred = base_torch.predict(model, val_loader)
    corr = np.corrcoef(val_y[:, 0], pred[:, 0])[0, 1]
    print(f'Transformer target {target_idx}: R={corr:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / f'transformer_{val_year}.npy', pred)


def main(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    train_years: typing.List[int],
    val_year: int,
) -> None:
    """
    Train Transformer models for wind speed correction.

    Args:
        data_dir: Directory containing npy data files.
        output_dir: Directory to save model predictions.
        train_years: Years used for training.
        val_year: Year used for validation/output.
    """
    _train_one_target(
        data_dir, train_years, val_year, 3, base_torch.THRES1, output_dir
    )
    _train_one_target(
        data_dir, train_years, val_year, 7, base_torch.THRES2, output_dir
    )


if __name__ == '__main__':
    print('Program transformer.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        paths = cfg['paths']
        years = cfg['constants']['years']
        main(
            data_dir=pathlib.Path(paths['wind_data']),
            output_dir=pathlib.Path(paths['wind2_model']) / 'transformer',
            train_years=years[:-1],
            val_year=years[-1],
        )
    elif len(sys.argv) == 5:
        main(
            data_dir=pathlib.Path(sys.argv[1]),
            output_dir=pathlib.Path(sys.argv[2]),
            train_years=[int(year) for year in sys.argv[3].split(',')],
            val_year=int(sys.argv[4]),
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program transformer.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
