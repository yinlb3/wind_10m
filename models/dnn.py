#!user/bin.python3
# -*- coding: utf-8 -*-

"""
DNN model for 10m wind speed correction.

MLP and Network-in-Network (NIN) style architectures for correcting
ECMWF 10m wind speed forecasts.

Founded in 2025-08-29
Modified in 2026-07-22
@author: yinlb
"""

import copy
import os
import sys
import typing
from pathlib import Path

import arrow
import numpy as np
import torch
from torch import nn
from torch.utils import data as Data

from src import utils

from base_torch import (
    DEVICE,
    DiceMSELoss2,
    THRES1,
    THRES2,
    interp,
    normalize,
    predict,
    prepare_data,
    train_model,
)


N_NEURONS = (64, 128, 256)
LR_EPOCHS = (5e-4, 5e-7, 400)
WEIGHT_DECAY = 1e-1
BATCH_SIZE = 16


class MLP(nn.Module):
    """Multi-layer perceptron for wind speed correction."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize MLP.

        Args:
            n_features: Number of input features.
            n_labels: Number of output labels.
            dropout: Dropout probability.
        """
        super(MLP, self).__init__()
        self.linear = nn.Sequential()
        for i, n_outputs in enumerate(N_NEURONS):
            n_inputs = N_NEURONS[i - 1] if i > 0 else n_features
            self.linear += nn.Sequential(
                nn.Linear(n_inputs, n_outputs, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.linear += nn.Sequential(
            nn.Linear(N_NEURONS[-1], n_labels, bias=True),
            nn.ReLU(inplace=True),
        )
        self._initialization()

    def copy(self) -> 'MLP':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def _initialization(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu'
                )
                nn.init.constant_(module.bias, 0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch = inputs.shape[0]
        return self.linear(inputs.view(batch, -1))

    def initialization(self) -> None:
        """Public alias for weight initialization."""
        self._initialization()


class NIN(nn.Module):
    """Network-in-Network style DNN for wind speed correction."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        device: typing.Union[int, str, None] = None,
    ) -> None:
        """
        Initialize NIN.

        Args:
            n_features: Number of input feature channels.
            n_labels: Number of output channels.
            device: Target device.
        """
        super(NIN, self).__init__()
        self.linear = nn.Sequential()
        for i, n_outputs in enumerate(N_NEURONS):
            n_inputs = N_NEURONS[i - 1] if i > 0 else n_features
            self.linear += nn.Sequential(
                nn.Conv1d(
                    in_channels=n_inputs,
                    out_channels=n_outputs,
                    kernel_size=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm1d(n_outputs, device=device),
                nn.ReLU(inplace=True),
            )
        self.linear += nn.Sequential(
            nn.Conv1d(
                in_channels=N_NEURONS[-1],
                out_channels=n_labels,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(n_labels, device=device),
            nn.ReLU(inplace=True),
        )
        self._initialization()

    def copy(self) -> 'NIN':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def _initialization(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(inputs)

    def initialization(self) -> None:
        """Public alias for weight initialization."""
        self._initialization()


class DesNIN(nn.Module):
    """Dense Network-in-Network for wind speed correction."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        channels: int = 64,
        deep: int = 4,
        device: typing.Union[int, str, None] = None,
    ) -> None:
        """
        Initialize DesNIN.

        Args:
            n_features: Number of input feature channels.
            n_labels: Number of output channels.
            channels: Number of hidden channels.
            deep: Number of dense layers.
            device: Target device.
        """
        super(DesNIN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=channels,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(channels, device=device),
            nn.ReLU(inplace=True),
        )
        self.hidden_layers = nn.ModuleList()
        for _ in range(1, deep):
            self.hidden_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    bias=False,
                    device=device,
                ),
                nn.BatchNorm1d(channels, device=device),
                nn.ReLU(inplace=True),
            ))
        self.output_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=n_labels,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(n_labels, device=device),
            nn.ReLU(inplace=True),
        )
        self._initialization()

    def copy(self) -> 'DesNIN':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def _initialization(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = [self.input_layer(inputs)]
        for hidden_layer in self.hidden_layers:
            features.append(
                hidden_layer(torch.sum(torch.stack(features), dim=0))
            )
        return self.output_layer(torch.sum(torch.stack(features), dim=0))

    def initialization(self) -> None:
        """Public alias for weight initialization."""
        self._initialization()


def _reshape_for_dnn(
    nwp: np.ndarray,
    obs: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Reshape NWP and observation arrays for DNN input.

    Args:
        nwp: NWP array of shape (samples, lead, stations, features).
        obs: Observation array of shape (samples, lead, stations).

    Returns:
        Reshaped (features, labels) arrays.
    """
    # (samples, lead, stations, features) -> (samples*stations, features*lead)
    nwp = np.reshape(
        np.transpose(nwp, (0, 3, 1, 2)),
        (-1, nwp.shape[3] * nwp.shape[1]),
    )
    # (samples, lead, stations) -> (samples*stations, lead)
    obs = np.reshape(np.transpose(obs, (0, 2, 1)), (-1, obs.shape[1]))
    return nwp, obs


def _build_valid_mask(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return mask of samples without missing values."""
    mask = np.ones(labels.shape[0], dtype=np.bool_)
    for i in range(labels.shape[-1]):
        mask &= ~np.isnan(labels[:, i])
        mask &= ~np.isnan(features).any(axis=1)
    return mask


def _train_one_target(
    data_dir: Path,
    train_years: typing.List[int],
    val_year: int,
    target_idx: int,
    thres: typing.Tuple[float, ...],
    output_dir: Path,
) -> None:
    """
    Train a DNN model for one target variable and save predictions.

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
        nwp, obs = prepare_data(
            data_dir / f'nwp{year}.npy',
            data_dir / f'ob{year}.npy',
            target_idx,
            is_0_84=False,
        )
        data_list.append(nwp)
        label_list.append(obs)
    train_data = np.concatenate(data_list, axis=0)
    train_label = np.concatenate(label_list, axis=0)
    val_nwp, val_obs = prepare_data(
        data_dir / f'nwp{val_year}.npy',
        data_dir / f'ob{val_year}.npy',
        target_idx,
        is_0_84=False,
    )

    train_data, train_label = _reshape_for_dnn(train_data, train_label)
    val_data, val_label = _reshape_for_dnn(val_nwp, val_obs)

    train_mask = _build_valid_mask(train_data, train_label)
    val_mask = _build_valid_mask(val_data, val_label)
    train_x = train_data[train_mask]
    val_x = val_data[val_mask]
    train_y = train_label[train_mask]
    val_y = val_label[val_mask]

    train_x, val_x, _, _ = normalize(train_x, val_x)

    train_dataset = Data.TensorDataset(
        torch.from_numpy(train_x).to(DEVICE),
        torch.from_numpy(train_y).to(DEVICE),
    )
    val_dataset = Data.TensorDataset(
        torch.from_numpy(val_x).to(DEVICE),
        torch.from_numpy(val_y).to(DEVICE),
    )
    train_loader = Data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = Data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = NIN(
        n_features=train_x.shape[1],
        n_labels=train_y.shape[1],
        device=DEVICE,
    )
    weights = np.ones(len(thres), dtype=np.float32)
    weights = weights / np.sum(weights)
    loss_function = DiceMSELoss2(thres, weights=weights)
    model = train_model(
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

    pred = predict(model, val_loader)
    corr = np.corrcoef(val_y[:, 0], pred[:, 0])[0, 1]
    print(f'DNN target {target_idx}: R={corr:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / f'dnn_{val_year}.npy', pred)


def main(
    data_dir: Path,
    output_dir: Path,
    train_years: typing.List[int],
    val_year: int,
) -> None:
    """
    Train DNN models for wind speed correction.

    Args:
        data_dir: Directory containing npy data files.
        output_dir: Directory to save model predictions.
        train_years: Years used for training.
        val_year: Year used for validation/output.
    """
    _train_one_target(data_dir, train_years, val_year, 3, THRES1, output_dir)
    _train_one_target(data_dir, train_years, val_year, 7, THRES2, output_dir)


if __name__ == '__main__':
    print('Program dnn.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        paths = cfg['paths']
        years = cfg['constants']['years']
        main(
            data_dir=Path(paths['wind_data']),
            output_dir=Path(paths['wind2_model']) / 'dnn',
            train_years=years[:-1],
            val_year=years[-1],
        )
    elif len(sys.argv) == 5:
        main(
            data_dir=Path(sys.argv[1]),
            output_dir=Path(sys.argv[2]),
            train_years=[int(year) for year in sys.argv[3].split(',')],
            val_year=int(sys.argv[4]),
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program dnn.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
