#!user/bin.python3
# -*- coding: utf-8 -*-

"""
CNN model for 10m wind speed correction.

1D CNN/ResNet architecture trained on EC forecast features to correct
10m wind speed forecasts.

Founded in 2025-08-29
Modified in 2026-09-15
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

from models import base_torch
from src import utils


# CNN/ResNet feature configuration.
N_FEATURES = (
    64,
    (3, 64, 64, 256),
    (4, 256, 128, 512),
    (6, 512, 256, 1024),
    (3, 1024, 512, 2048),
)
LR_EPOCHS = (1e-3, 1e-6, 1000)
WEIGHT_DECAY = 1e-1
BATCH_SIZE = 256


class Block(nn.Module):
    """Residual convolutional block for ResNet."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        device: typing.Union[int, str, None] = None,
    ) -> None:
        """
        Initialize Block.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels.
            out_channels: Number of output channels.
            device: Target device.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False,
            device=device,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels, device=device)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            device=device,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels, device=device)
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            device=device,
        )
        self.bn3 = nn.BatchNorm1d(out_channels, device=device)
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(out_channels, device=device),
        )
        self.relu = nn.ReLU(inplace=True)

    def copy(self) -> 'Block':
        """Return a deep copy of the block."""
        return copy.deepcopy(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = inputs
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone for wind speed correction."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        device: typing.Union[int, str, None] = None,
    ) -> None:
        """
        Initialize ResNet.

        Args:
            n_features: Number of input feature channels.
            n_labels: Number of output channels.
            device: Target device.
        """
        super(ResNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=N_FEATURES[0],
                kernel_size=7,
                padding=3,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(N_FEATURES[0], device=device),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(
            N_FEATURES[1][1], N_FEATURES[1][2], N_FEATURES[1][3],
            N_FEATURES[1][0], device,
        )
        self.layer2 = self._make_layer(
            N_FEATURES[2][1], N_FEATURES[2][2], N_FEATURES[2][3],
            N_FEATURES[2][0], device,
        )
        self.layer3 = self._make_layer(
            N_FEATURES[3][1], N_FEATURES[3][2], N_FEATURES[3][3],
            N_FEATURES[3][0], device,
        )
        self.layer4 = self._make_layer(
            N_FEATURES[4][1], N_FEATURES[4][2], N_FEATURES[4][3],
            N_FEATURES[4][0], device,
        )
        self.output = nn.Sequential(
            nn.Conv1d(
                in_channels=N_FEATURES[4][3],
                out_channels=n_labels,
                kernel_size=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(n_labels, device=device),
        )
        self._initialization()

    def copy(self) -> 'ResNet':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def _make_layer(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_blocks: int,
        device: typing.Union[int, str, None],
    ) -> nn.Sequential:
        """Create a stack of residual blocks."""
        layers = [Block(in_channels, hidden_channels, out_channels, device)]
        for _ in range(1, n_blocks):
            layers.append(
                Block(out_channels, hidden_channels, out_channels, device)
            )
        return nn.Sequential(*layers)

    def _initialization(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.input(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.output(out)
        return out

    def initialization(self) -> None:
        """Public alias for weight initialization."""
        self._initialization()


class CNN(nn.Module):
    """Simple 1D CNN for wind speed correction."""

    def __init__(
        self,
        n_features: int,
        n_labels: int,
        device: typing.Union[int, str, None] = None,
    ) -> None:
        """
        Initialize CNN.

        Args:
            n_features: Number of input feature channels.
            n_labels: Number of output channels.
            device: Target device.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.bn1 = nn.BatchNorm1d(64, device=device)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.bn2 = nn.BatchNorm1d(128, device=device)
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.bn3 = nn.BatchNorm1d(256, device=device)
        self.conv4 = nn.Conv1d(
            in_channels=256,
            out_channels=n_labels,
            kernel_size=1,
            device=device,
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialization()

    def copy(self) -> 'CNN':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

    def _initialization(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        return out

    def initialization(self) -> None:
        """Public alias for weight initialization."""
        self._initialization()


def _reshape_for_cnn(
    nwp: np.ndarray,
    obs: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Reshape NWP and observation arrays for CNN input.

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
    obs = np.reshape(np.transpose(obs, (0, 2, 1)), (-1, 1, obs.shape[1]))
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
    data_dir: Path,
    train_years: typing.List[int],
    val_year: int,
    target_idx: int,
    thres: typing.Tuple[float, ...],
    output_dir: Path,
) -> None:
    """
    Train a CNN model for one target variable and save predictions.

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

    train_data, train_label = _reshape_for_cnn(train_data, train_label)
    val_data, val_label = _reshape_for_cnn(val_nwp, val_obs)

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

    model = CNN(
        n_features=train_x.shape[1], n_labels=1, device=base_torch.DEVICE
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
    print(f'CNN target {target_idx}: R={corr:.4f}')

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / f'cnn_{val_year}.npy', pred)


def main(
    data_dir: Path,
    output_dir: Path,
    train_years: typing.List[int],
    val_year: int,
) -> None:
    """
    Train CNN models for wind speed correction.

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
    print('Program cnn.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        paths = cfg['paths']
        years = cfg['constants']['years']
        main(
            data_dir=Path(paths['wind_data']),
            output_dir=Path(paths['wind2_model']) / 'cnn',
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
        'Program cnn.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
