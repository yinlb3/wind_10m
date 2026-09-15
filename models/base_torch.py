#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Common training framework for PyTorch demo models.

Shared loss functions, data preprocessing helpers, and training utilities
used by CNN, DNN, and Transformer wind speed correction models.

Founded in 2025-08-29
Modified in 2026-09-15
@author: yinlb
"""

import copy
import os
import pathlib
import typing

import numpy as np
import torch
from torch import nn
from torch.utils import data as Data


SEED = 250829
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    DEVICE = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = 'cpu'

# Threshold sets used by loss functions.
THRES1 = (
    0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8
)
THRES2 = (
    0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8,
    24.5, 28.5, 32.7
)


def _check_device(
    device: typing.Union[int, str, None],
) -> typing.Union[int, str]:
    """Return a valid torch device identifier."""
    if device is None:
        return DEVICE
    return device


class DiceLoss(nn.Module):
    """Dice loss for regression with threshold-based soft labels."""

    def __init__(
        self,
        thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
        weights: typing.Union[
            typing.Tuple, typing.List, np.ndarray, None
        ] = None,
        smooth: float = 1e-5,
        reduction: str = 'mean',
   ) -> None:
        """
        Initialize DiceLoss.

        Args:
            thres: Thresholds for soft binarization.
            weights: Optional per-threshold weights.
            smooth: Smoothing factor to avoid division by zero.
            reduction: Reduction mode ('mean', 'sum', or 'none').
        """
        super(DiceLoss, self).__init__()
        self.thres = thres
        if weights is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = weights
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def copy(self) -> 'DiceLoss':
        """Return a deep copy of the loss object."""
        return copy.deepcopy(self)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
   ) -> torch.Tensor:
        """
        Compute dice loss.

        Args:
            inputs: Predicted values.
            targets: Target values.

        Returns:
            Loss tensor.
        """
        loss = 0
        for idx, threshold in enumerate(self.thres):
            input_s = self.sigmoid(inputs - threshold)
            target_s = self.sigmoid(targets - threshold)
            intersection = (input_s.view(-1) * target_s.view(-1)).sum()
            pred_sum = input_s.view(-1).sum()
            target_sum = target_s.view(-1).sum()
            dice = (
                (2 * intersection + self.smooth)
                / (pred_sum + target_sum + self.smooth)
            )
            loss += self.w[idx] * (1 - dice)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceMSELoss(nn.Module):
    """Combined dice loss and MSE loss."""

    def __init__(
        self,
        thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
        alpha: float = 0.1,
        weights: typing.Union[
            typing.Tuple, typing.List, np.ndarray, None
        ] = None,
        smooth: float = 1e-5,
        reduction: str = 'mean',
   ) -> None:
        """
        Initialize DiceMSELoss.

        Args:
            thres: Thresholds for dice loss.
            alpha: Weight of MSE loss.
            weights: Optional per-threshold weights.
            smooth: Smoothing factor.
            reduction: Reduction mode.
        """
        super(DiceMSELoss, self).__init__()
        self.thres = thres
        self.alpha = alpha
        if weights is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = weights
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(self.thres, weights=weights, reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def copy(self) -> 'DiceMSELoss':
        """Return a deep copy of the loss object."""
        return copy.deepcopy(self)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
   ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            inputs: Predicted values.
            targets: Target values.

        Returns:
            Loss tensor.
        """
        loss = (
            self.dice(inputs, targets)
            + self.alpha * self.mse(inputs, targets)
        )
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceMSELoss2(nn.Module):
    """Multiplicative dice and MSE loss."""

    def __init__(
        self,
        thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
        weights: typing.Union[
            typing.Tuple, typing.List, np.ndarray, None
        ] = None,
        smooth: float = 1e-5,
        reduction: str = 'mean',
   ) -> None:
        """
        Initialize DiceMSELoss2.

        Args:
            thres: Thresholds for dice loss.
            weights: Optional per-threshold weights.
            smooth: Smoothing factor.
            reduction: Reduction mode.
        """
        super(DiceMSELoss2, self).__init__()
        self.thres = thres
        if weights is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = weights
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(self.thres, weights=weights, reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def copy(self) -> 'DiceMSELoss2':
        """Return a deep copy of the loss object."""
        return copy.deepcopy(self)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
   ) -> torch.Tensor:
        """
        Compute multiplicative combined loss.

        Args:
            inputs: Predicted values.
            targets: Target values.

        Returns:
            Loss tensor.
        """
        loss = self.dice(inputs, targets) * self.mse(inputs, targets)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Interpolate 3-hourly EC forecast data to hourly values.

    Args:
        data: Input array with shape (..., time, ...).
        is_0_84: If True, interpolate the 0-84 hour range with extra
            transition points near hour 72.

    Returns:
        Interpolated array.
    """
    out = data.copy()
    if is_0_84:
        for i in range(24):
            out[..., 3 * i + 1, :, :] = (
                out[..., 3 * i, :, :] * 2 / 3
                + out[..., 3 * i + 3, :, :] / 3
            )
            out[..., 3 * i + 2, :, :] = (
                out[..., 3 * i, :, :] / 3
                + out[..., 3 * i + 3, :, :] * 2 / 3
            )
        for src, w1, w2 in (
            (73, 5 / 6, 1 / 6), (74, 2 / 3, 1 / 3),
            (75, 1 / 2, 1 / 2), (76, 1 / 3, 2 / 3),
            (77, 1 / 6, 5 / 6), (79, 5 / 6, 1 / 6),
            (80, 2 / 3, 1 / 3), (81, 1 / 2, 1 / 2),
            (82, 1 / 3, 2 / 3), (83, 1 / 6, 5 / 6),
        ):
            if src <= 77:
                out[..., src, :, :] = (
                    out[..., 72, :, :] * w1
                    + out[..., 78, :, :] * w2
                )
            else:
                out[..., src, :, :] = (
                    out[..., 78, :, :] * w1
                    + out[..., 84, :, :] * w2
                )
    else:
        for i in range(8):
            out[..., 3 * i + 1, :, :] = (
                out[..., 3 * i, :, :] * 2 / 3
                + out[..., 3 * i + 3, :, :] / 3
            )
            out[..., 3 * i + 2, :, :] = (
                out[..., 3 * i, :, :] / 3
                + out[..., 3 * i + 3, :, :] * 2 / 3
            )
    return out


def prepare_data(
    nwp_path: pathlib.Path,
    obs_path: pathlib.Path,
    target_idx: int,
    is_0_84: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess NWP and observation data for torch models.

    Args:
        nwp_path: pathlib.Path to NWP npy file.
        obs_path: pathlib.Path to observation npy file.
        target_idx: Index of the target variable in observation data.
        is_0_84: Whether to interpolate the 0-84 hour range.

    Returns:
        Tuple of (features, targets) arrays.
    """
    nwp = np.load(nwp_path)
    obs = np.load(obs_path)
    nwp = interp(nwp[:, 12:37, :, :], is_0_84=is_0_84)[:, 1:, :, :]
    obs = obs[:, 13:37, :, target_idx]
    obs[obs > 100] = np.nan
    return nwp, obs


def _build_mask(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return a boolean mask of valid samples across all features."""
    mask = np.ones(labels.shape[0], dtype=np.bool_)
    for time_idx in range(labels.shape[-1]):
        mask &= ~np.isnan(labels[..., time_idx])
        for feat_idx in range(data.shape[1]):
            mask &= ~np.isnan(data[:, feat_idx, :, time_idx])
    return mask


def normalize(
    train: np.ndarray,
    val: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training statistics.

    Args:
        train: Training features.
        val: Validation features.

    Returns:
        Normalized training features, validation features, mean, std.
    """
    mean = np.reshape(np.mean(train, axis=(0, 2)), (1, -1, 1))
    std = np.reshape(np.std(train, axis=(0, 2)), (1, -1, 1))
    return (train - mean) / std, (val - mean) / std, mean, std


def train_model(
    model: nn.Module,
    loss_function: nn.Module,
    train_loader: Data.DataLoader,
    val_loader: Data.DataLoader,
    lr_start: float,
    lr_end: float,
    n_epochs: int,
    weight_decay: float,
    thres: typing.Tuple[float, ...],
) -> nn.Module:
    """
    Train a torch model with exponential learning rate decay.

    Args:
        model: Torch model to train.
        loss_function: Loss function.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        lr_start: Initial learning rate.
        lr_end: Final learning rate.
        n_epochs: Number of epochs.
        weight_decay: Weight decay for AdamW.
        thres: Thresholds for dice loss weight update.

    Returns:
        Trained model.
    """
    lr_ratio = (lr_end / lr_start) ** (1 / (n_epochs - 1))
    for epoch in range(n_epochs):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr_start * lr_ratio ** epoch,
            weight_decay=weight_decay,
        )
        model.train()
        loss_function.train()
        train_yy = []
        pred_yy = []
        for batch_x, batch_y in train_loader:
            predict_y = model(batch_x)
            loss = loss_function(predict_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=10, norm_type=2
            )
            optimizer.step()
            train_yy.append(batch_y.cpu())
            pred_yy.append(predict_y.detach().cpu())
        train_yy = torch.vstack(train_yy)
        pred_yy = torch.vstack(pred_yy)
        train_loss = loss_function(train_yy, pred_yy)
        model.eval()
        loss_function.eval()
        with torch.no_grad():
            val_yy = []
            pred_y = []
            for batch_x, batch_y in val_loader:
                predict_y = model(batch_x)
                val_yy.append(batch_y.cpu())
                pred_y.append(predict_y.cpu())
            val_yy = torch.vstack(val_yy)
            pred_y = torch.vstack(pred_y)
            val_loss = loss_function(val_yy, pred_y)
        print(
            f'Epoch {epoch + 1}: '
            f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}'
        )
    return model


def predict(
    model: nn.Module,
    loader: Data.DataLoader,
) -> np.ndarray:
    """
    Generate predictions with a trained model.

    Args:
        model: Trained torch model.
        loader: Data loader for prediction.

    Returns:
        Predictions as a numpy array.
    """
    model.eval()
    pred_y = []
    with torch.no_grad():
        for batch_x, _ in loader:
            pred_y.append(model(batch_x).cpu())
    return torch.vstack(pred_y).numpy()
