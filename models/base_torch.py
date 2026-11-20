#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common training framework for PyTorch demo models.

Shared loss functions, data preprocessing helpers, and training utilities
used by CNN, DNN, and Transformer wind speed correction models.

Founded in 2025-08-29
Modified in 2026-09-18
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
    """
    Return a valid torch device identifier.

    Args:
        device (typing.Union[int, str, None]): Device to validate, None falls
            back to the module-level default.

    Returns:
        typing.Union[int, str]: The given device, or DEVICE when None.
    """
    if device is None:
        return DEVICE
    return device


def load_state_dict_compat(
    model: nn.Module,
    state_dict: typing.Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """
    Load a state dict, accepting checkpoints with legacy public keys.

    Model attributes became private (single leading underscore) on
    2026-09-18, which changed every key produced by state_dict().
    Checkpoints saved before that change still use the public names
    (e.g. 'conv1.weight'); this helper remaps such keys to the private
    form before loading, so old checkpoints stay loadable.

    Args:
        model: Model instance to load the weights into.
        state_dict: State dict, possibly using legacy public key names.
        strict: Forwarded to nn.Module.load_state_dict. Defaults to True.

    Returns:
        None

    Raises:
        RuntimeError: If the remapped keys still do not match the model.
    """
    valid_keys = set(model.state_dict().keys())
    remapped = dict()
    for key, value in state_dict.items():
        if key in valid_keys:
            remapped[key] = value
        elif ('_' + key) in valid_keys:
            remapped['_' + key] = value
        else:
            remapped[key] = value
    model.load_state_dict(remapped, strict=strict)


class DiceLoss(nn.Module):
    """
    Dice loss for regression with threshold-based soft labels.

    Attributes:
        _thres (typing.Union[typing.Tuple, typing.List, np.ndarray, None]):
            Thresholds used for the soft binarization.
        _w (np.ndarray): Weight of each threshold in the loss sum.
        _smooth (float): Smoothing factor added to the dice ratio.
        _reduction (str): Reduction mode, 'mean', 'sum', or 'none'.
        _sigmoid (nn.Sigmoid): Sigmoid producing the soft labels.
    """

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
        self._thres = thres
        if weights is None:
            self._w = np.ones(len(self._thres), dtype=np.float32)
            self._w = self._w / np.sum(self._w)
        else:
            self._w = weights
        self._smooth = smooth
        self._reduction = reduction
        self._sigmoid = nn.Sigmoid()

    def copy(self) -> 'DiceLoss':
        """
        Return a deep copy of the loss object.

        Returns:
            DiceLoss: Independent deep copy of this loss.
        """
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
        for idx, threshold in enumerate(self._thres):
            input_s = self._sigmoid(inputs - threshold)
            target_s = self._sigmoid(targets - threshold)
            intersection = (input_s.view(-1) * target_s.view(-1)).sum()
            pred_sum = input_s.view(-1).sum()
            target_sum = target_s.view(-1).sum()
            dice = (
                (2 * intersection + self._smooth)
                / (pred_sum + target_sum + self._smooth)
            )
            loss += self._w[idx] * (1 - dice)
        if self._reduction == 'mean':
            return loss.mean()
        if self._reduction == 'sum':
            return loss.sum()
        return loss


class DiceMSELoss(nn.Module):
    """
    Combined dice loss and MSE loss.

    Attributes:
        _thres (typing.Union[typing.Tuple, typing.List, np.ndarray, None]):
            Thresholds used for the soft binarization.
        _alpha (float): Weight of the MSE term.
        _w (np.ndarray): Weight of each threshold in the dice term.
        _smooth (float): Smoothing factor added to the dice ratio.
        _reduction (str): Reduction mode, 'mean', 'sum', or 'none'.
        _sigmoid (nn.Sigmoid): Sigmoid producing the soft labels.
        _dice (DiceLoss): Dice term of the combined loss.
        _mse (nn.MSELoss): MSE term of the combined loss.
    """

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
        self._thres = thres
        self._alpha = alpha
        if weights is None:
            self._w = np.ones(len(self._thres), dtype=np.float32)
            self._w = self._w / np.sum(self._w)
        else:
            self._w = weights
        self._smooth = smooth
        self._reduction = reduction
        self._sigmoid = nn.Sigmoid()
        self._dice = DiceLoss(self._thres, weights=weights, reduction='none')
        self._mse = nn.MSELoss(reduction='none')

    def copy(self) -> 'DiceMSELoss':
        """
        Return a deep copy of the loss object.

        Returns:
            DiceMSELoss: Independent deep copy of this loss.
        """
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
            self._dice(inputs, targets)
            + self._alpha * self._mse(inputs, targets)
        )
        if self._reduction == 'mean':
            return loss.mean()
        if self._reduction == 'sum':
            return loss.sum()
        return loss


class DiceMSELoss2(nn.Module):
    """
    Multiplicative dice and MSE loss.

    Attributes:
        _thres (typing.Union[typing.Tuple, typing.List, np.ndarray, None]):
            Thresholds used for the soft binarization.
        _w (np.ndarray): Weight of each threshold in the dice term.
        _smooth (float): Smoothing factor added to the dice ratio.
        _reduction (str): Reduction mode, 'mean', 'sum', or 'none'.
        _sigmoid (nn.Sigmoid): Sigmoid producing the soft labels.
        _dice (DiceLoss): Dice term of the combined loss.
        _mse (nn.MSELoss): MSE term of the combined loss.
    """

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
        self._thres = thres
        if weights is None:
            self._w = np.ones(len(self._thres), dtype=np.float32)
            self._w = self._w / np.sum(self._w)
        else:
            self._w = weights
        self._smooth = smooth
        self._reduction = reduction
        self._sigmoid = nn.Sigmoid()
        self._dice = DiceLoss(self._thres, weights=weights, reduction='none')
        self._mse = nn.MSELoss(reduction='none')

    def copy(self) -> 'DiceMSELoss2':
        """
        Return a deep copy of the loss object.

        Returns:
            DiceMSELoss2: Independent deep copy of this loss.
        """
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
        loss = self._dice(inputs, targets) * self._mse(inputs, targets)
        if self._reduction == 'mean':
            return loss.mean()
        if self._reduction == 'sum':
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
        # 1. Interpolate the hour 0-24 range from 3-hourly values.
        for i in range(24):
            out[..., 3 * i + 1, :, :] = (
                out[..., 3 * i, :, :] * 2 / 3
                + out[..., 3 * i + 3, :, :] / 3
            )
            out[..., 3 * i + 2, :, :] = (
                out[..., 3 * i, :, :] / 3
                + out[..., 3 * i + 3, :, :] * 2 / 3
            )

        # 2. Fill the transition hours between hour 72 and hour 84.
        # src: target hour index; w1, w2: weights of the two anchor hours
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
    # nwp: numerical weather prediction forecast array
    nwp = np.load(nwp_path)
    # obs: observation array
    obs = np.load(obs_path)
    nwp = interp(nwp[:, 12:37, :, :], is_0_84=is_0_84)[:, 1:, :, :]
    obs = obs[:, 13:37, :, target_idx]
    obs[obs > 100] = np.nan
    return nwp, obs


def _build_mask(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of valid samples across all features.

    Args:
        data (np.ndarray): Feature array with the feature axis at index 1.
        labels (np.ndarray): Label array with the time axis last.

    Returns:
        np.ndarray: Boolean mask selecting the valid samples.
    """
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
    # 1. Build the optimizer and the learning rate schedule.
    lr_ratio = (lr_end / lr_start) ** (1 / (n_epochs - 1))
    for epoch in range(n_epochs):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr_start * lr_ratio ** epoch,
            weight_decay=weight_decay,
        )

        # 2. Train one epoch and compute the training loss.
        model.train()
        loss_function.train()
        # yy: per-batch tensors stacked over all batches (y = one batch);
        # train_yy holds labels, pred_yy holds predictions
        train_yy = list()
        pred_yy = list()
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

        # 3. Evaluate the epoch on the validation set.
        model.eval()
        loss_function.eval()
        with torch.no_grad():
            val_yy = list()
            pred_y = list()
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
    pred_y = list()
    with torch.no_grad():
        for batch_x, _ in loader:
            pred_y.append(model(batch_x).cpu())
    return torch.vstack(pred_y).numpy()
