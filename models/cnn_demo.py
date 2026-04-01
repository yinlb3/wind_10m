#!user/bin.python3

"""
Founded in 2025-08-29
Modified in 2025-09-09
@author: yinlb
"""
import os
import sys
import typing
from typing import Optional, Union

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd

import torch
from torch import nn, device
from torch.utils import data as Data


SEED = 250829
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
DEVICE = 0
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
N_FEATURES = (
    64,
    (3, 64, 64, 256),
    (4, 256, 128, 512),
    (6, 512, 256, 1024),
    (3, 1024, 512, 2048)
)
LR_EPOCHS = (1e-3, 1e-6, 1000)
WEIGHT_DECAY = 1e-1
BATCH_SIZE = 256


THRES1 = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8)
THRES2 = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7)


class DiceLoss(nn.Module):
    def __init__(
            self: nn.Module,
            thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
            w: typing.Union[typing.Tuple, typing.List, np.ndarray, None] = None,
            smooth: int = 1e-5,
            reduction: str = 'mean'
    ):
        super(DiceLoss, self).__init__()
        self.thres = thres
        if w is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = w
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def forward(
            self: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ):
        loss = 0
        for i, t in enumerate(self.thres):
            input_s = self.sigmoid(inputs - t)
            target_s = self.sigmoid(targets - t)
            a = (input_s.view(-1) * target_s.view(-1)).sum()
            ab = input_s.view(-1).sum()
            ac = target_s.view(-1).sum()
            dice = (2 * a + self.smooth) / (ab + ac + self.smooth)
            loss += self.w[i] * (1 - dice)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceMSELoss(nn.Module):
    def __init__(
            self: nn.Module,
            thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
            alpha: float = 0.1,
            w: typing.Union[typing.Tuple, typing.List, np.ndarray, None] = None,
            smooth: int = 1e-5,
            reduction: str = 'mean'
    ):
        super(DiceMSELoss, self).__init__()
        self.thres = thres
        self.alpha = alpha
        if w is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = w
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(self.thres, w=w, reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(
            self: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ):
        loss = self.dice(inputs, targets) + self.alpha * self.mse(inputs, targets)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceMSELoss2(nn.Module):
    def __init__(
            self: nn.Module,
            thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
            w: typing.Union[typing.Tuple, typing.List, np.ndarray, None] = None,
            smooth: int = 1e-5,
            reduction: str = 'mean'
    ):
        super(DiceMSELoss2, self).__init__()
        self.thres = thres
        if w is None:
            self.w = np.ones(len(self.thres), dtype=np.float32)
            self.w = self.w / np.sum(self.w)
        else:
            self.w = w
        self.smooth = smooth
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(self.thres, w=w, reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(
            self: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ):
        loss = self.dice(inputs, targets) * self.mse(inputs, targets)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def initialization(self):
        self._initialization()


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            device: typing.Union[int, str, None] = None
    ):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            device=device
        )
        self.bn1 = nn.BatchNorm1d(num_features=hidden_channels, device=device)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            device=device
        )
        self.bn2 = nn.BatchNorm1d(num_features=hidden_channels, device=device)
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            device=device
        )
        self.bn3 = nn.BatchNorm1d(num_features=out_channels, device=device)
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=out_channels, device=device)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
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
    def __init__(
            self: nn.Module,
            n_features: int,
            n_labels: int,
            device: typing.Union[int, str, None] = None
    ):
        super(ResNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=N_FEATURES[0],
                kernel_size=7,
                stride=1,
                padding=3,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=N_FEATURES[0], device=device),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.__make_layer(
            in_channels=N_FEATURES[1][1],
            hidden_channels=N_FEATURES[1][2],
            out_channels=N_FEATURES[1][3],
            n_blocks=N_FEATURES[1][0],
            device=device
        )
        self.layer2 = self.__make_layer(
            in_channels=N_FEATURES[2][1],
            hidden_channels=N_FEATURES[2][2],
            out_channels=N_FEATURES[2][3],
            n_blocks=N_FEATURES[2][0],
            device=device
        )
        self.layer3 = self.__make_layer(
            in_channels=N_FEATURES[3][1],
            hidden_channels=N_FEATURES[3][2],
            out_channels=N_FEATURES[3][3],
            n_blocks=N_FEATURES[3][0],
            device=device
        )
        self.layer4 = self.__make_layer(
            in_channels=N_FEATURES[4][1],
            hidden_channels=N_FEATURES[4][2],
            out_channels=N_FEATURES[4][3],
            n_blocks=N_FEATURES[4][0],
            device=device
        )
        self.output = nn.Sequential(
            nn.Conv1d(
                in_channels=N_FEATURES[4][3],
                out_channels=n_labels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=n_labels, device=device),
            nn.ReLU(inplace=True)
        )
        self._initialization()

    def __make_layer(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            device: typing.Union[int, str, None] = None
    ) -> nn.Sequential:
        layers = [Block(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            device=device
        )]
        for i in range(1, n_blocks):
            layers.append(Block(
                in_channels=out_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                device=device
            ))
        return nn.Sequential(*layers)

    def _initialization(self: nn.Module):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = self.input(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.output(out)
        return out

    def initialization(self: nn.Module):
        self._initialization()


class CNN(nn.Module):
    def __init__(
            self: nn.Module,
            n_features: int,
            n_labels: int,
            device: typing.Union[int, str, None] = None
    ):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=64, device=device),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv1d(
            #     in_channels=64,
            #     out_channels=64,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     dilation=1,
            #     groups=64,
            #     bias=False,
            #     device=device
            # ),
            # nn.BatchNorm1d(num_features=64, device=device),
            # nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=128, device=device),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            # nn.Conv1d(
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     dilation=1,
            #     groups=128,
            #     bias=False,
            #     device=device
            # ),
            # nn.BatchNorm1d(num_features=128, device=device),
            # nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=256, device=device),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=n_labels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            nn.BatchNorm1d(num_features=n_labels, device=device),
            nn.ReLU(inplace=True)
        )
        self._initialization()

    def _initialization(self: nn.Module):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def initialization(self: nn.Module):
        self._initialization()


class Acc:
    def __init__(
            self,
            thres: typing.Union[typing.Tuple, typing.List, np.ndarray, None],
            ob: np.ndarray,
            pr: np.ndarray
    ):
        self.thres = thres
        self.n_grades = len(self.thres) + 1
        self.ob = ob
        self.pr = pr
        self.ob_grade = np.zeros_like(self.ob, dtype=np.int_) - 1
        self.ob_grade[~np.isnan(self.ob)] = 0
        for i in range(self.n_grades - 1):
            self.ob_grade[self.ob >= self.thres[i]] = i + 1
        self.pr_grade = np.zeros_like(self.pr, dtype=np.int_) - 1
        self.pr_grade[~np.isnan(self.pr)] = 0
        for i in range(self.n_grades - 1):
            self.pr_grade[self.pr >= self.thres[i]] = i + 1
        self.hxjz = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.hxjz[i, j] = np.sum((self.ob_grade == i) & (self.pr_grade == j))
        self.n = np.sum(self.hxjz)

    def get_me(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        index = self.pr + self.ob > 0
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_fs(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob_grade[index]
        pr = self.pr_grade[index]
        return float(100 - 40 * np.mean(np.abs((pr - ob) / len(self.thres))))

    def get_r(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.corrcoef(ob, pr)[0, 1])

    def get_ts(self) -> np.ndarray:
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        ts = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_grade > i) & (self.ob_grade > i))
            nb = np.sum((self.pr_grade > i) & (self.ob_grade <= i))
            nc = np.sum((self.pr_grade <= i) & (self.ob_grade > i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets(self) -> np.ndarray:
        ets = np.zeros(self.n_grades , dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            nd = np.sum((self.pr_grade != i) & (self.ob_grade != i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_ets2(self) -> np.ndarray:
        ets = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_grade > i) & (self.ob_grade > i))
            nb = np.sum((self.pr_grade > i) & (self.ob_grade <= i))
            nc = np.sum((self.pr_grade <= i) & (self.ob_grade > i))
            nd = np.sum((self.pr_grade <= i) & (self.ob_grade <= i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_bias(self) -> np.ndarray:
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            bias[i] = (na + nb) / (na + nc) if na + nc != 0 else np.nan
        return bias

    def get_far(self) -> np.ndarray:
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            far[i] = nb / (na + nb) if na + nb != 0 else np.nan
        return far

    def get_mar(self) -> np.ndarray:
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i + 1) & (self.ob_grade == i + 1))
            nc = np.sum((self.pr_grade != i + 1) & (self.ob_grade == i + 1))
            mar[i] = nc / (na + nc) if na + nc != 0 else np.nan
        return mar

    def get_hxjz(self) -> np.ndarray:
        return self.hxjz

    def get_oa(self) -> float:
        return np.sum(np.diag(self.hxjz)) / self.n

    def get_kappa(self) -> float:
        a = np.sum(self.hxjz, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.hxjz, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.n / self.n
        return (self.get_oa() - pe) / (1 - pe)


def interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    if is_0_84:
        for i in range(24):
            data[:, 3 * i + 1, :, :] = data[:, 3 * i, :, :] * 2 / 3 + data[:, 3 * i + 3, :, :] / 3
            data[:, 3 * i + 2, :, :] = data[:, 3 * i, :, :] / 3 + data[:, 3 * i + 3, :, :] * 2 / 3
        data[:, 73, :, :] = data[:, 72, :, :] * 5 / 6 + data[:, 78, :, :] / 6
        data[:, 74, :, :] = data[:, 72, :, :] * 2 / 3 + data[:, 78, :, :] / 3
        data[:, 75, :, :] = data[:, 72, :, :] / 2 + data[:, 78, :, :] / 2
        data[:, 76, :, :] = data[:, 72, :, :] / 3 + data[:, 78, :, :] * 2 / 3
        data[:, 77, :, :] = data[:, 72, :, :] / 6 + data[:, 78, :, :] * 5 / 6
        data[:, 79, :, :] = data[:, 78, :, :] * 5 / 6 + data[:, 84, :, :] / 6
        data[:, 80, :, :] = data[:, 78, :, :] * 2 / 3 + data[:, 84, :, :] / 3
        data[:, 81, :, :] = data[:, 78, :, :] / 2 + data[:, 84, :, :] / 2
        data[:, 82, :, :] = data[:, 78, :, :] / 3 + data[:, 84, :, :] * 2 / 3
        data[:, 83, :, :] = data[:, 78, :, :] / 6 + data[:, 84, :, :] * 5 / 6
    else:
        for i in range(8):
            data[:, 3 * i + 1, :, :] = data[:, 3 * i, :, :] * 2 / 3 + data[:, 3 * i + 3, :, :] / 3
            data[:, 3 * i + 2, :, :] = data[:, 3 * i, :, :] / 3 + data[:, 3 * i + 3, :, :] * 2 / 3
    return data


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter "second" cannot be negative.')
    elif is_abbreviation:
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
            time_str = str(second / 60) + 'minutes'
        else:
            time_str = str(second / 3600) + 'hours'

    return time_str


def main():
    print(N_FEATURES)
    data_list = list()
    label_list = list()
    for i in range(2017, 2022):
        data_list.append(np.reshape(np.load(fr'D:\data\wind\nwp{i}.npy'), shape=(-1, 85 * 97 * 30)))
        label_list.append(np.reshape(np.load(fr'D:\data\wind\ob{i}.npy'), shape=(-1, 85 * 97 * 8)))
    train_data = np.reshape(np.vstack(data_list), shape=(-1, 85, 97, 30))
    train_data = interp(train_data[:, 12:37, :, :])[:, 1:, :, :]
    train_label = np.reshape(np.vstack(label_list), shape=(-1, 85, 97, 8))[:, 13:37, :, 3]
    val_data = interp(np.load(r'D:\data\wind\nwp2022.npy')[:, 12:37, :, :])[:, 1:, :, :]
    val_label = np.load(r'D:\data\wind\ob2022.npy')[:, 13:37, :, 3]
    train_label[train_label > 100] = np.nan
    val_label[val_label > 100] = np.nan
    train_data = np.reshape(np.transpose(train_data, axes=(0, 3, 1, 2)), shape=(-1, 30, 24))
    train_label = np.reshape(np.transpose(train_label, axes=(0, 2, 1)), shape=(-1, 1, 24))
    val_data = np.reshape(np.transpose(val_data, axes=(0, 3, 1, 2)), shape=(-1, 30, 24))
    val_label = np.reshape(np.transpose(val_label, axes=(0, 2, 1)), shape=(-1, 1, 24))
    print(np.nanmax(train_label))
    print(np.nanmax(val_label))
    index_train = np.ones(shape=train_data.shape[0], dtype=np.bool_)
    for i in range(24):
        index_train &= ~np.isnan(train_label[:, 0, i])
        for j in range(30):
            index_train &= ~np.isnan(train_data[..., j, i])
    index_val = np.ones(shape=val_data.shape[0], dtype=np.bool_)
    for i in range(24):
        index_val &= ~np.isnan(val_label[:, 0, i])
        for j in range(30):
            index_val &= ~np.isnan(val_data[..., j, i])
    train_x = train_data[index_train, ...]
    val_x = val_data[index_val, ...]
    data_mean = np.reshape(np.mean(train_x, axis=(0, 2)), shape=(1, -1, 1))
    data_std = np.reshape(np.std(train_x, axis=(0, 2)), shape=(1, -1, 1))
    train_x = (train_x - data_mean) / data_std
    val_x = (val_x - data_mean) / data_std
    train_y = train_label[index_train]
    val_y = val_label[index_val]
    print(train_x.shape)
    print(val_x.shape)
    print(train_y.shape)
    print(val_y.shape)
    train_dataset = Data.TensorDataset(torch.from_numpy(train_x).cuda(), torch.from_numpy(train_y).cuda())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataset = Data.TensorDataset(torch.from_numpy(val_x).cuda(), torch.from_numpy(val_y).cuda())
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = CNN(n_features=30, n_labels=1, device=DEVICE)
    w = np.ones(len(THRES1), dtype=np.float32)
    w = w / np.sum(w)
    loss_function = DiceMSELoss2(THRES1, w=w)
    q = (LR_EPOCHS[1] / LR_EPOCHS[0]) ** (1 / (LR_EPOCHS[2] - 1))
    for epoch in range(LR_EPOCHS[2]):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LR_EPOCHS[0] * q ** epoch,
            weight_decay=WEIGHT_DECAY
        )
        model.train()
        loss_function.train()
        train_yy = list()
        pred_yy = list()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            predict_y = model(batch_x)
            loss = loss_function(predict_y, batch_y)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            train_yy.append(batch_y.cpu())
            pred_yy.append(predict_y.detach().cpu())
        train_yy = torch.vstack(train_yy)
        pred_yy = torch.vstack(pred_yy)
        train_loss = loss_function(train_yy, pred_yy)
        acc = Acc(THRES1, train_yy.numpy(), pred_yy.numpy())
        w = 1 - np.array(acc.get_ts2())
        w = w / np.sum(w)
        model.eval()
        loss_function.eval()
        with torch.no_grad():
            val_yy = list()
            pred_y = list()
            for step, (batch_x, batch_y) in enumerate(val_loader):
                predict_y = model(batch_x)
                val_yy.append(batch_y.cpu())
                pred_y.append(predict_y.cpu())
            val_yy = torch.vstack(val_yy)
            pred_y = torch.vstack(pred_y)
            val_loss = loss_function(val_yy, pred_y)
        print(f'epoch {epoch + 1}: {train_loss:.4f} {val_loss:.4f}')
        loss_function = DiceMSELoss2(THRES1, w=w)
    model.eval()
    loss_function.eval()
    pred_y = list()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(val_loader):
            predict_y = model(batch_x)
            pred_y.append(predict_y.cpu())
    pred_y = torch.vstack(pred_y).numpy()
    print('MLP')
    acc = Acc(THRES1, val_y, pred_y)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_ts2())

    data_list = list()
    label_list = list()
    for i in range(2017, 2022):
        data_list.append(np.reshape(np.load(fr'D:\data\wind\nwp{i}.npy'), shape=(-1, 85 * 97 * 30)))
        label_list.append(np.reshape(np.load(fr'D:\data\wind\ob{i}.npy'), shape=(-1, 85 * 97 * 8)))
    train_data = np.reshape(np.vstack(data_list), shape=(-1, 85, 97, 30))
    train_data = interp(train_data[:, 12:37, :, :])[:, 1:, :, :]
    train_label = np.reshape(np.vstack(label_list), shape=(-1, 85, 97, 8))[:, 13:37, :, 7]
    val_data = interp(np.load(r'D:\data\wind\nwp2022.npy')[:, 12:37, :, :])[:, 1:, :, :]
    val_label = np.load(r'D:\data\wind\ob2022.npy')[:, 13:37, :, 7]
    train_label[train_label > 100] = np.nan
    val_label[val_label > 100] = np.nan
    train_data = np.reshape(np.transpose(train_data, axes=(0, 3, 1, 2)), shape=(-1, 30, 24))
    train_label = np.reshape(np.transpose(train_label, axes=(0, 2, 1)), shape=(-1, 1, 24))
    val_data = np.reshape(np.transpose(val_data, axes=(0, 3, 1, 2)), shape=(-1, 30, 24))
    val_label = np.reshape(np.transpose(val_label, axes=(0, 2, 1)), shape=(-1, 1, 24))
    print(np.nanmax(train_label))
    print(np.nanmax(val_label))
    index_train = np.ones(shape=train_data.shape[0], dtype=np.bool_)
    for i in range(24):
        index_train &= ~np.isnan(train_label[:, 0, i])
        for j in range(30):
            index_train &= ~np.isnan(train_data[..., j, i])
    index_val = np.ones(shape=val_data.shape[0], dtype=np.bool_)
    for i in range(24):
        index_val &= ~np.isnan(val_label[:, 0, i])
        for j in range(30):
            index_val &= ~np.isnan(val_data[..., j, i])
    train_x = train_data[index_train, ...]
    val_x = val_data[index_val, ...]
    data_mean = np.reshape(np.mean(train_x, axis=(0, 2)), shape=(1, -1, 1))
    data_std = np.reshape(np.std(train_x, axis=(0, 2)), shape=(1, -1, 1))
    train_x = (train_x - data_mean) / data_std
    val_x = (val_x - data_mean) / data_std
    train_y = train_label[index_train]
    val_y = val_label[index_val]
    print(train_x.shape)
    print(val_x.shape)
    print(train_y.shape)
    print(val_y.shape)
    train_dataset = Data.TensorDataset(torch.from_numpy(train_x).cuda(), torch.from_numpy(train_y).cuda())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataset = Data.TensorDataset(torch.from_numpy(val_x).cuda(), torch.from_numpy(val_y).cuda())
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = CNN(n_features=30, n_labels=1, device=DEVICE)
    w = np.ones(len(THRES2), dtype=np.float32)
    w = w / np.sum(w)
    loss_function = DiceMSELoss2(THRES2, w=w)
    q = (LR_EPOCHS[1] / LR_EPOCHS[0]) ** (1 / (LR_EPOCHS[2] - 1))
    for epoch in range(LR_EPOCHS[2]):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LR_EPOCHS[0] * q ** epoch,
            weight_decay=WEIGHT_DECAY
        )
        model.train()
        loss_function.train()
        train_yy = list()
        pred_yy = list()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            predict_y = model(batch_x)
            loss = loss_function(predict_y, batch_y)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            train_yy.append(batch_y.cpu())
            pred_yy.append(predict_y.detach().cpu())
        train_yy = torch.vstack(train_yy)
        pred_yy = torch.vstack(pred_yy)
        train_loss = loss_function(train_yy, pred_yy)
        acc = Acc(THRES2, train_yy.numpy(), pred_yy.numpy())
        w = 1 - np.array(acc.get_ts2())
        w = w / np.sum(w)
        model.eval()
        loss_function.eval()
        with torch.no_grad():
            val_yy = list()
            pred_y = list()
            for step, (batch_x, batch_y) in enumerate(val_loader):
                predict_y = model(batch_x)
                val_yy.append(batch_y.cpu())
                pred_y.append(predict_y.cpu())
            val_yy = torch.vstack(val_yy)
            pred_y = torch.vstack(pred_y)
            val_loss = loss_function(val_yy, pred_y)
        print(f'epoch {epoch + 1}: {train_loss:.4f} {val_loss:.4f}')
        loss_function = DiceMSELoss2(THRES2, w=w)
    model.eval()
    loss_function.eval()
    pred_y = list()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(val_loader):
            predict_y = model(batch_x)
            pred_y.append(predict_y.cpu())
    pred_y = torch.vstack(pred_y).numpy()
    print('MLP')
    acc = Acc(THRES2, val_y, pred_y)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_ts2())


if __name__ == '__main__':
    print('The program "cnn_demo.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "cnn_demo.py" runs out in {:s}.'.format(format_time(running_time)))
