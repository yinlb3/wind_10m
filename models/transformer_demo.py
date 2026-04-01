#!user/bin.python3

"""
Founded in 2025-09-09
Modified in 2026-02-01
@author: yinlb
"""
import os
import math
import sys
import typing

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd

import torch
from torch import nn
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
LR_EPOCHS = (1e-2, 1e-4, 100)
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


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(
            self: nn.Module,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000,
            device: typing.Union[int, str, None] = None
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(dim0=0, dim1=1)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :].transpose(dim0=1, dim1=2)
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
            self: nn.Transformer,
            nfeat: int,
            nlabel: int,
            ninp: int = 512,
            nhead: int = 8,
            nhid: int = 2048,
            nlayers: int = 6,
            dropout: float = 0.5,
            device: typing.Union[int, str, None] = None
    ):
        super(TransformerModel, self).__init__(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            num_encoder_layers=nlayers,
            batch_first=True,
            device=device
        )
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(
            d_model=ninp,
            dropout=dropout,
            device=device
        )
        self.input_emb = nn.Sequential(
            nn.Conv1d(
                in_channels=nfeat,
                out_channels=ninp,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            # nn.BatchNorm1d(num_features=ninp, device=device),
            nn.LayerNorm(normalized_shape=[ninp, 24], device=device),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=ninp,
                out_channels=nlabel,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                device=device
            ),
            # nn.BatchNorm1d(num_features=nlabel, device=device),
            nn.LayerNorm(normalized_shape=[nlabel, 24], device=device),
            nn.ReLU()
        )
        self.ninp = ninp
        self.init_weights()

    def _generate_square_subsequent_mask(self: nn.Transformer, sz: int):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self: nn.Transformer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(tensor=m.weight, a=-0.1, b=0.1)
                nn.init.zeros_(tensor=m.bias)

    def forward(self: nn.Transformer, src: torch.Tensor) -> torch.Tensor:
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.transpose(dim0=1, dim1=2)
        out = self.encoder(src)
        out = out.transpose(dim0=1, dim1=2)
        out = self.decoder(out)
        return out


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
    model = TransformerModel(nfeat=30, nlabel=1, device=DEVICE)
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
    model = TransformerModel(nfeat=30, nlabel=1, device=DEVICE)
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
    print('The program "transformer_demo.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "transformer_demo.py" runs out in {:s}.'.format(format_time(running_time)))
