import numpy as np
import torch
import os
import sys

from typing import NoReturn
import abc
from typing import Optional, Dict
import pandas as pd


class LargeScheduler:
    def __init__(self, args, optimizer) -> None:
        super().__init__()
        self.learning_rate = args.learning_rate
        self.decay_fac = args.decay_fac
        self.lradj = args.lradj
        self.use_multi_gpu = args.use_multi_gpu
        self.optimizer = optimizer
        self.args = args
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None
        self.warmup_steps = args.warmup_steps

    def schedule_epoch(self, epoch: int):
        if self.lradj == 'type1':
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'type2':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch - 1) // 1))}
        elif self.lradj == 'type4':
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch) // 1))}
        elif self.lradj == 'type3':
            self.learning_rate = 1e-4
            lr_adjust = {epoch: self.learning_rate if epoch < 3 else self.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif self.lradj == 'cos_epoch':
            lr_adjust = {epoch: self.learning_rate / 2 * (1 + math.cos(epoch / self.args.cos_max_decay_epoch * math.pi))}
        else:
            return

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def schedule_step(self, n: int):
        if self.lradj == 'cos_step':
            if n < self.args.warmup_steps:
                res = (self.args.cos_max - self.learning_rate) / self.args.warmup_steps * n + self.learning_rate
                self.last = res
            else:
                t = (n - self.args.warmup_steps) / (self.args.cos_max_decay_steps - self.args.warmup_steps)
                t = min(t, 1.0)
                res = self.args.cos_min + 0.5 * (self.args.cos_max - self.args.cos_min) * (1 + np.cos(t * np.pi))
                self.last = res
        elif self.lradj == 'constant_with_warmup':
            if n < self.warmup_steps:
                # Linear warmup
                res = self.learning_rate * n / max(1, self.warmup_steps)
            else:
                # Constant learning rate after warmup
                res = self.learning_rate
        else:
            return

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = res
        if n % 500 == 0:
            print('Updating learning rate to {}'.format(res))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
    # def __call__(self, val_loss, model, prototype_buffer, normal_mean, normal_cov, abnormal_mean, abnormal_cov, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            # self.save_checkpoint(val_loss, model, prototype_buffer, normal_mean, normal_cov, abnormal_mean, abnormal_cov, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            # self.save_checkpoint(val_loss, model, prototype_buffer, normal_mean, normal_cov, abnormal_mean, abnormal_cov, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    # def save_checkpoint(self, val_loss, model, prototype_buffer, normal_mean, normal_cov, abnormal_mean, abnormal_cov, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'prototype_buffer': prototype_buffer,
        #     'normal_mean': normal_mean,  # Move to CPU before saving
        #     'normal_cov': normal_cov,
        #     'abnormal_mean': abnormal_mean,
        #     'abnormal_cov': abnormal_cov,
        #     }, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class HiddenPrints:
    def __init__(self, rank):
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Singleton(type):
    """
    Used to construct singleton classes through the method of meta classes
    """

    _instance_dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_dict:
            cls._instance_dict[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance_dict[cls]


class DataPoolImpl(metaclass=abc.ABCMeta):
    """
    Base class for data pool implementations

    This class acts as a data client in each worker that feeds data to the pipeline.
    Techniques such as local caching may be adopted to improve performance.
    """

    @abc.abstractmethod
    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        Gets time series data by name

        The returned DataFrame follows the OTB protocol.

        :param name: The name of the series to get.
        :return: Time series data in DataFrame format. If the time series is not available,
            return None.
        """

    @abc.abstractmethod
    def get_covariates(self, name: str) -> Optional[Dict]:
        """
        Gets time series' covariates by name

        :param name: The name of the series to get.
        :return: The dictionary of series' covariates. If the covariates is not available,
            return None.
        """

    @abc.abstractmethod
    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        Gets the meta information of time series by name

        We do not return the meta information of unexisting series even if
        the meta information itself is available.

        :param name: The name of the series to get.
        :return: Meta information data in Series format. If the meta information or the
            corresponding series is not available, return None.
        """


class DataPool(metaclass=Singleton):
    """
    The global interface of data pools
    """

    def __init__(self):
        self.pool = None

    def set_pool(self, pool: DataPoolImpl) -> NoReturn:
        """
        Set the global data pool object

        :param pool: a DataPoolImpl object.
        """
        self.pool = pool

    def get_pool(self) -> DataPoolImpl:
        """
        Get the global data pool object
        """
        return self.pool