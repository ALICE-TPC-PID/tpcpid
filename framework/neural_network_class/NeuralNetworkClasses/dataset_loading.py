"""Tensor-backed datasets and bounded, vectorized batch loading."""
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


class dataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same number of rows')
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def mem_size(self):
        return self.X.numel() * self.X.element_size() + self.y.numel() * self.y.element_size()


class TensorBatchSampler(Sampler):
    """Keep shuffle indices in one int64 tensor, never a Python list per row.

    Training pads equally across ranks like DistributedSampler. Validation does
    not pad, so each observation contributes exactly once to the global metric.
    """
    def __init__(self, size, batch_size, shuffle=False, rank=0, world_size=1,
                 pad=False, seed=0):
        self.size, self.batch_size = size, int(batch_size)
        self.shuffle, self.rank, self.world_size = shuffle, rank, world_size
        self.pad, self.seed, self.epoch = pad, seed, 0
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        rows = ((self.size + self.world_size - 1) // self.world_size if self.pad
                else len(range(self.rank, self.size, self.world_size)))
        return (rows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = None
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.size, generator=generator)
        rows = ((self.size + self.world_size - 1) // self.world_size if self.pad
                else len(range(self.rank, self.size, self.world_size)))
        for start in range(0, rows, self.batch_size):
            stop = min(start + self.batch_size, rows)
            first, end = self.rank + start*self.world_size, self.rank + stop*self.world_size
            if order is None and end <= self.size + self.world_size - 1:
                yield slice(first, min(end, self.size), self.world_size)
            else:
                indices = torch.arange(first, end, self.world_size) % self.size
                yield order[indices] if order is not None else indices


def batch_loader(data, batch_size, shuffle=False, rank=0, world_size=1,
                 pad=False, num_workers=0, pin_memory=False):
    if data.X.is_cuda and num_workers:
        raise ValueError('CUDA-resident datasets require num_workers=0')
    sampler = TensorBatchSampler(len(data), batch_size, shuffle, rank, world_size, pad)
    loader = DataLoader(data, batch_size=None, sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=bool(pin_memory and data.X.device.type == 'cpu'),
                        persistent_workers=num_workers > 0)
    return loader, sampler


class DataLoading:
    def __init__(self, training_data, validation_data, batch_sizes=None, num_workers=0,
                 X_data_scalers=None, y_data_scalers=None, transform_data=True,
                 shuffle_every_epoch=True, copy_to_device=False, verbose=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and copy_to_device else 'cpu')
        self.num_workers = num_workers
        self.transform_data = transform_data
        self.shuffle_every_epoch = shuffle_every_epoch
        self.batch_sizes = [1] if batch_sizes is None else batch_sizes
        if not self.batch_sizes or any(int(n) <= 0 for n in self.batch_sizes):
            raise ValueError('batch_sizes must contain positive integers')
        if X_data_scalers is None:
            X_data_scalers = [('box-cox', preprocessing.PowerTransformer(method='box-cox'))]
        if y_data_scalers is None:
            y_data_scalers = [('standard scaler', preprocessing.StandardScaler())]
        self.scalingX = ScalingX(X_data_scalers if transform_data else [], copy_to_dev=False)
        self.scalingY = ScalingY(y_data_scalers if transform_data else [], copy_to_dev=False)
        self.datasetTS = dataset(self.scalingX.scale(training_data[0]).to(self.device),
                                 self.scalingY.scale(training_data[1]).to(self.device))
        self.datasetVS = dataset(self.scalingX.scale(validation_data[0]).to(self.device),
                                 self.scalingY.scale(validation_data[1]).to(self.device))
        self.inverse_X = InverseScaling(self.scalingX.fitted_scalers_X, copy_to_dev=False)
        self.inverse_Y = InverseScaling(self.scalingY.fitted_scalers_y, copy_to_dev=False)
        self.sizeTS, self.sizeVS = self.datasetTS.mem_size(), self.datasetVS.mem_size()
        self.loadTS = self.loadVS = True
        if verbose:
            print(f'Training: {len(self.datasetTS):,} rows, {self.sizeTS/2**20:.1f} MiB; '
                  f'validation: {len(self.datasetVS):,} rows, {self.sizeVS/2**20:.1f} MiB; storage: {self.device}')


class _Scaling:
    def __init__(self, scalers=None, newscale=True, copy_to_dev=True):
        self.scalers = deepcopy(scalers or [])
        self.newscale, self.copy_to_dev = newscale, copy_to_dev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fitted = False

    def scale(self, data):
        if self.newscale:
            self.fitted = Pipeline(self.scalers).fit(data) if self.scalers else False
            self.newscale = False
        # Compatibility attributes used by existing callers.
        self.pipe_X = self.fitted_scalers_X = self.pipe_y = self.fitted_scalers_y = self.fitted
        transformed = self.fitted.transform(data) if self.fitted else data
        result = torch.as_tensor(transformed, dtype=torch.float32)
        return result.to(self.device) if self.copy_to_dev else result


class ScalingX(_Scaling):
    pass


class ScalingY(_Scaling):
    pass


class InverseScaling:
    def __init__(self, scalers_fit, copy_to_dev=True):
        self.scalers_fit, self.copy_to_dev = scalers_fit, copy_to_dev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def scale(self, data):
        device = data.device if isinstance(data, torch.Tensor) else self.device
        values = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
        result = self.scalers_fit.inverse_transform(values) if self.scalers_fit else values
        output = torch.as_tensor(result, dtype=torch.float32)
        return output.to(device) if self.copy_to_dev else output

    __call__ = scale
