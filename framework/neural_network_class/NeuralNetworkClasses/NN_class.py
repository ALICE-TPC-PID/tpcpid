from copy import deepcopy
import os
import timeit
import socket
import subprocess
import onnx

import numpy as np

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .custom_loss_functions import *
from .dataset_loading import batch_loader

### NN: A class for training a Neural network and predicting output (so to say a wrapper class for a General_NN)


class NN():

    def __init__(self, neural_net):
        self.network = neural_net.float()
        self.rank = 0

    def __call__(self, X):
        return self.network(X)

    def forward(self, X):
        if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
            model = self.network.module
        else:
            model = self.network
        return model(X)

    def training(self, data, multigpu=-1, pin_memory=1, epochs=1, epochs_ls=None,
                 optimizer=optim.Adam, scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                 learning_rate=0.01, weight_decay=0, loss_function=None, weights=False,
                 verbose=True, nsamples=np.inf, copy_to_device=1, patience=5, factor=0.5,
                 shuffle_every_epoch=None, device=None, cpu_threads=1,
                 validation_batch_size=65536):
        """Train with bounded batches and sample-weighted, detached epoch metrics.

        nsamples retains its historical meaning as a maximum number of training
        batches (now exact). Loss functions must return a scalar batch mean.
        """
        self.rank, self.worldsize = 0, 1
        launched = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
        self.multigpu = launched > 1 if multigpu == -1 else bool(multigpu)
        if cpu_threads is not None:
            torch.set_num_threads(int(cpu_threads))
        if not data.loadTS or not data.loadVS or not len(data.datasetTS) or not len(data.datasetVS):
            raise ValueError('Nonempty training and validation datasets are required')
        self.epochs_ls = [0] if epochs_ls is None else list(epochs_ls)
        if (not self.epochs_ls or self.epochs_ls[0] != 0 or
                self.epochs_ls != sorted(set(self.epochs_ls)) or
                len(self.epochs_ls) != len(data.batch_sizes)):
            raise ValueError('epochs_ls must start at zero, increase, and match batch_sizes')
        if nsamples <= 0 or (np.isfinite(nsamples) and int(nsamples) != nsamples):
            raise ValueError('nsamples must be a positive number of batches or infinity')
        if self.multigpu:
            self.multigpu = multigpu
            self.rank, self.worldsize = self.multigpu_training_setup()
            self.multigpu = True
        else:
            if launched > 1:
                raise ValueError('Multiple processes launched but distributed training disabled')
            if device is None:
                device = ('cuda:0' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
            self.device = str(device)
            self.network.to(self.device)
        self.verbose = verbose and self.rank == 0
        self.epochs = epochs
        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = scheduler(self.optimizer, patience=patience, factor=factor)
        self.loss_function = nn.MSELoss() if loss_function is None else loss_function
        model = self.network.module if isinstance(self.network, DDP) else self.network
        if data.transform_data:
            model.scaling_X, model.scaling_y = data.scalingX, data.scalingY
            model.inverse_X, model.inverse_Y = data.inverse_X, data.inverse_Y
        self.pin_memory = bool(pin_memory is not False and pin_memory != 0 and 'cuda' in self.device)
        shuffle = data.shuffle_every_epoch if shuffle_every_epoch is None else shuffle_every_epoch
        validation_loader, _ = batch_loader(data.datasetVS, validation_batch_size,
            rank=self.rank, world_size=self.worldsize, num_workers=data.num_workers,
            pin_memory=self.pin_memory, seed=data.seed)
        training_loss, validation_loss = [], []
        train_loader = None

        def batch_loss(network, X, y):
            if weights:
                # Recover weights before transferring CPU features to the GPU.
                batch_weights = (data.inverse_X(X)[:, -1] if data.transform_data
                                 else X[:, -1]).to(self.device, non_blocking=self.pin_memory)
                X = X[:, :-1]
            if copy_to_device:
                X = X.to(self.device, non_blocking=self.pin_memory)
                y = y.to(self.device, non_blocking=self.pin_memory)
            if weights:
                return self.loss_function(network(X), y, weights=batch_weights)
            return self.loss_function(network(X), y)

        try:
            for epoch in range(int(epochs)):
                start = timeit.default_timer()
                if epoch in self.epochs_ls:
                    idx = self.epochs_ls.index(epoch)
                    train_loader, sampler = batch_loader(data.datasetTS, data.batch_sizes[idx],
                        shuffle=shuffle, rank=self.rank, world_size=self.worldsize,
                        pad=self.multigpu, num_workers=data.num_workers,
                        pin_memory=self.pin_memory, seed=data.seed)
                sampler.set_epoch(epoch)
                self.network.train()
                model.mode = 'train'
                # Accumulate detached tensors on-device; one synchronization per epoch.
                totals = torch.zeros(4, device=self.device, dtype=torch.float64)
                for counter, (X, y) in enumerate(train_loader):
                    if counter >= nsamples:
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = batch_loss(self.network, X, y)
                    loss.backward()
                    self.optimizer.step()
                    totals[0] += loss.detach() * len(X)
                    totals[1] += len(X)
                self.network.eval()
                model.mode = 'eval'
                # Evaluate the unwrapped model: ranks can have unequal validation
                # batch counts. Synchronize buffers once, not in each forward.
                if self.multigpu:
                    for buffer in model.buffers():
                        dist.broadcast(buffer, src=0)
                with torch.no_grad():
                    for X, y in validation_loader:
                        loss = batch_loss(model, X, y)
                        totals[2] += loss * len(X)
                        totals[3] += len(X)
                if self.multigpu:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                tr_sum, tr_count, val_sum, val_count = totals.cpu().tolist()
                tr_loss, val_loss = tr_sum/tr_count, val_sum/val_count
                training_loss.append(tr_loss)
                validation_loss.append(val_loss)
                self.scheduler.step(val_loss)
                if self.verbose:
                    print(f'Epoch {epoch+1}/{epochs} | Batch size: {data.batch_sizes[idx]} '
                          f'| Training: {tr_loss:.6g} | Validation: {val_loss:.6g} '
                          f'| {timeit.default_timer()-start:.3f} s', flush=True)
        finally:
            if self.multigpu and dist.is_initialized():
                dist.destroy_process_group()
        self.training_loss, self.validation_loss = training_loss, validation_loss
        self.network.eval()
        model.mode = 'eval'


    def save_losses(self, path=["./training_loss.txt", "./validation_loss.txt"]):

        if self.rank == 0:
            np.savetxt(path[0], self.training_loss)
            np.savetxt(path[1], self.validation_loss)
            print("Training and validation loss saved!")

    def eval(self):

        self.network.mode='eval'
        self.network = self.network.eval()

    def multigpu_training_setup(self):
        """Use torchrun or Slurm rendezvous without a racy node-local host file."""
        rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
        if self.multigpu > 1 and self.multigpu != world_size:
            raise ValueError('Requested GPU count does not match launched process count')
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed GPU training requires CUDA/ROCm')
        if 'MASTER_ADDR' not in os.environ:
            nodes = os.environ.get('SLURM_JOB_NODELIST')
            if int(os.environ.get('SLURM_NNODES', '1')) == 1 and nodes:
                os.environ['MASTER_ADDR'] = socket.gethostname()
            elif nodes:
                os.environ['MASTER_ADDR'] = subprocess.check_output(
                    ['scontrol', 'show', 'hostnames', nodes], text=True).splitlines()[0]
            elif world_size == 1:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
            else:
                raise ValueError('Set MASTER_ADDR or launch with Slurm/torchrun')
        job_id = int(os.environ.get('SLURM_JOB_ID', '0'))
        os.environ.setdefault('MASTER_PORT', str(20000 + job_id % 20000))
        os.environ['RANK'], os.environ['WORLD_SIZE'] = str(rank), str(world_size)
        # Slurm may expose just one GPU per task, or all GPUs on the node.
        gpu = 0 if torch.cuda.device_count() == 1 else local_rank
        torch.cuda.set_device(gpu)
        self.device = f'cuda:{gpu}'
        dist.init_process_group(backend='nccl', init_method='env://')
        self.network = DDP(self.network.to(self.device), device_ids=[gpu], output_device=gpu)
        return rank, world_size

    def save_net(self, path="./net.pt", avoid_q=False):

        if self.rank == 0:
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                model = self.network.module
            else:
                model = self.network

            checkpoint = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "training_loss": getattr(self, "training_loss", None),
                "validation_loss": getattr(self, "validation_loss", None),
            }

            if hasattr(self, "optimizer"):
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

            if hasattr(self, "scheduler"):
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

            if not avoid_q and os.path.isfile(path):
                response = input("File exists. Do you want to overwrite it? [y/n] ")
                if response.lower() not in ["y", "yes"]:
                    print("Network not saved!")
                    return

            torch.save(checkpoint, path)
            print("Network saved")

    def load_net(self, path, map_location="cpu", load_optimizer=False, load_scheduler=False):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        model = self.network.module if isinstance(
            self.network, torch.nn.parallel.DistributedDataParallel
        ) else self.network

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            if load_optimizer and hasattr(self, "optimizer") and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if load_scheduler and hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "training_loss" in checkpoint:
                self.training_loss = checkpoint["training_loss"]
            if "validation_loss" in checkpoint:
                self.validation_loss = checkpoint["validation_loss"]
        else:
            model.load_state_dict(checkpoint)

        model.eval()

    def jit_script_model(self):

        self._jit_script_model = torch.jit.script(self.network)
        print("Model converted to jit_script, saved in self.jit_script_model")


    def save_jit_script(self, path="./net_jit_script.pt"):

        if self.rank == 0:
            self.jit_script_model()
            torch.jit.save(self._jit_script_model, path)

            print("Model saved!")


    def save_onnx(self, example_data=torch.tensor([[]],requires_grad=True).float(), path="./net_onnx.onnx"):

        if self.rank == 0:
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                model = self.network.module
            else:
                model = self.network

            model = deepcopy(model).to("cpu", dtype=torch.float32).eval()
            example_data = example_data.to(device="cpu", dtype=torch.float32)

            torch.onnx.export(model,                                            # model being run
                                example_data,                                   # model input (or a tuple for multiple inputs)
                                path,                                           # where to save the model (can be a file or file-like object)
                                export_params=True,                             # store the trained parameter weights inside the model file
                                external_data=False,                            # Stores the model weights in the same file
                                opset_version=14,                               # the ONNX version to export the model to: https://onnxruntime.ai/docs/reference/compatibility.html
                                do_constant_folding=True,                       # whether to execute constant folding for optimization
                                input_names=['input'],                          # the model's input names
                                output_names=['output'],                        # the model's output names
                                dynamo=True,                                    # Disable torchdynamo for export: FIXME This needs to be tested with dynamic_axes=... being changed to dynamic_shapes=({0: torch.export.Dim("batch_size")},),
                                dynamic_shapes={"x": {0: torch.export.Dim("batch_size")}}
            )


    def check_onnx(self, path="./net_onnx.onnx"):
        if self.rank == 0:
            onnx.checker.check_model(onnx.load(path))
            print("ONNX checker: Success!")
