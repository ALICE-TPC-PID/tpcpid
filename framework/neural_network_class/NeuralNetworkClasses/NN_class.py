import os
import timeit
import socket
import onnx

import numpy as np

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .custom_loss_functions import *

### NN: A class for training a Neural network and predicting output (so to say a wrapper class for a General_NN)


class NN():

    def __init__(self, neural_net):
        self.network = neural_net.float()

    def __call__(self, X):
        return self.network(X)

    def forward(self, X):
        if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
            model = self.network.module
        else:
            model = self.network
        return model(X)

    def training(self, data, multigpu=-1, pin_memory=1, epochs=1, epochs_ls=[0],
                 optimizer=optim.Adam, scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                 learning_rate=0.01, weight_decay=0, loss_function=nn.MSELoss(), weights=False,
                 verbose=True, nsamples=np.inf, copy_to_device = 1, patience=5, factor=0.5):

        self.rank = 0
        self.worldsize = 1
        self.multigpu = multigpu
        self.verbose = verbose and (int(os.environ.get("SLURM_PROCID", "0"))==0)

        if self.multigpu:
            self.rank, self.worldsize = self.multigpu_training_setup()

        if self.verbose:
            print("\n============ Neural Network training ============\n")

        ### Setting the device on which to run ###

        if not self.multigpu:
            if self.settings["MACHINE_OPTIONS"]["device"] is None:
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    # if self.settings["MACHINE_OPTIONS"]["device_id"] is None or "all":
                    #     self.device = "cuda"
                    # elif type(self.settings["MACHINE_OPTIONS"]["device_id"]) == type(list()):
                    #     self.device = "cuda:" + ",".join(list(map(str, self.settings["MACHINE_OPTIONS"]["device_id"])))
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.settings["MACHINE_OPTIONS"]["device"]

            self.network.to(device=self.device, dtype=self.dtype)

        ### Printing device information
        if self.multigpu:
            if self.verbose and self.rank == 0:
                print("[Multi-GPU training]", self.worldsize, "participating GPUs.")
                for i in range(self.worldsize):
                    dev_id = torch.cuda.device_count()
                    dev_name = torch.cuda.get_device_name(i)
                    print("Device ID: ", i, ", Device name: ", dev_name)
        else:
            if ("cuda" in self.device):
                dev_id = torch.cuda.current_device()  # returns you the ID of your current device
                dev_name = torch.cuda.get_device_name(dev_id)
                # dev_mem_use = torch.cuda.memory_allocated(dev_id)          #returns you the current GPU memory usage by tensors in bytes for a given device
                # dev_mem_man = torch.cuda.memory_reserved(dev_name)         #returns you the current GPU memory managed by caching allocator in bytes for a given device
                torch.cuda.empty_cache()  # clear variables in cache that are unused
                if self.verbose:
                    print("\nRunning on GPU")
                    print("Device ID: ", dev_id, ", Device name: ", dev_name, "\n")
            elif self.device == "mps":
                if self.verbose:
                    print("\nRunning on MPS\n")
            else:
                if self.settings["MACHINE_OPTIONS"]["cpu_threads"] is not None:
                    torch.set_num_threads(int(self.settings["MACHINE_OPTIONS"]["cpu_threads"]))
                if self.verbose:
                    print("\nRunning on CPU")
                    print("{} CPU threads\n".format(torch.get_num_threads()))


        ### Setting some variables of the network ###

        self.epochs = epochs
        self.epochs_ls = epochs_ls
        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = scheduler(self.optimizer, patience=patience, factor=factor)
        self.loss_function = loss_function


        ### Checking if data was loaded properly and extracting the scalers ###

        if not data.loadTS or not data.loadVS:
            print("The data has not been loaded. Shutting down.")
            exit()

        if data.transform_data:
            self.network.scaling_X = data.scalingX
            self.network.scaling_y = data.scalingY
            self.network.inverse_X = data.inverse_X
            self.network.inverse_Y = data.inverse_Y

        self.pin_memory = pin_memory
        if self.pin_memory is None:
            self.pin_memory = ("cuda" in self.device or "mps" in self.device) # and (self.dtype == torch.float)


        ######################### Starting the training #########################

        self.network.mode='train'
        training_loss = []
        validation_loss = []

        for epoch in range(int(epochs)):

            start_time = timeit.default_timer() # Timer start

            if epoch in self.epochs_ls:
                idx_tr = self.epochs_ls.index(epoch)
                if self.verbose:
                    print("\n--- Batch size:", self.epochs_ls[idx_tr], "---\n")

            if self.multigpu:
                dist.barrier()
                sampler = DistributedSampler(data.datasetTS, num_replicas=self.worldsize, rank=self.rank, shuffle=True)
                sampler.set_epoch(epoch)
            else:
                sampler = None

            ### Iterating through the training data ###

            train_dataloader = DataLoader(
                data.datasetTS,
                batch_size=data.batch_sizes[idx_tr],
                num_workers=data.num_workers,
                pin_memory=self.pin_memory,
                shuffle=(False if self.multigpu else self.settings["DATA_OPTIONS"]["shuffle_every_epoch"]),
                sampler=sampler
            )

            tr_loss = 0
            av_tr_loss = 0

            for counter, entry_tr in enumerate(train_dataloader, 0):
                BX, BY = entry_tr

                if (copy_to_device and self.device != "cpu"):
                    BX = BX.to(device=self.device)
                    BY = BY.to(device=self.device)

                if counter > nsamples:
                    break
                else:
                    if weights:
                        weights_array_tr = data.inverse_X(BX)[:,-1].flatten()
                        BX = BX[:,:-1]
                    else:
                        weights_array_tr = 1.

                    self.optimizer.zero_grad()
                    training_out = self.network(BX)
                    loss = self.loss_function(training_out, BY, weights=weights_array_tr)
                    loss.backward()
                    self.optimizer.step()
                    tr_loss += loss

                    # if verbose:
                    #     av_tr_loss += loss.item()
                    #     if counter % 1000 == 999:
                    #         print("{val1} minibatches. Average training loss: {val2}".format(val1=counter+1, val2 = np.round(av_tr_loss/1000.,6)))
                    #         av_tr_loss = 0

                if self.multigpu:
                    dist.barrier()

                del BX, BY
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            #----------------------------


            ### Iterating through the validation data ###

            validation_dataloader= DataLoader(data.datasetVS, batch_size=len(data.datasetVS),num_workers=data.num_workers, pin_memory=(not torch.cuda.is_available()), shuffle=data.shuffle_every_epoch)

            val_loss = 0
            for counter, entry_val in enumerate(validation_dataloader, 0):

                val_obs, val_label = entry_val

                if (copy_to_device and self.device != "cpu"):
                    val_obs = val_obs.to(device=self.device)
                    val_label = val_label.to(device=self.device)

                if weights:
                    weights_array_val = data.inverse_X(val_obs)[:,-1].flatten()
                    val_obs = val_obs[:,:-1].float()
                else:
                    weights_array_val = 1.

                validation_out = self.network(val_obs.float())
                val_loss += loss_function(validation_out, val_label, weights=weights_array_val).cpu()

            #    if verbose and counter % 1000 == 999:
            #        print("Gone through {} samples in validation set".format(counter+1))

            #----------------------------


            ### Prepare for output: Training loss and validation loss normalized to number of batches ###

            tr_loss = tr_loss.cpu()/len(train_dataloader)
            val_loss = val_loss.cpu()
            training_loss.append(float(tr_loss.detach().cpu()))
            validation_loss.append(float(val_loss.detach().cpu()))
            self.scheduler.step(val_loss.detach().item())

            #----------------------------


            end_time = timeit.default_timer() # Timer end


            ### Printing the stats ###
            if self.verbose:
                print('Epoch ', epoch+1, '/', epochs , '|'
                    ' Batch-size: ', data.batch_sizes[idx_tr],
                    ', Validation loss: ', np.round(val_loss.detach().numpy(),6),
                    ', Training loss: ', np.round(tr_loss.detach().numpy(),6),
                    ', Execution time: ', np.round(end_time-start_time,3), 's')

            #----------------------------

        #print(list(self.network.parameters()))
        if self.verbose:
            print("\nTraining finished!\n")

        ###########################################################################

        #self.network.to('cpu')
        self.training_loss = training_loss
        self.validation_loss = validation_loss

        self.network.mode='eval'

        if self.multigpu > 0:
            dist.destroy_process_group()
            self.delete_masteraddr_file()


    def save_losses(self, path=["./training_loss.txt", "./validation_loss.txt"]):

        if self.rank == 0:
            np.savetxt(path[0], self.training_loss)
            np.savetxt(path[1], self.validation_loss)
            print("Training and validation loss saved!")

    def eval(self):

        self.network.mode='eval'
        self.network = self.network.eval()

    def multigpu_training_setup(self):
        """
        Setup PyTorch DistributedDataParallel using Slurm environment variables.
        Automatically handles single-node and multi-node jobs.

        Assumes:
            - Slurm launches the job with srun
            - self.multigpu contains the requested total number of GPUs (world size)
            - 8 GPUs per node
        """

        # ----------------------------------------------------------------------
        # 1. Read Slurm variables
        # ----------------------------------------------------------------------
        slurm_procid   = int(os.environ["SLURM_PROCID"])    # global rank
        slurm_localid  = int(os.environ["SLURM_LOCALID"])   # local rank on this node
        slurm_ntasks   = int(os.environ["SLURM_NTASKS"])    # total tasks (world size)
        slurm_nodeid   = int(os.environ["SLURM_NODEID"])    # node index (0 .. nnodes-1)
        slurm_nnodes   = int(os.environ["SLURM_NNODES"])    # number of nodes
        slurm_jobid    = int(os.environ["SLURM_JOBID"])
        user           = os.environ["USER"]

        # Let user-provided n_gpus override Slurm if desired
        worldsize = self.multigpu if self.multigpu > 0 else slurm_ntasks

        # ----------------------------------------------------------------------
        # 2. Set PyTorch DDP expected environment variables
        # ----------------------------------------------------------------------
        os.environ["RANK"]       = str(slurm_procid)
        os.environ["WORLD_SIZE"] = str(worldsize)
        os.environ["LOCAL_RANK"] = str(slurm_localid)

        # ----------------------------------------------------------------------
        # 3. Determine MASTER_ADDR (multi-node safe)
        #    Only node 0 writes its hostname to a file shared across nodes.
        # ----------------------------------------------------------------------
        self.hostfile = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"master_addr_{user}_{slurm_jobid}.txt")

        if slurm_nodeid == 0:
            # Node 0 writes its hostname
            with open(self.hostfile, "w") as f:
                f.write(socket.gethostname())

        # Slurm ensures simultaneous startup → we simply read the file
        with open(self.hostfile, "r") as f:
            master_addr = f.read().strip()

        os.environ["MASTER_ADDR"] = master_addr

        # ----------------------------------------------------------------------
        # 4. Safe, collision-free master port
        # ----------------------------------------------------------------------
        master_port = 20000 + (slurm_jobid % 20000)
        os.environ["MASTER_PORT"] = str(master_port)

        # ----------------------------------------------------------------------
        # 5. Initialize Process Group
        # ----------------------------------------------------------------------
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        # ----------------------------------------------------------------------
        # 6. Bind process to its GPU
        # ----------------------------------------------------------------------
        torch.cuda.set_device(slurm_localid)
        self.device = f"cuda:{slurm_localid}"

        # ----------------------------------------------------------------------
        # 7. Wrap model in DDP
        # ----------------------------------------------------------------------
        self.network = DDP(self.network.to(self.device),
                        device_ids=[slurm_localid],
                        output_device=slurm_localid)

        return slurm_localid, worldsize

    def delete_masteraddr_file(self):
        """
        Delete the temporary master address file created during multi-node DDP setup.
        Only node 0 should perform the deletion.
        """

        if self.rank == 0:
            try:
                os.remove(self.hostfile)
            except OSError:
                pass

    def save_net(self, path="./net.pt", avoid_q=False):

        if self.rank == 0:
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                model = self.network.module
            else:
                model = self.network

            checkpoint = {
                "model_state_dict": model.to("cpu").state_dict(),
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

        self.jit_script_model = torch.jit.script(self.network)
        print("Model converted to jit_script, saved in self.jit_script_model")


    def save_jit_script(self, path="./net_jit_script.pt"):

        if self.rank == 0:
            self.jit_script_model = self.jit_script_model()
            torch.jit.save(self.jit_script_model, path)

            print("Model saved!")


    def save_onnx(self, example_data=torch.tensor([[]],requires_grad=True).float(), path="./net_onnx.onnx"):

        if self.rank == 0:
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                model = self.network.module
            else:
                model = self.network

            model = model.to("cpu", dtype=torch.float32)
            example_data = example_data.to(device="cpu", dtype=torch.float32)

            torch.onnx.export(model,                                            # model being run
                                example_data,                                   # model input (or a tuple for multiple inputs)
                                path,                                           # where to save the model (can be a file or file-like object)
                                export_params=True,                             # store the trained parameter weights inside the model file
                                opset_version=14,                               # the ONNX version to export the model to: https://onnxruntime.ai/docs/reference/compatibility.html
                                do_constant_folding=True,                       # whether to execute constant folding for optimization
                                input_names=['input'],                          # the model's input names
                                output_names=['output'],                        # the model's output names
                                dynamic_axes={'input': {0: 'batch_size'},       # variable length axes
                                            'output': {0: 'batch_size'}})


    def check_onnx(self,path="./net_onnx.pt"):
        if self.rank == 0:
            try:
                onnx_model = onnx.load(path)
                onnx.checker.check_model(onnx_model)
                print("ONNX checker: Success!")
            except:
                print("Failure in ONNX checker!")