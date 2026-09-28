"""Explicit benchmark; not part of unit-test discovery.

Examples (run in a compute allocation):
  python -m tests.benchmark_training_memory --rows 200000 --legacy
  python -m tests.benchmark_training_memory --rows 200000
  srun ... python tests/benchmark_training_memory.py --rows 20000000 --train
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from framework.neural_network_class.NeuralNetworkClasses.dataset_loading import DataLoading, dataset, batch_loader
from framework.neural_network_class.NeuralNetworkClasses.NN_class import NN

parser=argparse.ArgumentParser()
parser.add_argument('--rows',type=int,default=200000)
parser.add_argument('--legacy',action='store_true')
parser.add_argument('--train',action='store_true')
args=parser.parse_args()
torch.set_num_threads(1)
baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start=time.perf_counter()
X=np.full((args.rows,7),0.1,dtype=np.float32)
y=np.full((args.rows,1),0.7,dtype=np.float32)
if args.train:
    n=args.rows*9//10
    data=DataLoading([X[:n],y[:n]],[X[n:],y[n:]],batch_sizes=[262144],transform_data=False,verbose=False)
    model=nn.Sequential(nn.Linear(7,12),nn.ReLU(),
        *[nn.Sequential(nn.Linear(12,12),nn.ReLU()) for _ in range(9)],nn.Linear(12,1))
    network=NN(model)
    network.training(data,epochs=2,validation_batch_size=65536,verbose=True)
    result={'rank':network.rank,'rows':args.rows,'training_loss':network.training_loss,
            'validation_loss':network.validation_loss,
            'parameter_sum':sum(p.detach().double().sum().item() for p in model.parameters()),
            'gpu_peak_MiB':torch.cuda.max_memory_allocated()/2**20}
else:
    if args.legacy:
        # Original dataset implementation and default per-row collation.
        values=list(zip(torch.tensor(X),torch.tensor(y)))
        loader=DataLoader(values,batch_size=512)
    else:
        values=dataset(torch.from_numpy(X),torch.from_numpy(y))
        loader,_=batch_loader(values,512)
    construction=time.perf_counter()-start
    t=time.perf_counter()
    count=sum(len(a) for a,b in loader)
    result={'rows':count,'legacy':args.legacy,'construction_s':construction,'iteration_s':time.perf_counter()-t}
result.update(elapsed_s=time.perf_counter()-start,
              peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              RSS_increase_MiB=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-baseline)/1024)
print('BENCHMARK '+json.dumps(result),flush=True)
