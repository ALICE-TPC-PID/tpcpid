"""Run with two Slurm GPU tasks; tests uneven validation and checkpoint exports."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from framework.neural_network_class.NeuralNetworkClasses.dataset_loading import DataLoading
from framework.neural_network_class.NeuralNetworkClasses.NN_class import NN

X=np.arange(34,dtype=np.float32).reshape(17,2)/20
y=np.arange(17,dtype=np.float32).reshape(-1,1)/10
model=torch.nn.Linear(2,1)
with torch.no_grad():
    model.weight.fill_(.25)
    model.bias.fill_(.1)
expected=torch.nn.functional.mse_loss(model(torch.from_numpy(X[10:])),torch.from_numpy(y[10:])).item()
data=DataLoading([X[:10],y[:10]],[X[10:],y[10:]],batch_sizes=[3],transform_data=False,verbose=False)
net=NN(model)
net.training(data,epochs=2,learning_rate=0,validation_batch_size=3,verbose=False)
np.testing.assert_allclose(net.validation_loss,[expected]*2,rtol=1e-6)
assert not model.training
if net.rank==0:
    with tempfile.TemporaryDirectory() as d:
        before=next(model.parameters()).device
        net.save_net(str(Path(d)/'model.pt'),avoid_q=True)
        assert next(model.parameters()).device==before
        net.save_onnx(torch.zeros(1,2),str(Path(d)/'model.onnx'))
        assert next(model.parameters()).device==before
print(f'Rank {net.rank}: uneven distributed validation matches full reference; PASS',flush=True)
