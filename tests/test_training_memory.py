"""Run with python -m unittest discover -s tests -v from the repository root."""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import uproot
from framework.neural_network_class.NeuralNetworkClasses.dataset_loading import (
    dataset, DataLoading, TensorBatchSampler, batch_loader)
from framework.neural_network_class.NeuralNetworkClasses.NN_class import NN
from framework.neural_network_class.NeuralNetworkClasses.custom_loss_functions import weighted_mse_loss
from framework.neural_network_class.NeuralNetworkClasses.extract_from_root import load_tree, r_load_tree


class TrainingTests(unittest.TestCase):
    def test_root_rdata_loader_is_lazy(self):
        marker = object()
        calls = []

        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        fake_root = types.SimpleNamespace(
            Experimental=types.SimpleNamespace(
                ML=types.SimpleNamespace(RDataLoader=loader)
            )
        )
        with patch.dict(sys.modules, {"ROOT": fake_root}):
            result = r_load_tree("rdf", batch_size=128)

        self.assertIs(result, marker)
        self.assertEqual(calls, [(("rdf",), {"batch_size": 128})])

    def data(self, sizes=(13, 7), batches=(4,)):
        X = np.arange(sum(sizes)*2, dtype=np.float32).reshape(-1,2)/20
        y = X.sum(axis=1, keepdims=True)
        n=sizes[0]
        return DataLoading([X[:n],y[:n]], [X[n:],y[n:]], batch_sizes=list(batches),
                           transform_data=False, copy_to_device=False, verbose=False)

    def test_storage_and_cpu_flag(self):
        with patch('torch.cuda.is_available', return_value=True):
            data = self.data()
        self.assertEqual(data.datasetTS.X.device.type, 'cpu')
        self.assertFalse(hasattr(data.datasetTS, 'list'))
        self.assertEqual(data.sizeTS, 13*3*4)
        X = torch.zeros(10,2)
        self.assertEqual(dataset(X,X).X.data_ptr(), X.data_ptr())

    def test_shards_and_padding(self):
        for n in (1, 2, 5, 19):
            for world in (1,2,8):
                for shuffle in (False, True):
                    ranks=[]
                    lengths=[]
                    for rank in range(world):
                        sampler=TensorBatchSampler(n,3,shuffle,rank,world,False)
                        chunks=[torch.arange(n)[idx].tolist() for idx in sampler]
                        ranks.extend(sum(chunks, []))
                        padded=TensorBatchSampler(n,3,shuffle,rank,world,True)
                        lengths.append(sum(len(torch.arange(n)[idx]) for idx in padded))
                    self.assertEqual(sorted(ranks),list(range(n)))
                    self.assertEqual(len(set(lengths)),1)
        sampler=TensorBatchSampler(100,8,True)
        one=torch.cat(list(sampler))
        sampler.set_epoch(1)
        self.assertFalse(torch.equal(one,torch.cat(list(sampler))))

    def test_scalers_are_fitted_on_training_only(self):
        x=np.arange(12,dtype=np.float32).reshape(6,2)
        data=DataLoading([x,x[:,:1]],[x+100,x[:,:1]+100],
            X_data_scalers=[('s',StandardScaler())], y_data_scalers=[], verbose=False)
        np.testing.assert_allclose(data.scalingX.fitted.named_steps['s'].mean_,x.mean(0))
        np.testing.assert_allclose(data.inverse_X(data.datasetVS.X),x+100,rtol=1e-6)

    def test_bounded_validation_modes_and_metric(self):
        class Probe(nn.Linear):
            def __init__(self):
                super().__init__(2,1)
                self.calls=[]
            def forward(self,x):
                self.calls.append((len(x),self.training,torch.is_grad_enabled()))
                return super().forward(x)
        data=self.data(batches=(4,2))
        model=Probe()
        net=NN(model)
        net.training(data,multigpu=0,epochs=2,epochs_ls=[0,1],learning_rate=0,
                     validation_batch_size=3,verbose=False,shuffle_every_epoch=False)
        val=[x for x in model.calls if not x[1]]
        self.assertTrue(all(n<=3 and not grad for n,_,grad in val))
        with torch.no_grad():
            expected=nn.MSELoss()(model(data.datasetVS.X),data.datasetVS.y).item()
        self.assertAlmostEqual(net.validation_loss[-1],expected,places=5)
        self.assertFalse(model.training)
        self.assertTrue(all(isinstance(x,float) for x in net.training_loss))

    def test_batch_limit_and_default_loss(self):
        data=self.data()
        model=nn.Linear(2,1)
        net=NN(model)
        net.training(data,multigpu=0,nsamples=1,verbose=False,shuffle_every_epoch=False)
        self.assertEqual(net.optimizer.state[model.weight]['step'].item(),1)

    def test_seed_reproduces_initialization_shuffle_and_losses(self):
        def train_once(seed):
            torch.manual_seed(seed)
            data=self.data(batches=(4,))
            data.seed=seed
            net=NN(nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,1)))
            net.training(data,multigpu=0,epochs=3,verbose=False,
                         shuffle_every_epoch=True)
            state={key:value.detach().clone() for key,value in net.network.state_dict().items()}
            return state,net.training_loss,net.validation_loss

        state_a,train_a,val_a=train_once(42)
        state_b,train_b,val_b=train_once(42)
        self.assertEqual(train_a,train_b)
        self.assertEqual(val_a,val_b)
        self.assertTrue(all(torch.equal(state_a[key],state_b[key]) for key in state_a))

    def test_multioutput_weighted_loss(self):
        x=torch.tensor([[1.,2.],[3.,4.]])
        weights=torch.tensor([2.,3.])
        self.assertEqual(weighted_mse_loss(x,torch.zeros_like(x),1).item(),7.5)
        self.assertEqual(weighted_mse_loss(x,torch.zeros_like(x),weights).item(),61.25)
        data=self.data()
        net=NN(nn.Linear(1,1))
        net.training(data,multigpu=0,weights=True,loss_function=weighted_mse_loss,verbose=False)

    def test_bounded_onnx_prediction(self):
        from framework.neural_network_class.NeuralNetworkClasses.inference import predict_onnx
        class Session:
            def run(self, outputs, feed):
                x=feed['input']
                self.assertions(x)
                return [x.sum(axis=1,keepdims=True)]
            assertions=lambda _,x: None
        session=Session()
        session.assertions=lambda x: (self.assertLessEqual(len(x),3),self.assertEqual(x.dtype,np.float32))
        x=np.arange(22).reshape(11,2)
        np.testing.assert_array_equal(predict_onnx(session,x,3),x.sum(1,keepdims=True))

    def test_vectorized_workers(self):
        data=self.data()
        loader,_=batch_loader(data.datasetTS,4,num_workers=2)
        self.assertEqual(sum(len(x) for x,y in loader),13)

    def test_root_mixed_dtypes_and_latest_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'mixed.root')
            with uproot.recreate(path) as f:
                f.mktree('first', {'x': np.array([1], dtype=np.float32)})
                f.mktree('second', {'x': np.array([1.0000000001], dtype=np.float64)})
                f.mktree('first', {'x': np.array([2], dtype=np.float32)})
            _,values=load_tree().load(path,use_vars=['x'])
            self.assertEqual(values.dtype,np.float64)
            self.assertEqual(values.shape,(2,1))
            self.assertIn(2.,values[:,0])
            self.assertIn(1.0000000001,values[:,0])

    def test_root_ttree_rntuple_order_limit_wildcard(self):
        with tempfile.TemporaryDirectory() as d:
            for kind in ('tree','ntuple'):
                path=os.path.join(d,kind+'.root')
                with uproot.recreate(path) as f:
                    values={'b':np.arange(11,dtype=np.float32),'a':np.arange(11,dtype=np.float64)+20}
                    if kind=='tree': f.mktree('data',values)
                    else: f['data']=values
                labels,values=load_tree().load(path,use_vars=['a','b'],limit=5,dtype=np.float32,step_size=2)
                self.assertEqual(labels.tolist(),['a','b'])
                self.assertEqual(values.shape,(5,2))
                self.assertEqual(values.dtype,np.float32)
                np.testing.assert_array_equal(values[:,0],np.arange(5)+20)
            _,values=load_tree().load(os.path.join(d,'*.root'),use_vars=['b'],limit=3)
            self.assertEqual(values.shape,(6,1))
            with self.assertRaises(RuntimeError):
                load_tree().load(path,use_vars=['missing'])
            _,values=load_tree().load(path,limit=0)
            self.assertEqual(len(values),0)

if __name__=='__main__':
    unittest.main()
