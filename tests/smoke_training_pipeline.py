"""Small isolated MEAN -> SIGMA -> FULL round-trip including ONNX exports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import onnxruntime as ort
import onnx
import uproot

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='tpc-training-smoke-') as folder:
    folder=Path(folder)
    config=json.loads((root/'run/configs/configuration.json').read_text())
    config['settings']['framework']=str(root)
    config['output']={'general':{'training':str(folder)},
        'createTrainingDataset':{'training_data':str(folder/'data.root')},
        'fitBBGraph':{'BBparameters':[]},'trainNeuralNet':{'QApath':str(folder/'qa')}}
    config['trainNeuralNetOptions']['configuration']=str(root/'run/configs/nnconfig.py')
    config['trainNeuralNetOptions']['numberOfEpochs']=2
    config['trainNeuralNetOptions']['scheduler']='slurm'
    config['trainNeuralNetOptions'].update(save_as_pt='True',save_as_onnx='True',save_loss_in_files='True')
    labels=config['createTrainingDatasetOptions']['labels_x']+config['createTrainingDatasetOptions']['labels_y']
    rng=np.random.default_rng(42)
    with uproot.recreate(folder/'data.root') as f:
        # Deliberately reverse branch insertion order to exercise named selection.
        f['data']={key:rng.uniform(.1,1,2001).astype(np.float32) for key in reversed(labels)}
    config_path=folder/'config.json'
    config_path.write_text(json.dumps(config))
    command=[sys.executable,str(root/'framework/training_neural_networks/train_single_sigma.py'),'-c',str(config_path)]
    for mode in ('mean','sigma','full'):
        output=folder/'networks'/('network_'+mode)
        output.mkdir(parents=True)
        subprocess.run(command+['--train-mode',mode.upper()],check=True)
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        model_path=str(output/('net_onnx_'+mode+'.onnx'))
        assert onnx.load(model_path).opset_import[0].version == 14
        session=ort.InferenceSession(model_path,sess_options=options,providers=['CPUExecutionProvider'])
        prediction=session.run(None,{'input':np.ones((3,len(labels)-2),np.float32)})[0]
        assert prediction.shape==(3,2 if mode=='full' else 1),prediction.shape
        assert np.isfinite(prediction).all()
    subprocess.run([sys.executable,str(root/'framework/training_neural_networks/shell_script_creation.py'),'-c',str(config_path),'--job-script','/tmp/train.py'],check=True)
    script=(folder/'TRAIN.sh').read_text()
    expected_cpus=config['trainNeuralNetOptions']['slurm']['cpus-per-task']
    assert f'#SBATCH --cpus-per-task={expected_cpus}' in script
    assert 'export MASTER_ADDR=' in script
    subprocess.run(['bash','-n',str(folder/'TRAIN.sh')],check=True)
print('MEAN/SIGMA/FULL checkpoint and ONNX round-trips, generated Slurm script: PASS')
