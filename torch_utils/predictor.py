import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
np.random.seed(0)

class TorchPredictor():
    def __init__(self, name, model, preprocessor=None, postprocessor=None, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.name = name
        self.checkpoint_path = pathlib.Path(kwargs.get('checkpoint_folder', f'./models/{name}_chkpts'))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
    def _get_checkpoints(self, name=None):
        checkpoints = []
        checkpoint_path = self.checkpoint_path if name is None else pathlib.Path(kwargs.get('checkpoint_folder', f'./models/{name}_chkpts'))
        for cp in self.checkpoint_path.glob('checkpoint_*'):
            checkpoint_name = str(cp).split('/')[-1]
            checkpoint_epoch = int(checkpoint_name.split('_')[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        return checkpoints

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        if name is None:
            checkpoints =  self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[0]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint['epoch']
        return None

    # pass single batch input, without batch axis
    def predict_one(self, x):
        self.model.eval()
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        if type(x) is not torch.Tensor:
            x = torch.tensor(x,  dtype=torch.float32)
        with torch.no_grad():
            if type(x) is list:
                x = [xi.to(self.device).unsqueeze(0) for xi in x]
            else:
                x = x.to(self.device).unsqueeze(0)
            y_pred = self.model(x)
            if self.device == 'cuda':
                y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            if self.postprocessor is not None:
                y_pred = self.postprocessor(y_pred)
            return y_pred
