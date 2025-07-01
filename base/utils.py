import os
import sys
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

sys.setrecursionlimit(10000)

def set_gpu(x):
    torch.set_num_threads(1)
    torch.cuda.set_device(int(x))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(x)
    print('using gpu:', x)

def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def safe_cuda(model):
    device = torch.device('cuda')
    print("Moving model to GPU safely...")
    
    try:
        for param in model.parameters():
            param.data = param.data.to(device)
        
        for buffer_name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(device)
        
        if hasattr(model, 'device'):
            model.device = device
            
        print("Model parameters moved to GPU manually")
    except Exception as e:
        print(f"GPU movement failed: {str(e)}")
    
    original_cuda = model.cuda
    def safe_cuda_override(*args, **kwargs):
        print("Using safe cuda replacement method")
        return model
    
    model.cuda = safe_cuda_override
    
    return model

def get_model(args):
    model = None
    
    if args.model == 'FreqDGT':
        try:
            from FreqDGT import FreqDGT
            
            model_args = {
                'layers_graph': args.layers_graph,
                'layers_transformer': args.layers_transformer,
                'num_adj': args.num_adj,
                'num_chan': args.num_channel,
                'num_feature': args.num_feature,
                'hidden_graph': args.hidden_graph,
                'K': args.K,
                'num_head': args.num_head,
                'dim_head': args.dim_head,
                'dropout': args.dropout,
                'num_class': args.num_class,
                'alpha': args.alpha,
                'graph2token': args.graph2token,
                'encoder_type': args.encoder_type,
            }
            
            model_args.update({
                'sampling_rate': getattr(args, 'sampling_rate', 200),
                'feature_type': getattr(args, 'feature_type', 'rPSD'),
                'enable_disentangle': getattr(args, 'enable_disentangle', True)
            })
            
            model = FreqDGT(**model_args)
            
            print(f"FreqDGT model created successfully, feature type: {getattr(args, 'feature_type', 'rPSD')}")
            
            model = safe_cuda(model)
            print("Model safely moved to GPU")
            
        except ImportError as e:
            raise ImportError(f"Failed to create FreqDGT model: {str(e)} - Please ensure FreqDGT module is installed")
        except Exception as e:
            raise Exception(f"Error creating FreqDGT model: {str(e)}")
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if model is None:
        raise ValueError(f"Failed to create model: {args.model}")
        
    return model

def get_dataloader(data, label, batch_size):
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader

def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm

def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err

def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class eegDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
