import os
import gc
import copy
import time
import random
import string

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import BertTokenizer, BertModel, BertConfig

# Suppress warnings
import warnings
# warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CONFIG = {
    'seed': 42,
    'max_length': 128, 
    'train_batch_size': 32,
    'valid_batch_size': 32,
    'num_classes': 6,
    'model_name': 'distilbert-base-uncased',
    'n_accumulate': 1,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'learning_rate': 1e-4,
    'weight_decay': 1e-6,
    'scheduler': None,
    'epochs': 2,
    'train_split': 0.8
}

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])