import logging
import sys
import torch
import numpy as np
import monai
from monai.config import print_config
import argparse
from monai.data
import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from med_model import densenet_model
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score
)
from med_logger import setting_logger
from med_utils import set_seed

# seed 설정, device 설정
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logger 파일 추후 수정 필요
logger = setting_logger('./logfile/logfile.txt')

model = densenet_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)




cfg = {
    'max_epochs' : 5,
    'model' : 'densenet_model',
    'opt' : 'adamw',
    'lr' : 0.001,
    'weight decay' : 1e-4,
    'label_smoothing' : 0,
    'lr_scheduler' : 'cosineannelinglr',
    'lr_warmup_epochs' : 3,
    'val_resize_size' : 108,
    'val_crop_size' : 96,
    'train_crop_size' : 96,
}

