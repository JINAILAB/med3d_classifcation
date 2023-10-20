import sys
import yaml
import torch
import numpy as np
import os
import monai
import argparse
from med_model import densenet_model
from med_logger import setting_logger
from med_utils import set_seed, train_one_epoch, evaluate_one_epoch
from med_dataset import load_data
from med_transforms import train_transforms, val_transforms
from med_model import densenet_model
from types import SimpleNamespace
from torch import nn
from datetime import datetime

# seed 설정, device 설정
set_seed()

# argparse 설정
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument('--infer', help='inference by the pretrained model', action='store_true')
    
    return parser

# cfg 참고
# cfg = {
#     'max_epochs' : 5,
#     'model' : 'densenet_model',
#     'opt' : 'adamw',
#     'lr' : 0.001,
#     'weight decay' : 1e-4,
#     'label_smoothing' : 0,
#     'lr_scheduler' : 'cosineannelinglr',
#     'lr_warmup_epochs' : 3,
#     'lr_min' : 0.0,
#     'val_resize_size' : 108,
#     'val_crop_size' : 96,
#     'train_crop_size' : 96,
#     'weight_decay' : 1e-4,
#     'img_dirs' : '/home/',
#     'work_dir' : '/home/',
#     'lr_warmup_decay' : 0.01,
#     'infererence_pretrain_dir' : '/home'
# }

def train_and_val(cfg, logger, main_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_savedir = os.path.join(main_folder, 'model')
    
    logger.debug('model is', cfg.model)
    model = densenet_model.to(device)
    
    # loss 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    # Optimizer 설정
    if cfg.opt == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9
        )
        
    elif cfg.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
    # learning rate 전략 설정
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs, eta_min=cfg.lr_min)
    # warmup_lr 설정
    if cfg.lr_warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.lr_warmup_decay, total_iters=cfg.lr_warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[cfg.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
    
    _, train_loader, _, valid_loader = load_data(cfg.img_dirs, train_transforms, val_transforms,cfg.batch_size, logger, test_size=0.3)
    
    
    print("Start training")
    
    # metrics 설정
    metrics = {
        "acc": {"value": 0, "filename": "acc_best_model.pth"},
        "f1": {"value": 0, "filename": "f1_best_model.pth"},
        "auroc": {"value": 0, "filename": "auroc_best_model.pth"}
    }

    for epoch in range(cfg.epochs):
        train_one_epoch(model, criterion, epoch, optimizer, train_loader, device, logger)
        lr_scheduler.step()
        eval_acc, eval_f1, c_matrix, eval_auroc = evaluate_one_epoch(model, criterion, epoch, valid_loader, device, logger)
        
        current_metrics = {"acc": eval_acc, "f1": eval_f1, "auroc": eval_auroc}
        
        for key in metrics:
            if metrics[key]["value"] < current_metrics[key]:
                metrics[key]["value"] = current_metrics[key]
                save_path = os.path.join(main_folder, 'model', metrics[key]["filename"])
                torch.save(model.state_dict(), save_path)

            
    for key in metrics:
        logger.debug(f'best {key} is {metrics[key]["value"]}')
        
# 추론 코드 추후 작성
def inference(cfg):
    pass
        

if __name__ == '__main__':
    # args, yaml 파일 가져오기
    args = get_args_parser().parse_args()
    with open("cfg.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)
        cfg = SimpleNamespace(**cfg)

    # 모델과 로그 저장 폴더 생성
    # 현재의 날짜와 시간을 "YYYYMMDD_HHMM_model" 형식으로 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M_model")
    # main폴더 생성
    main_folder = os.path.join(cfg.workdir, current_time)
    os.makedirs(main_folder, exist_ok=True)
    # 'model'과 'log' 폴더 생성
    os.makedirs(os.path.join(main_folder, 'model'), exist_ok=True)
    os.makedirs(os.path.join(main_folder, 'log'), exist_ok=True)

    print(f"'{main_folder}' 아래에 'model'과 'log' 폴더에 데이터가 저장됩니다.")
    # logger 생성 장소 지정
    logger = setting_logger(os.path.join(main_folder, 'log', 'logfile.txt'))
    if args.inference:
        inference()
    else:
        train_and_val(cfg, logger, main_folder)
    
    
    
    
    
