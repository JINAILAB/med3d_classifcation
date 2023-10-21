import sys
import yaml
import torch
import numpy as np
import os
import monai
import argparse
from med_model import densenet_model
from med_logger import setting_logger
from med_utils import set_seed, train_one_epoch, evaluate_one_epoch, batch_inference
from med_dataset import load_data, load_test_data
from med_transforms import train_transforms, val_transforms
from med_model import densenet_model
from types import SimpleNamespace
from torch import nn
from datetime import datetime
import pandas as pd
from collections import Counter


# argparse 설정
def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument('--infer', help='inference by the pretrained model', action='store_true')
    parser.add_argument('--ensemble', help='inference by the pretrained model', action='store_true')
    parser.add_argument('--train_only_one_file', help='inference by the pretrained model', action='store_true')
    
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
#     'seed' : 66
# }

def train_and_val(cfg, logger, main_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        "acc": {"value": 0, "filename": "acc_best_model.pth", "confusion_matrix" : 0},
        "f1": {"value": 0, "filename": "f1_best_model.pth", "confusion_matrix" : 0},
        "auroc": {"value": 0, "filename": "auroc_best_model.pth", "confusion_matrix" : 0}
    }

    for epoch in range(cfg.epochs):
        train_one_epoch(model, criterion, epoch, optimizer, train_loader, device, logger)
        lr_scheduler.step()
        eval_acc, eval_f1, c_matrix, eval_auroc = evaluate_one_epoch(model, criterion, epoch, valid_loader, device, logger)
        
        current_metrics = {"acc": eval_acc, "f1": eval_f1, "auroc": eval_auroc}
        
        for key in metrics:
            if metrics[key]["value"] < current_metrics[key]:
                metrics[key]["value"] = current_metrics[key]
                metrics[key]['confusion_matrix'] = c_matrix
                save_path = os.path.join(main_folder, 'model', metrics[key]["filename"])
                torch.save(model.state_dict(), save_path)

            
    for key in metrics:
        logger.debug(f'best {key} is {metrics[key]["value"]}')
        logger.debug(f'best {key} confusion_matrix is \n {metrics[key]["confusion_matrix"]}')
        
# 추론 코드 
def inference(cfg, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = densenet_model.to(device)
    model.load_state_dict(torch.load(cfg.infererence_pretrain_dir))
    _, test_loader = load_test_data(cfg.img_dirs, val_transforms, cfg.batch_size)
    all_predictions = batch_inference(model, test_loader, device, logger, cfg.model_path)

    # 예측된 확률에서 가장 높은 값의 인덱스를 가져와서 그에 해당하는 클래스 이름을 얻습니다.
    # 예시로, 클래스 이름을 'Type1', 'Type2', ... 라고 가정합니다.
    # 클래스 이름이 다르다면 이 부분을 수정해주세요.
    # 확률 추가
    predicted_classes = [f"Type{pred.argmax() + 1}" for pred in all_predictions]
    predicted_probabilities = [float(pred.max()) for pred in all_predictions]

    # ChallengeID 리스트 생성
    challenge_ids = [f"HT_Subject_{i:03}" for i in range(121, 201)]

    # DataFrame 생성
    df = pd.DataFrame({
        'ChallengeID': challenge_ids,
        'Submit_HTType': predicted_classes,
        'Probability': predicted_probabilities
    })

    # CSV로 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M_model")
    os.makedirs(os.path.join(cfg.work_dir, 'infer_csv'), exist_ok=True)
    df.to_csv(os.path.join(cfg.work_dir, 'infer_csv', 'predictions_'+cfg.seed+'_'+current_time+'.csv'), index=False)
    


    
def ensemble_csv(voting='hard'):
    ## hard voting 구현
    if voting == 'hard':
        # CSV 파일 목록 가져오기
        csv_dir = os.path.join(cfg.work_dir, 'infer_csv')
        csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

        # 각 CSV 파일에서 예측 결과를 가져오기
        all_preds = []
        for file in csv_files:
            df = pd.read_csv(file)
            all_preds.append(df['Submit_HTType'].tolist())

        # 하드 보팅으로 앙상블
        ensemble_preds = []
        for i in range(len(all_preds[0])):  # 각 이미지에 대해
            preds_for_image = [preds[i] for preds in all_preds]
            most_common, _ = Counter(preds_for_image).most_common(1)[0]  # 가장 많이 예측된 클래스 찾기
            ensemble_preds.append(most_common)

        
    elif voting == 'soft':
        all_probs = []
        num_classes = None

        # 각 CSV 파일에서 확률 값을 가져오기
        for file in csv_files:
            df = pd.read_csv(file)
            probs = df['Probability'].tolist()
            all_probs.append(probs)
            if num_classes is None:
                num_classes = len(df['Submit_HTType'].unique())

        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = [prob.argmax() for prob in ensemble_probs]

    else:
        raise ValueError("Unknown voting type. Choose either 'hard' or 'soft'.")
    
    # 앙상블 결과를 CSV로 저장
    ensemble_df = pd.DataFrame({
        'ChallengeID': df['ChallengeID'],
        'Submit_HTType': ensemble_preds,
    })

    ensemble_df.to_csv(os.path.join(csv_dir, 'ensemble_predictions.csv'), index=False)

if __name__ == '__main__':
    # args, yaml 파일 가져오기
    args = get_args_parser().parse_args()
    with open("cfg.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)
        cfg = SimpleNamespace(**cfg)
    # seed 설정
    set_seed(cfg.seed)
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
        inference(cfg, logger)
        
    elif args.ensemble:
        pass
    
    elif args.train_only_one_file:
        train_and_val(cfg, logger, main_folder)
    
    
    
    
    
