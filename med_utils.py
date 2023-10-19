import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random

def set_seed(seed_value=66):
    """
    주어진 시드 값으로 모든 라이브러리의 랜덤 시드를 설정합니다.
    
    Args:
        seed_value (int): 설정할 랜덤 시드 값.
    """
    torch.manual_seed(seed_value)
    
    # GPU를 사용할 경우 추가적인 시드 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        cudnn.benchmark = False
        cudnn.deterministic = True

    np.random.seed(seed_value)
    random.seed(seed_value)
    
    

def train_one_epoch(model, criterion, epoch, optimizer, data_loader, device, logger, acc_metric, f1_metric):
    logger.info(f'train epoch : {epoch}')
    model.train()
    
    train_loss = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        
        total += targets.size(0) 
        # preds = preds.cpu()
        # targets = targets.cpu()
        
        acc = acc_metric(preds, targets)
        f1 = f1_metric(preds, targets)

    acc = acc_metric.compute()
    f1 = f1_metric.compute()

    logger.info(f'Epoch {epoch:<4} ,train_Loss = {train_loss / total :<10}, train_acc = {acc:<10}, train_f1 = {f1:<10}')
    
    acc_metric.reset()
    f1_metric.reset()
    
    

    
    
    
    
    