import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
import time
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryAUROC
)



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
    
    

def train_one_epoch(model, criterion, epoch, optimizer, data_loader, device, logger):
    logger.info(f'train epoch : {epoch}')
    model.train()
    
    train_loss = 0
    total = 0
    
    acc_metric = BinaryAccuracy()
    f1_metric = BinaryF1Score()
    
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
        auroc = auroc(preds, targets)

    acc = acc_metric.compute()
    f1 = f1_metric.compute()
    auroc = auroc.compute()

    logger.debug(f'Epoch {epoch:<3} ,train_Loss = {train_loss / total :<8}, train_acc = {acc:<8}, train_f1 = {f1:<8}, train_auroc = {auroc:<8}')
    
    acc_metric.reset()
    f1_metric.reset()
    
def evaluate_one_epoch(model, criterion, epoch, valid_loader, device, logger):
    print('\n[ Test epoch: %d ]' % epoch)
    model.eval()
    valid_loss = 0
    total = 0
    
    acc_metric = BinaryAccuracy()
    f1_metric = BinaryF1Score()
    confmat = confmat= BinaryConfusionMatrix()
    auroc = BinaryAUROC(thresholds=None)
    
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            preds = model(inputs)
            
            valid_loss += criterion(preds, targets).item()
            
            
            acc = acc_metric(preds, targets)
            f1 = f1_metric(preds, targets)
            c_matrix = confmat(preds, targets)
            auroc= auroc(preds, targets)
            
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc.compute()
        logger.debug(f'Epoch {epoch:<3} ,valid_Loss = {valid_loss / total :<8}, valid_acc = {acc:<8}, valid_f1 = {f1:<8}, , valid_auroc = {auroc:<8}')
            
        acc_metric.reset()
        f1_metric.reset()
        confmat.reset()
        auroc.reset()
        
    return acc, f1, c_matrix, auroc
        
    
    
    
    
    