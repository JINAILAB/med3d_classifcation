import os
import pandas as pd
from monai.data import DataLoader, ImageDataset
from glob import glob
from sklearn.model_selection import train_test_split
import torch

# load train_dataset, train_loader, valid_dataset, valid_loader
import os
import pandas as pd
from monai.data import DataLoader, ImageDataset
import glob
from sklearn.model_selection import train_test_split
import torch

# load train_dataset, train_loader, valid_dataset, valid_loader
def load_data(img_dirs, train_transforms, valid_transforms, batch_size, logger, test_size=0.3):
    df = pd.read_csv('/workspace/blockstorage/jyp/StudyHT_Open.csv')
    # gre img_dirs sort
    gre_img_dirs = glob.glob('/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/gre_in_MNI_brain.nii.gz')
    mni_img_dirs = glob.glob('/workspace/nasr/pub66n1/topic1/ImageData/*/MNI_Space/t2star_in_MNI_brain.nii.gz')
    img_dirs = sorted(gre_img_dirs + mni_img_dirs)
    labels = list(map(int, list(df.loc[:119, 'GT_HTType'])))
    logger.debug(f'image number is  {len(labels)}')
    train_img_dirs = img_dirs[:120]
    train_img_dirs, valid_img_dirs, train_labels, valid_labels = train_test_split(train_img_dirs,
                                                                                  labels,
                                                                                  test_size=test_size,
                                                                                  stratify=labels)
    logger.debug(f'train_img_number is {len(train_img_dirs)} valid_img_number  is {len(valid_img_dirs)}')
    train_dataset = ImageDataset(image_files=train_img_dirs, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    
    valid_dataset = ImageDataset(image_files=valid_img_dirs, labels=valid_labels, transform=valid_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    
    return train_dataset, train_loader, valid_dataset, valid_loader
    
    
def load_test_data(img_dirs, test_transforms, batch_size):
    img_dirs = sorted(glob.glob('/workspace/blockstorage/*/'))
    test_img_dirs = img_dirs[120:]
    test_dataset = ImageDataset(image_files=test_img_dirs, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    
    return test_dataset, test_dataloader

    


