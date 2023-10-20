import yaml

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
    'weight_decay' : 1e-4,
    'img_dirs' : '/home/ncp/blockstorage'
}

with open("cfg.yaml", 'w') as outfile:
    yaml.dump(cfg, outfile)