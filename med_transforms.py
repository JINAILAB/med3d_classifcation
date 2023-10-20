from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

train_transforms = Compose([ScaleIntensity(), 
                            EnsureChannelFirst(), 
                            Resize((96, 96, 96)),
                            ])


val_transforms = Compose([ScaleIntensity(), 
                          EnsureChannelFirst(), 
                          Resize((96, 96, 96)),
                          ])