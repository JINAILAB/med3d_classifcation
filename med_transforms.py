from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

train_transforms = Compose([ScaleIntensity(), 
                            EnsureChannelFirst(), 
                            Resize((96, 96, 96)),
                            ])


val_transforms = Compose([ScaleIntensity(), 
                          EnsureChannelFirst(), 
                          Resize((96, 96, 96)),
                          ])



# for semgentation transform
# train_transform = Compose(
#     [
#         # load 4 Nifti images and stack them together
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#         RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#     ]
# )
# val_transform = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#     ]
# )