from src.constants import *
from src.data.custom_transforms import GetFilesTrain, LoadMHD, GetFilesTest
import numpy as np

# For some reason this breaks caching :(

# joint_transforms = monai.transforms.Compose(
#     [
#         LoadMHD(),
#         monai.transforms.AddChanneld(keys=['img', 'mask']),
#         monai.transforms.Spacingd(keys=["img", "mask"], pixdim=SPACINGS, mode=("bilinear", "nearest")),
#         monai.transforms.SqueezeDimd(keys=['img', 'mask'], dim=0),
#         monai.transforms.AsChannelFirstd(keys=['img', 'mask']),
#         monai.transforms.AddChanneld(keys=['img', 'mask']),
#         monai.transforms.ScaleIntensityRanged(keys=["img"], a_min=CT_WINDOW_MIN, a_max=CT_WINDOW_MAX, b_min=0, b_max=1, clip=True),
#     ]
# )


train_transform = monai.transforms.Compose(
    [
        GetFilesTrain(),
        LoadMHD(),
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.Spacingd(keys=["img", "mask"], pixdim=SPACINGS, mode=("bilinear", "nearest")),
        monai.transforms.SqueezeDimd(keys=['img', 'mask'], dim=0),
        monai.transforms.AsChannelFirstd(keys=['img', 'mask']),
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.ScaleIntensityRanged(keys=["img"], a_min=CT_WINDOW_MIN, a_max=CT_WINDOW_MAX, b_min=0, b_max=1, clip=True),
        monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
        # monai.transforms.EnsureTyped(keys=["img", "mask"], track_meta=False),

        monai.transforms.RandCropByPosNegLabeld(keys=["img", "mask"],
                                                spatial_size=CROP_SIZE,
                                                pos=1,
                                                neg=1,
                                                num_samples=1,
                                                label_key="mask",
                                                ),
        monai.transforms.SpatialPadd(keys=['img', 'mask'], spatial_size=CROP_SIZE),
        monai.transforms.RandFlipd(keys=['img', 'mask'], prob=0.1, spatial_axis=0),
        monai.transforms.RandFlipd(keys=['img', 'mask'], prob=0.1, spatial_axis=0),

        monai.transforms.RandFlipd(keys=['img', 'mask'], prob=0.1, spatial_axis=0),

        monai.transforms.RandShiftIntensityd(keys=["img"], offsets=0.10, prob=0.50),
 
    ]
)


test_transform = monai.transforms.Compose(
    [
        GetFilesTest(),
        LoadMHD(),
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.Spacingd(keys=["img", "mask"], pixdim=SPACINGS, mode=("bilinear", "nearest")),
        monai.transforms.SqueezeDimd(keys=['img', 'mask'], dim=0),
        monai.transforms.AsChannelFirstd(keys=['img', 'mask']),
        monai.transforms.AddChanneld(keys=['img', 'mask']),
        monai.transforms.ScaleIntensityRanged(keys=["img"], a_min=CT_WINDOW_MIN, a_max=CT_WINDOW_MAX, b_min=0, b_max=1, clip=True),
        monai.transforms.CropForegroundd(keys=["img", "mask"], source_key="img"),
        # monai.transforms.EnsureTyped(keys=["img", "mask"], track_meta=True),


    ]
)



