from src.constants import *

SWINUNETR = monai.networks.nets.SwinUNETR(img_size=CROP_SIZE, in_channels=1, out_channels=3, depths=(2, 2, 2, 2),
                                          num_heads=(3, 6, 12, 24), feature_size=24, norm_name='instance',
                                          drop_rate=0.1, attn_drop_rate=0.0, dropout_path_rate=0.0, normalize=True,
                                          use_checkpoint=True, spatial_dims=3)

UNet = monai.networks.nets.UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=[8, 16, 32, 64, 128],
                                strides=[2, 2, 2, 2], kernel_size=3, up_kernel_size=3, num_res_units=2, act='PRELU',
                                norm='INSTANCE', dropout=0.1, bias=True, adn_ordering='NDA')
