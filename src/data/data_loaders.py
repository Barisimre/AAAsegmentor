from src.constants import *
from src.data.transforms import train_transform, test_transform
import os


def get_loaders():

    filenames = os.listdir(DATA_PATH + "/images")
    filenames = [f for f in filenames if 'mhd' in f]
    train_dataset = monai.data.CacheDataset(filenames, transform=train_transform, num_workers=16)
    train_loader = monai.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    filenames = os.listdir(DATA_PATH + "/test_images")
    filenames = [f for f in filenames if 'mhd' in f]
    test_dataset = monai.data.CacheDataset(filenames, transform=test_transform, num_workers=16)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader
