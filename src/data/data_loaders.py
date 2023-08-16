from src.constants import *
from src.data.transforms import train_transform, test_transform
import os


def get_loaders():

    filenames = os.listdir(DATA_PATH + "/images")
    test_names = [f"EL_{n}pre.mhd" for n in TEST_PATIENTS]
    train_filenames = [f for f in filenames if 'mhd' in f and f not in test_names][:DATA_LIMIT]
    test_filenames = [f for f in filenames if 'mhd' in f and f in test_names][:DATA_LIMIT]

    # train_filenames = [f for f in filenames if 'mhd' in f and ]
    # test_filenames = []

    train_dataset = monai.data.CacheDataset(train_filenames, transform=train_transform, num_workers=16)
    train_loader = monai.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    test_dataset = monai.data.CacheDataset(test_filenames, transform=test_transform, num_workers=16)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    return train_loader, test_loader
