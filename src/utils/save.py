import nibabel as nib
import numpy as np


def save_numpy_to_nifti(arr, path):
    img = nib.Nifti1Image(arr, np.eye(4))
    nib.save(img, path)
