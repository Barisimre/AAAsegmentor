import nibabel as nib
import itk
import itkwidgets
import numpy as np
from ipywidgets import HBox

image_file = 'image.nii.gz'
mask_file = 'mask.nii.gz'
prediction_file = 'prediction.nii.gz'

image = nib.load(image_file)
mask = nib.load(mask_file)
prediction = nib.load(prediction_file)

image_itk = itk.GetImageFromArray(image.get_fdata().astype(np.float32))
mask_itk = itk.GetImageFromArray(mask.get_fdata().astype(np.float32))
prediction_itk = itk.GetImageFromArray(prediction.get_fdata().astype(np.float32))

viewer_image = itkwidgets.view(image_itk, cmap=itkwidgets.cm.gray, ui_collapsed=True)
viewer_mask = itkwidgets.view(mask_itk, cmap=itkwidgets.cm.jet, ui_collapsed=True, opacity=0.5)
viewer_prediction = itkwidgets.view(prediction_itk, cmap=itkwidgets.cm.jet, ui_collapsed=True, opacity=0.5)

HBox([viewer_image, viewer_mask, viewer_prediction])
