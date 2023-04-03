from src.constants import *
import monai


class LoadMHD(monai.transforms.Transform):

    def __init__(self, keys=None):
        pass

    def __call__(self, item):
        # Paths
        img = item['img']
        mask = item['mask']

        reader = monai.data.ITKReader()

        img, img_meta = reader.get_data(reader.read(img))
        mask, mask_meta = reader.get_data(reader.read(mask))

        return {
            'img': monai.data.MetaTensor(img, affine=img_meta['affine']),
            'mask': monai.data.MetaTensor(mask, affine=mask_meta['affine'])
        }


class GetFilesTrain(monai.transforms.Transform):

    def __init__(self, keys=None):
        pass

    def __call__(self, item):
        return {'img': f"{DATA_PATH}/images/{item}", 'mask': f"{DATA_PATH}/masks/{item}"}


class GetFilesTest(monai.transforms.Transform):

    def __init__(self, keys=None):
        pass

    def __call__(self, item):
        return {'img': f"{DATA_PATH}/test_images/{item}", 'mask': f"{DATA_PATH}/test_masks/{item}"}

