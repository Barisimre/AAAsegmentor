from src.constants import *
import monai


class LoadMHD(monai.transforms.Transform):

    def __init__(self, keys=None):
        self.reader = monai.data.ITKReader()

    def __call__(self, item):
        # Paths
        img = item['img']
        mask = item['mask']

        img, img_meta = self.reader.get_data(self.reader.read(img))
        mask, mask_meta = self.reader.get_data(self.reader.read(mask))

        return {
            'img': monai.data.MetaTensor(img, affine=img_meta['affine']),
            'mask': monai.data.MetaTensor(mask, affine=mask_meta['affine']),
            'name': item['name']
        }


class GetFilesTrain(monai.transforms.Transform):

    def __init__(self, keys=None):
        pass

    def __call__(self, item):
        return {'img': f"{DATA_PATH}/images/{item}", 'mask': f"{DATA_PATH}/masks/{item}", 'name': item}


class GetFilesTest(monai.transforms.Transform):

    def __init__(self, keys=None):
        pass

    def __call__(self, item):
        return {'img': f"{DATA_PATH}/images/{item}", 'mask': f"{DATA_PATH}/masks/{item}", 'name': item}