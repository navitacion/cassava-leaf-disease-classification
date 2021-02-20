import albumentations as albu
from albumentations.pytorch import ToTensorV2
from abc import ABCMeta



def get_transforms(transform_name, img_size):
    _dict = {
        't_1': ImageTransform_1(img_size=img_size),
        't_2': ImageTransform_2(img_size=img_size),
        't_3': ImageTransform_3(img_size=img_size),
        't_4': ImageTransform_4(img_size=img_size),
        't_5': ImageTransform_5(img_size=img_size),
        't_6': ImageTransform_6(img_size=img_size),
        't_7': ImageTransform_7(img_size=img_size),
    }

    return _dict[transform_name]


class BaseTransform(metaclass=ABCMeta):
    def __init__(self):
        self.transform = None

    def __call__(self, img, phase='train'):
        augmented = self.transform[phase](image=img)
        augmented = augmented['image']

        return augmented



class ImageTransform_1(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_1, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_2(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_2, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_3(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_3, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_4(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_4, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.OneOf([
                    albu.GridDistortion(p=1.0),
                    albu.OpticalDistortion(p=1.0),
                ], p=0.5),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_5(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_5, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.OneOf([
                    albu.GridDistortion(p=1.0),
                    albu.OpticalDistortion(p=1.0),
                ], p=0.5),
                albu.Normalize(mean, std),
                albu.CoarseDropout(max_height=15, max_width=15, max_holes=8, p=0.5),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_6(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_6, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.Normalize(mean, std),
                albu.CoarseDropout(max_height=15, max_width=15, max_holes=8, p=0.5),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }


class ImageTransform_7(BaseTransform):
    def __init__(self, img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageTransform_7, self).__init__()
        self.transform = {
            'train': albu.Compose([
                albu.RandomResizedCrop(img_size, img_size),
                albu.ColorJitter(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Transpose(p=0.5),
                albu.CLAHE(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.Normalize(mean, std),
                albu.CoarseDropout(max_height=15, max_width=15, max_holes=8, p=0.5),
                ToTensorV2(),
            ], p=1.0),

            'val': albu.Compose([
                albu.Resize(img_size, img_size),
                albu.Normalize(mean, std),
                ToTensorV2(),
            ], p=1.0)
        }
