# abstact class representing a sequence of augmentation transformations


from dataAugmentations import *
from randAugment import *

AUTOTUNE = tf.data.experimental.AUTOTUNE
from flagSettings import *


class AugmentationEngine:
    def transform(self, data):  # function to be implemented with a sequence of trasformations
        pass


class SimClrAugmentation(AugmentationEngine):
    def transform(self, data, augment=augmentation_type):
        if augment == 'simclr':
            print('using simclr augmentations')
            data = data.map(lambda x, y: (augmentBatch(x, y)),
                            num_parallel_calls=AUTOTUNE)
        if augment == 'rand':
            print('using randaug augmentations')
            aug = RandAugment(rand_augs, rand_strength)
            data = data.map(lambda x, y: (aug.__call__(x, y)),
                            num_parallel_calls=AUTOTUNE)
        return data


class TestAugmentation(AugmentationEngine):
    def transform(self, data):
        data = data.map(lambda x, y: (x, x, x, y),
                        num_parallel_calls=AUTOTUNE)
        print('using test augmentations')
        return data


class AugmentationStudy(AugmentationEngine):
    def __init__(self, augmentation1, augmentation2):
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def augment(self, input_data, augmentation_type):
        if augmentation_type == "crop":
            return cropResize(input_data)
        elif augmentation_type == "cutout":
            return cut_out(input_data)
        elif augmentation_type == "color":
            raise colorJitter(input_data, s=1)
        elif augmentation_type == "sobel":
            raise NotImplemented("Implement sobel")
        elif augmentation_type == "gaussian_noise":
            raise NotImplemented("Gaussian noise not implemented")
        elif augmentation_type == "gaussian_blur":
            std = random.uniform(.1, 2)
            return gaussianBlur(input_data, std)
        elif augmentation_type == "rotate":
            return rotate(input_data)
        else:
            raise Exception("Invalid argument for augment test")

    def transform(self, data):
        data = data.map(lambda x, y: (x, self.augment(x, self.augmentation1), self.augment(x, self.augmentation1), y),
                        num_parallel_calls=AUTOTUNE)
        data = data.map(lambda x, x_a1, x_a2, y: (
        x, self.augment(x_a1, self.augmentation2), self.augment(x_a2, self.augmentation2), y),
                        num_parallel_calls=AUTOTUNE)
        return data
