# Abstract class representing a sequence of augmentation transformations


from dataAugmentations import *
from flagSettings import *
from randAugment import *

AUTOTUNE = tf.data.experimental.AUTOTUNE


class AugmentationEngine:
    def transform(self, data):  # Function to override when using the engine
        pass


class SimClrAugmentation(AugmentationEngine):
    """
    Engine for the SimCLR framework.
    augment: Describes if using the standard augmentaion strategy or the new randAugment
    """

    def transform(self, data, augment=augmentation_type):
        if augment == 'simclr':
            print('Using SimCLR augmentations')
            data = data.map(lambda x, y: (augment_batch(x, y)),
                            num_parallel_calls=AUTOTUNE)
        if augment == 'rand':
            print('using RandAugment augmentations')
            aug = RandAugment(rand_augs, rand_strength)
            data = data.map(lambda x, y: (aug.__call__(x, y)), num_parallel_calls=AUTOTUNE)
        return data


class AugmentationStudy(AugmentationEngine):
    """
    For the augmentation study we only augment one branch for the image, keeping one of the copy in orignal state
    """

    def __init__(self, augmentation1, augmentation2):
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def augment(self, image):
        return self.augmentation2(self.augmentation1(image))

    def transform(self, data):
        """
        We augment 1 image and keep the second one the same
        :param data: Image to be augmented
        :return: (image, augmented image, image, labels)
        """

        return data.map(lambda x, y: (x, self.augmentation2(self.augmentation1(x)), x, y), num_parallel_calls=AUTOTUNE)


class LinearEvalAugmentation(AugmentationEngine):
    """
    Augmentation engine for linear evaluation
    """

    def transform(self, data):
        return data.map(lambda x, y: (linear_evaluation_augment(x), y), num_parallel_calls=AUTOTUNE)
