#abstact class representing a sequence of augmentation transformations


from dataAugmentations import *
from randAugment import *
AUTOTUNE = tf.data.experimental.AUTOTUNE
from flagSettings import *


class AugmentationEngine:
    def transform(self, data): #function to be implemented with a sequence of trasformations
        pass

class SimClrAugmentation(AugmentationEngine):
    def transform(self, data, augment=augmentation_type):
        if augment == 'simclr':
            print('using simclr augmentations')
            data = data.map(lambda x, y: (augmentBatch(x, y)),
                            num_parallel_calls=AUTOTUNE)
        if augment == 'rand':
            print('using randaug augmentations')
            aug = RandAugment(rand_augs,rand_strength)
            data = data.map(lambda x, y: (aug.__call__(x, y)),
                            num_parallel_calls=AUTOTUNE)
        return data

class TestAugmentation(AugmentationEngine):
    def transform(self, data):
        data = data.map(lambda x, y: (x, x, x, y),
                        num_parallel_calls=AUTOTUNE)
        print('using test augmentations')
        return data