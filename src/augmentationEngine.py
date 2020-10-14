#abstact class representing a sequence of augmentation transformations


from src.dataAugmentations import *
AUTOTUNE = tf.data.experimental.AUTOTUNE


class AugmentationEngine:
    def transform(self, data): #function to be implemented with a sequence of trasformations
        pass

class SimClrAugmentation(AugmentationEngine):
    def transform(self, data):
        data = data.map(lambda x, y: (augmentBatch(x, y)),
                        num_parallel_calls=AUTOTUNE)
        return data

class TestAugmentation(AugmentationEngine):
    def transform(self, data):
        data = data.map(lambda x, y: (x, x, x, y),
                        num_parallel_calls=AUTOTUNE)
        return data