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
            data = data.map(lambda x, y: (augment_batch(x, y)),
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


@tf.function
def show(x):
    for i, j in x: tf.print(i, j)


class AugmentationStudy(AugmentationEngine):
    def __init__(self, augmentation1, augmentation2):
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def augment(self, input_data, augmentation_type1, augmentation_type2):
        augmentations = [augmentation_type1, augmentation_type2]
        x = input_data
        for augmentation in augmentations:
            if augmentation == "crop":
                x = crop_resize(x)
            elif augmentation == "cutout":
                x = cut_out(x)
            elif augmentation == "color":
                x = color_jitter(x, s=1)
            elif augmentation == "sobel":
                x = sobel(x)
            elif augmentation == "gaussian_noise":
                x = gaussian_noise(x)
            elif augmentation == "gaussian_blur":
                std = random.uniform(.1, 2)
                x = gaussian_blur(x, std)
            elif augmentation == "rotate":
                x = rotate_randomly(x)
            elif augmentation == "nothing":
                pass
            else:
                raise Exception("Invalid argument for augment test")
        return x

    def transform(self, data, show_image_before=False, show_image_after=False):
        """
        We augment 1 image and keep the second one the same
        :param data:
        :return: (image, augmented image, image, labels)
        
        """

        if show_image_before:
            for i, j in data:
                plt.imshow(i.numpy() / 255)
                plt.show()

                img = sobel(i)
                # plt.imshow(img.numpy()/255)
                plt.imshow(img.numpy() / 255)
                plt.show()
                break

        data = data.map(lambda x, y: (
        x, tf.py_function(func=self.augment, inp=[x, self.augmentation1, self.augmentation2], Tout=tf.float32), x, y),
                        num_parallel_calls=AUTOTUNE)

        if show_image_after:
            for i, j, k, l in data:
                plt.imshow(j.numpy() / 255)
                plt.show()
                plt.imshow(k.numpy() / 255)
                plt.show()
                break

        return data
