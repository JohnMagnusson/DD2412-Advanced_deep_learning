import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy import ndimage

import flagSettings


def crop_resize(image):
    """Crops image to random size and resizes to original shape
    Args:
        image: a single image or batch of images
    Returns:
        Tensor of image(s) croped and resized
    """
    width, height, color_channels = image.shape
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    ratio = int(width) / int(height)
    crop_start, crop_size, bound = tf.image.sample_distorted_bounding_box(image_size=tf.shape(image),
                                                                          bounding_boxes=bbox,
                                                                          min_object_covered=0.1,
                                                                          aspect_ratio_range=(
                                                                              3. / 4 * ratio, 4. / 3. * ratio),
                                                                          area_range=(0.08, 1.0),
                                                                          max_attempts=100)
    y_start, x_start, size = tf.unstack(crop_start)
    h, w, size = tf.unstack(crop_size)
    crop_img = tf.image.crop_to_bounding_box(image, y_start, x_start, h, w)
    original_size = tf.image.resize(crop_img, (height, width), method='nearest', preserve_aspect_ratio=False)

    return original_size


def cut_out(img):
    """
    Source: https://arxiv.org/pdf/1708.04552.pdf (16 × 16 pixels cut out size on cifar-10)
    Apples random cutouts and replaces the whole with a constant value
    :param img:
    :return: Image with a cutout in it
    """
    lol = 3
    visualizeTensorImage(img)
    image =  tfa.image.cutout_ops.random_cutout(images=img, mask_size=(15, 15), constant_values=0, seed=12,
                                              data_format="channels_last")
    visualizeTensorImage(image)
    return image

def gaussian_blur(image, std):
    """Applies gaussian blur with kernel size at 1/10th of the image size
    Args:
        image: a single image or batch of images (if can get to work with tensorflow-addons)
        std: a random value between 0.1 and 2
    Returns:
        Image(s) with a gaussian blur applied
    """
    width, height, color_channels = image.shape
    blured_image = tfa.image.gaussian_filter2d(image,
                                               (int(np.round(int(width) * .1, 0)), int(np.round(int(height) * .1, 0))),
                                               std)
    # blured_image = cv2.GaussianBlur(image,(int(np.round(int(width)*.1,0)),int(np.round(int(height)*.1,0))),std)
    return blured_image


def flip(image):
    """Applies a horizontal flip
    Args:
        image: a single image or batch of images (if can get to work with tensorflow-addons)
    Returns:
        Flipped image
    """
    # flipped = cv2.flip(image,1)
    flipped = tf.image.flip_left_right(image)
    return flipped


def rotate_randomly(image):
    """
    Flips and image randomly in 360 degrees
    :param image:
    :return:
    """

    return tf.keras.preprocessing.image.random_rotation(x=image, rg=360)


def color_jitter(image, s):
    """Applies a color jitter with random brightness, contrast, saturation, and hue
    Args:
        image: a single image or batch of images
        s: the strength of the jitter (default = 1)
    Returns:
        Tensor of Image(s) with the color jitter applied
    """
    image = tf.image.random_brightness(image, max_delta=0.8 * s)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    image = tf.image.random_hue(image, max_delta=0.2 * s)
    # image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def colorDrop(image):
    """Applies a color drop (B&W) to image(s)
    Args:
        image: a single image or batch of images
    Returns:
        Tensor of Image(s) with the color drop applied
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


# Todo verify that this works correctly, source: https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
def sobel(image):
    """
    Applies sobel filter to image
    :param image: Input image
    :return: Image with applied Sobel filter
    """
    image = image.astype('int32')
    dx = ndimage.sobel(image, 1)  # horizontal derivative
    dy = ndimage.sobel(image, 0)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return image


def visualizeTensorImage(img):
    """Prints the image from a tensor form
    Args:
        image: a single image
    Returns:
        Shows image
    """
    # image = tf.Session().run(image)   # old stuff, eeekk
    # image = np.squeeze(image)
    # plt.imshow(np.round(image, 0))
    # print(img)
    # # todo test tthe under
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # # img = tf.image.resize(img, [500,500])
    # plt.imshow(img.numpy())
# for x,y in image: print(x,y)
    return img
        # plt.imshow(x.numpy()/255)


def randomApply(image):
    """Randomly applies each augmentation according to the probabilities stated in SimCLR
    Args:
        image: a single image or batch of images
    Returns:
        Tensor of Image(s) with the random augmentations applied
    """
    # normal_image = image.copy()

    # apply flip
    rand = random.randrange(0, 100)
    if rand < 50:
        image = flip(image)

    # apply crop
    image = crop_resize(image)

    # apply color jitter
    rand = random.randrange(0, 100)
    if rand < 80:
        image = color_jitter(image, flagSettings.color_jitter_strength)

    # apply color drop
    rand = random.randrange(0, 100)
    if rand < 20:
        image = colorDrop(image)

    if flagSettings.use_gaussian_blur:
        # apply gaussian blur
        rand = random.randrange(0, 100)
        if rand < 50:
            std = random.uniform(.1, 2)
            image = gaussian_blur(image, std)
        else:
            pass

    return image


def augmentBatch(images, labels):
    """Applies all augmentations to batch according to SimCLR
    Args:
        images: a batch of images
        labels: the batches labels 
    Returns:
        original images, augmented images, and associated labels in separate lists
    """
    augment1 = []
    augment2 = []

    # for image in images:
    #    augment1.append(randomApply(images))
    #    augment2.append(randomApply(images))

    return images, randomApply(images), randomApply(images), labels


def fine_tune_augment(image):
    # apply flip
    # rand = random.randrange(0, 100)
    # if rand < 50:
    #     image = flip(image)
    #
    # # apply crop
    # image = crop_resize(image)
    return image
