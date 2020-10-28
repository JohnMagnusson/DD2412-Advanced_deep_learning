import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

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


def cut_out(image, pad_size=8, replace=0):
    """
    Apples random cutouts and replaces the whole with a constant value
    :param pad_size: Padding of the image
    :param replace: Value that the cut out will be replaced with
    :param image: Image to be augmented
    :return: Image with a cutout in it
    """
    # set padsize to half of the width/height of cutout box
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)
    cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace, image)

    return image


def gaussian_blur(image):
    """Applies gaussian blur with kernel size at 1/10th of the image size
    Args:
        image: a single image or batch of images (if can get to work with tensorflow-addons)
    Returns:
        Image(s) with a gaussian blur applied
    """
    std = random.uniform(.1, 2)  # std: a random value between 0.1 and 2
    width, height, color_channels = image.shape
    blured_image = tfa.image.gaussian_filter2d(image,
                                               (int(np.round(int(width) * .1, 0)), int(np.round(int(height) * .1, 0))),
                                               std)
    return blured_image


def flip(image):
    """Applies a horizontal flip
    Args:
        image: a single image or batch of images (if can get to work with tensorflow-addons)
    Returns:
        Flipped image
    """
    flipped = tf.image.flip_left_right(image)
    return flipped


def rotate_randomly(image):
    """
    Flips and image randomly in 360 degrees (in the paper they have constant, 90, 180, 270.
    We rotate randomly 0-360 degrees.
    :param image: Image to be augmented
    :return: Rotate image
    """
    return tf.keras.preprocessing.image.random_rotation(x=image.numpy(), row_axis=1, col_axis=0, channel_axis=2, rg=360)


def nothing(image):
    """
    Do no augmentation on the image
    :param image: Image to be augmented
    :return: Input image
    """
    return image


def color_jitter(image, s=1):
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


def color_drop(image):
    """Applies a color drop (B&W) to image(s)
    Args:
        image: a single image or batch of images
    Returns:
        Tensor of Image(s) with the color drop applied
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


def sobel(image):
    """
    Applies sobel filter to image
    :param image: Input image
    :return: Image with applied Sobel filter
    """

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    kernel = tf.tile(sobel_x[..., None], [1, 1, 3])[..., None]
    conv = tf.nn.depthwise_conv2d(image[None, ...], kernel, strides=[1, 1, 1, 1], padding='SAME')

    return tf.reshape(conv, [32, 32, 3])


def gaussian_noise(image):
    """
    Applies Gaussian noise to the images
    :param image: Input image
    :return: Noisy image
    """

    image = tf.clip_by_value(image / 255, 0, 1)
    with tf.name_scope('Add_gaussian_noise'):
        noise = tf.compat.v1.random.normal(shape=tf.shape(image), mean=0.0, stddev=(10) / (255), dtype=tf.float32)
        noise_img = image + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0) * 255
    return noise_img


def random_apply(image):
    """Randomly applies each augmentation according to the probabilities stated in SimCLR
    Args:
        image: a single image or batch of images
    Returns:
        Tensor of Image(s) with the random augmentations applied
    """

    # apply crop
    image = crop_resize(image)

    # apply flip
    rand = random.randrange(0, 100)
    if rand < 50:
        image = flip(image)

    # apply color jitter
    rand = random.randrange(0, 100)
    if rand < 80:
        image = color_jitter(image, flagSettings.color_jitter_strength)

    # apply color drop
    rand = random.randrange(0, 100)
    if rand < 20:
        image = color_drop(image)

    if flagSettings.use_gaussian_blur:  # On CIFAR-10 we do not use gaussian blur
        # apply gaussian blur
        rand = random.randrange(0, 100)
        if rand < 50:
            image = gaussian_blur(image)

    return image


def augment_batch(images, labels):
    """Applies all augmentations to batch according to SimCLR
    Args:
        images: a batch of images
        labels: the batches labels 
    Returns:
        original images, augmented images, and associated labels in separate lists
    """

    return images, random_apply(images), random_apply(images), labels


def fine_tune_augment(image):
    """
    Augmentation strategy used for fine-tuning the network
    :param image: Image to augment
    :return: Augmented image
    """

    # apply crop
    image = crop_resize(image)

    # apply flip
    rand = random.randrange(0, 100)
    if rand < 50:
        image = flip(image)

    return image.numpy()


def linear_evaluation_augment(image):
    """
    Augmentation strategy used for linear evaluation
    :param image: Image to augment
    :return: Augmented image
    """

    # apply flip
    rand = random.randrange(0, 100)
    if rand < 50:
        image = flip(image)

    # apply crop
    image = crop_resize(image)
    return image
