"""
This augmentation framework is an attempt of replication of the paper,
"RandAugment: Practical automated data augmentation with a reduced search space". Source: https://arxiv.org/pdf/1909.13719.pdf
Githubs that helped us get it to work, https://github.com/szacho/augmix-tf,https://github.com/ildoonet/pytorch-randaugment
"""

import math
import random

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def int_parameter(level, maxval):
    return tf.cast(level * maxval / 10, tf.int32)


def float_parameter(level, maxval):
    return tf.cast((level) * maxval / 10., tf.float32)


def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)


def affine_transform(image, transform_matrix):
    DIM = image.shape[0]
    XDIM = DIM % 2

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transform_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def blend(image1, image2, factor):
    """
    Blend 2 images
    :param image1:
    :param image2:
    :param factor: How they belnd
    :return:
    """

    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def shear_x(img, v):  # [-0.3, 0.3]
    print("applying shear_x")
    lvl = float_parameter(sample_level(v), 0.3 * 25)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    s2 = tf.math.sin(lvl)
    shear_x_matrix = tf.reshape(tf.concat([one, s2, zero, zero, one, zero, zero, zero, one], axis=0), [3, 3])

    transformed = affine_transform(img, shear_x_matrix)
    return transformed


def shear_y(img, v):  # [-0.3, 0.3]
    print("applying shear_y")
    lvl = float_parameter(sample_level(v), 0.3 * 25)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(lvl)
    shear_y_matrix = tf.reshape(tf.concat([one, zero, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    transformed = affine_transform(img, shear_y_matrix)
    return transformed


def translate_x(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    print("applying translate_x")
    lvl = int_parameter(sample_level(v), img.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_x_matrix = tf.reshape(tf.concat([one, zero, zero, zero, one, lvl, zero, zero, one], axis=0), [3, 3])

    transformed = affine_transform(img, translate_x_matrix)
    return transformed


def translate_y(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    print("applying translate_y")
    lvl = int_parameter(sample_level(v), img.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_y_matrix = tf.reshape(tf.concat([one, zero, lvl, zero, one, zero, zero, zero, one], axis=0), [3, 3])

    transformed = affine_transform(img, translate_y_matrix)
    return transformed


def rotate(img, v):  # [-30, 30]
    print("applying rotate")
    degrees = float_parameter(sample_level(v), 180)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    degrees = tf.cond(rand_var > 0.5, lambda: degrees, lambda: -degrees)
    degrees == v

    pi = tf.constant(math.pi)
    angle = pi * degrees / 180  # convert degrees to radians
    angle = tf.cast(angle, tf.float32)
    # define rotation matrix
    c1 = tf.math.cos(angle)
    s1 = tf.math.sin(angle)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    transformed = affine_transform(img, rotation_matrix)
    return transformed


def auto_contrast(img, _):
    print("applying AutoContrast")
    img = tf.cast(tf.math.scalar_mul(255, img), tf.int32)

    def scale_channel(img):
        lo = tf.cast(tf.reduce_min(img), tf.float32)
        hi = tf.cast(tf.reduce_max(img), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.int32)

        result = tf.cond(hi > lo, lambda: scale_values(img), lambda: img)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(img[:, :, 0])
    s2 = scale_channel(img[:, :, 1])
    s3 = scale_channel(img[:, :, 2])
    img = tf.stack([s1, s2, s3], 2)
    return img


def equalize(img, _):
    print("applying Equalize")

    def scale_channel(im, c):
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.float32)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(img, 0)
    s2 = scale_channel(img, 1)
    s3 = scale_channel(img, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def solarize(img, v):  # [0, 256]
    print("applying Solarize")
    threshold = 128
    return tf.where(img < threshold, img, 255 - img)


def solarize_add(img, addition=0, threshold=128):
    print("applying SolarizeAdd")
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    addition = tf.cond(rand_var > 0.5, lambda: addition, lambda: -addition)

    added_image = tf.cast(img, tf.float32) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 1), tf.float32)
    return tf.where(img < threshold, added_image, img)


def posterize(img, v):  # [4, 8]
    print("applying Posturize")
    lvl = int_parameter(sample_level(v), 8)
    shift = 8 - lvl
    shift = tf.cast(shift, tf.uint8)
    image = tf.cast(tf.math.scalar_mul(255, img), tf.uint8)
    image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    return image


def contrast(img, v):  # [0.1,1.9]
    print("applying Contrast")
    factor = float_parameter(sample_level(v), 2)
    factor = tf.reshape(factor, [])
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    factor = tf.cond(rand_var > 0.5, lambda: factor, lambda: 2.0 - factor)
    return tf.image.adjust_contrast(img, factor)


def color(img, v):  # [0.1,1.9]
    print("applying color")
    factor = float_parameter(sample_level(v), 1.8) + 0.1
    img = tf.cast(tf.math.scalar_mul(255, img), tf.uint8)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    blended = blend(degenerate, img, factor)
    return blended
    # return tf.cast(tf.clip_by_value(tf.math.divide(blended, 255), 0, 1), tf.float32)


def brightness(img, v):  # [0.1,1.9]
    print("applying Brightness")
    img = tf.cast(tf.clip_by_value(tf.math.divide(img, 255), 0, 1), tf.float32)
    delta = float_parameter(sample_level(v), 1)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    delta = tf.cond(rand_var > 0.5, lambda: delta, lambda: -delta)
    return tf.cast(tf.multiply(tf.clip_by_value(tf.image.adjust_brightness(img, delta=delta), 0, 1), 255), tf.int32)


def sharpness(img, v):  # [0.1,1.9]
    print("applying Sharpness")
    level = float_parameter(sample_level(v), 2)
    return tfa.image.sharpness(img, level)


def cutout(image, v):  # [0, 60] => percentage: [0, 0.2]
    print("applying Cutout")

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    replace = 0

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)
    cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    pad_size = int(image.get_shape().as_list()[0] * v)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace,
        image)
    return image


def flip(image, _):
    print("applying flip")
    """Applies a horizontal flip
    Args:
        image: a single image or batch of images (if can get to work with tensorflow-addons)
    Returns:
        Flipped image
    """
    flipped = tf.image.flip_left_right(image)
    return flipped


def crop_resize(image, _):
    print("applying crop_resize")
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


def identity(img, _):
    print("applying Identity")
    return img


def augment_list():
    """
    18 operations and their ranges
    Each tuple in the list is the function do to do.
    The second parameter is normalization values towards the strength value (min and max).
    Note not all functions use the strength. Hence , the min max value.
    :returns the list of tuples of augments with max and min value (augment function, min value, max value)
    """

    l = [
        (identity, 0., 10.),
        (shear_x, 0., 10.),
        (shear_y, 0., 10.),
        (translate_x, 0., 10.),
        (translate_y, 0., 10.),
        (rotate, 0, 10.),
        (auto_contrast, 0, 10.),
        (equalize, 0, 10.),
        (solarize, 0, 10.),
        (posterize, 4, 10.),
        (contrast, 0.1, 10.),
        (color, 0.1, 10.),
        (brightness, 0.1, 10.),
        (sharpness, 0.1, 10.),
        (cutout, 0, 0.2),  # Specific for CIFAR-10
        (flip, 0, 10.),  # Specific for CIFAR-10
        (crop_resize, 0, 10.)  # Specific for CIFAR-10
    ]
    return l


class RandAugment:
    def __init__(self, augments, strength):
        self.n = augments
        self.m = strength  # [0, 10]
        self.augment_list = augment_list()

    def __call__(self, img, labels):
        ops1 = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops1:
            val1 = (float(self.m) / 10) * float(maxval - minval) + minval
            img1 = op(img, val1)

        ops2 = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops2:
            val2 = (float(self.m) / 10) * float(maxval - minval) + minval
            img2 = op(img, val2)

        return tf.cast(img, tf.float32), tf.cast(img1, tf.float32), tf.cast(img2, tf.float32), labels
