#using methods from https://arxiv.org/pdf/1909.13719.pdf, https://github.com/szacho/augmix-tf/blob/master/augmix/transformations.py, https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import math
import matplotlib.pyplot as plt

def int_parameter(level, maxval):
    return tf.cast(level * maxval / 10, tf.int32)

def float_parameter(level, maxval):
    return tf.cast((level) * maxval / 10., tf.float32)

def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)

def affine_transform(image, transform_matrix):
    DIM = image.shape[0]
    XDIM = DIM%2 
    
    x = tf.repeat(tf.range(DIM//2,-DIM//2,-1), DIM)
    y = tf.tile(tf.range(-DIM//2,DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transform_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM,DIM,3])

def blend(image1, image2, factor):
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


def ShearX(img, v):  # [-0.3, 0.3]
    print("applying ShearX")
    lvl = float_parameter(sample_level(v), 0.3*25)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    s2 = tf.math.sin(lvl)
    shear_x_matrix = tf.reshape(tf.concat([one,s2,zero, zero,one,zero, zero,zero,one],axis=0), [3,3])   

    transformed = affine_transform(img, shear_x_matrix)
    return transformed


def ShearY(img, v):  # [-0.3, 0.3]
    print("applying ShearY")
    lvl = float_parameter(sample_level(v), 0.3*25)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(lvl)
    shear_y_matrix = tf.reshape(tf.concat([one,zero,zero, zero,c2,zero, zero,zero,one],axis=0), [3,3])   
    
    transformed = affine_transform(img, shear_y_matrix)
    return transformed


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    print("applying TranslateX")
    lvl = int_parameter(sample_level(v), img.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_x_matrix = tf.reshape(tf.concat([one,zero,zero, zero,one,lvl, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(img, translate_x_matrix)
    return transformed


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    print("applying TranslateY")
    lvl = int_parameter(sample_level(v), img.shape[0] / 3)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    lvl = tf.cond(rand_var > 0.5, lambda: lvl, lambda: -lvl)

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    lvl = tf.cast(lvl, tf.float32)
    translate_y_matrix = tf.reshape(tf.concat([one,zero,lvl, zero,one,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(img, translate_y_matrix)
    return transformed


def Rotate(img, v):  # [-30, 30]
    print("applying Rotate")
    degrees = float_parameter(sample_level(v), 180)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    degrees = tf.cond(rand_var > 0.5, lambda: degrees, lambda: -degrees)
    degrees == v

    pi = tf.constant(math.pi)
    angle = pi*degrees/180 # convert degrees to radians
    angle = tf.cast(angle, tf.float32)
    # define rotation matrix
    c1 = tf.math.cos(angle)
    s1 = tf.math.sin(angle)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one], axis=0), [3,3])

    transformed = affine_transform(img, rotation_matrix)
    return transformed


def AutoContrast(img, _):
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


def Invert(img, _):
    print("applying Invert")
    #img = tf.convert_to_tensor(img)
    return 255 - img

def Equalize(img, _):
    print("applying Equalize")
    image = tf.cast(tf.math.scalar_mul(255, img), tf.uint8)

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


def Solarize(img, v):  # [0, 256]
    print("applying Solarize")
    threshold = 128
    return tf.where(img < threshold, img, 255 - img)


def SolarizeAdd(img, addition=0, threshold=128):
    print("applying SolarizeAdd")
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    addition = tf.cond(rand_var > 0.5, lambda: addition, lambda: -addition)

    added_image = tf.cast(img, tf.float32) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 1), tf.float32)
    return tf.where(img < threshold, added_image, img)


def Posterize(img, v):  # [4, 8]
    print("applying Posturize")
    lvl = int_parameter(sample_level(v), 8)
    shift = 8 - lvl
    shift = tf.cast(shift, tf.uint8)
    image = tf.cast(tf.math.scalar_mul(255, img), tf.uint8)
    image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    return image
    #return tf.cast(tf.clip_by_value(tf.math.divide(img, 255), 0, 1), tf.float32)


def Contrast(img, v):  # [0.1,1.9]
    print("applying Contrast")
    factor = float_parameter(sample_level(v), 2)
    factor = tf.reshape(factor, [])
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    factor = tf.cond(rand_var > 0.5, lambda: factor, lambda: 2.0 - factor)
    return tf.image.adjust_contrast(img, factor)



def Color(img, v):  # [0.1,1.9]
    print("applying color")
    factor = float_parameter(sample_level(v), 1.8) + 0.1
    img = tf.cast(tf.math.scalar_mul(255, img), tf.uint8)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))
    blended = blend(degenerate, img, factor)
    return blended
    #return tf.cast(tf.clip_by_value(tf.math.divide(blended, 255), 0, 1), tf.float32)


def Brightness(img, v):  # [0.1,1.9]
    print("applying Brightness")
    img = tf.cast(tf.clip_by_value(tf.math.divide(img,255), 0, 1), tf.float32)
    delta = float_parameter(sample_level(v), 1)
    rand_var = tf.random.uniform(shape=[], dtype=tf.float32)
    delta = tf.cond(rand_var > 0.5, lambda: delta, lambda: -delta) 
    return tf.cast(tf.multiply(tf.clip_by_value(tf.image.adjust_brightness(img, delta=delta),0,1),255),tf.int32)


def Sharpness(img, v):  # [0.1,1.9]
    #need addons #havn't tested yet
    print("applying Sharpness")
    level = int_parameter(sample_level(v), 2)
    return tfa.image.sharpness(img,level)


def Cutout(image, v):  # [0, 60] => percentage: [0, 0.2]
    print("applying Cutout")
    pad_size = v
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random_uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)

    cutout_center_width = tf.random_uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

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


def Identity(img, v):
    print("applying Identity")
    return img


def augment_list():  # 16 oeprations and their ranges
    l = [
        (Identity, 0., 10.), 
        (ShearX, 0., 10.),  # 0
        (ShearY, 0., 10.),  # 1 
        (TranslateX, 0., 10.),  # 2
        (TranslateY, 0., 10.),  # 3
        (Rotate, 0, 10.),  # 4
        (AutoContrast, 0, 10.),  # 5 
        #(Invert, 0, 1),  # 6 #not in paper
        (Equalize, 0, 10.),  # 7
        (Solarize, 0, 10.),  # 8
        (Posterize, 4, 10.),  # 9 
        (Contrast, 0.1, 10.),  # 10
        (Color, 0.1, 10.),  # 11
        (Brightness, 0.1, 10.),  # 12
        (Sharpness, 0.1, 10.),  # 13 #need addons #havn't tested yet
        #(Cutout, 0, 0.2),  # 14 #not in paper
        ]
    return l


class RandAugment:
    def __init__(self, augments, strength):
        self.n = augments
        self.m = strength      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img, labels):
        ops1 = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops1:
            val1 = (float(self.m) / 30) * float(maxval - minval) + minval
            img1 = op(img, val1)
            
        ops2 = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops2:
            val2 = (float(self.m) / 30) * float(maxval - minval) + minval
            img2 = op(img, val2)


        return tf.cast(img,tf.float32),tf.cast(img1,tf.float32), tf.cast(img2,tf.float32), labels

