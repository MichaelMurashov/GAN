from tensorlayer.prepro import *


def crop_sub_images(x, is_random=True):
    x = crop(x, 384, 384, is_random)
    x = x / (225. / 2.)
    x = x - 1
    return x


def downsample(x):
    x = imresize(x, [96, 96], 'bicubic')
    x = x / (225. / 2.)
    x = x - 1
    return x
