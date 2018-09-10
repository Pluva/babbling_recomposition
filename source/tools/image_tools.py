"""
    Image tools.
"""


#python
import os

#numpy
import numpy as np


def get_sub_images(image, sub_shape, padding=None):
    """
        !! DEPRECATED FOR NOW, FAR TOO COSTLY COMPUTATION WISE. !!
        Return an array of all the sub images of given size from given source image.
    
        :param image: Source image.
        :param sub_shape: Sub image size.
        :param padding: Padding size, set to None for no padding.
        :return: An array of images.
        :type image: numpy array
        :type sub_shape: tuple
        :type padding: int list ((before1, after1) [,(before2, after2)]) [default=None]
        :rtype: numpy array
    """

    if padding != None:
        image = np.pad(image, padding, 'constant', constant_value=(0))

    w,h,c = image.shape
    sw, sh, sc = sub_shape

    nb_imgs = (w-sw-1)*(h-sh-1)
    sub_images = np.zeros((nb_imgs, ) + sub_shape, dtype=image.dtype)

    #greedy method
    #Unpractical
    j = 0
    for i in range(w-sw-1):
        for k in range(h-sh-1):
            sub_images[j] = image[i:i+sw, k:k+sh]
            j+=1

    return sub_images
