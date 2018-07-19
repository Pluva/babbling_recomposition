
"""
    Data Generators Functions.
    Handle the loading of dataset into usable format for keras networks.

    read_numpy_data_file:
    generate_numpy_arrays:
"""

#python
import numpy as np
import os
# import multiprocessing.pool
# from functools import partial

#keras
import keras.backend as K
from keras.preprocessing import image as Image




def read_numpy_data_file(directory, filename, types, fnames_format='absolute'):
    """"
        Read and returns the list of files to open as a numpy array, and the corresponding array of labels.
        The given file should contain the list of files and associated labels.
        
        :param directory: Absolute path to the directory containing the source file.
        :param filename: Name of the file containing the files to load and their corresponding labels.
            Each line is formatted as follows 'filename;lbl1;lbl2;lbl3;...' 
        :param types: Specify types of the variables to find in the file.
        :param fnames_format: Format of the filenames [defautl=absolute, relative].
        :type directory: string
        :type filename: string
        :type types: list of variables types
        :type fnames_format: string
        :return: The list of files and list of labels.
        :rtype: ([string], [[types]])
    """

    files = []
    labels = []

    for line in open(os.path.join(directory, filename)):
        line = line.split('\n')[0]
        splits = line.split(';')

        if fnames_format == 'absolute':
            files.append(splits[0])
        elif fnames_format == 'relative':
            files.append(os.path.join(directory, splits[0]))

        label = []
        t = 0
        for value in splits[1:]:
            value = types[t](value)
            t += 1
            label.append(value)
        
        labels.append(label)


    return (files, labels)


def generate_numpy_arrays(files, labels, image_shape, label_shape, label_types, data_format=None, grayscale=False):
    """
        Return two numpy array, one containg the images, one containing the labels.

        :param classes: IDs of the images to load.
        :param labels: File containing the association between the IDs and labels. 
        :param label_shape: Shape of the label data.
        :param label_types: Types of each label dimension.
        :param data_format: 
        :param grayscale: Loar images as grayscale [default=False, True].
        :type classes: int list
        :type labels: string
        :type label_shape: (int, )
        :type label_types: [type_dim1, ]
        :type data_format:
        :type grayscale: boolean
        :return: Two numpy array, respectively the images and the labels.
        :rtype: (numpy array, numpy array)
    """

    if data_format is None:
        data_format = K.image_data_format()

    images = np.zeros((len(files), ) + image_shape, dtype=K.floatx())
    lbls = np.zeros((len(files), ) + label_shape, dtype=label_types)

    i = 0
    for file in files:
        img = Image.load_img(file, grayscale=grayscale, target_size=image_shape)
        x = Image.img_to_array(img, data_format=data_format)
        images[i] = x
    
        lbls[i] = labels[i]

        i += 1


    return (images, lbls)


def data_aug_gen():
    """
        Construct and return an ImageDataGenerator with the given numpy dataset.
    """

    Image.ImageDataGenerator()



    return