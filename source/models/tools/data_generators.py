
"""
    Data Generators Functions.
    Handle the loading of dataset into usable format for keras networks.
    Certain functions assumes the format of the labels of this dataset.

    read_numpy_data_file:
    generate_numpy_arrays:
"""

#python
import os
import fnmatch
import random

#numpy
import numpy as np

#keras
import keras.backend as K
from keras.preprocessing import image as Image




def read_numpy_data_file(directory, filename, types, fnames_format='relative'):
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

    for line in open(os.path.join(directory, filename), mode='r'):
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

def generate_numpy_arrays(files, labels, image_shape, label_shape, label_type=float, data_format=None, grayscale=False):
    """
        Read the given input files parameter, and labels parameters.
        Load the corresponding images.
        Return two numpy array, one containg the images, one containing the labels.

        :param classes: IDs of the images to load.
        :param labels: File containing the association between the IDs and labels. 
        :param label_shape: Shape of the label data.
        :param label_type: Type for the labels, unique here for efficiency and quick handling.
        :param data_format: Image data format, used to load images. By default corresponds to that of Keras backend.
        :param grayscale: Loar images as grayscale [default=False, True].
        :type classes: int list
        :type labels: string
        :type label_shape: (int, )
        :type label_type: python types [default=float, int]
        :type data_format: dtype
        :type grayscale: boolean
        :return: Two numpy array, respectively the images and the labels.
        :rtype: (numpy array, numpy array)
    """

    if data_format is None:
        data_format = K.image_data_format()

    images = np.zeros((len(files), ) + image_shape, dtype=K.floatx())
    lbls = np.zeros((len(files), ) + label_shape, dtype=label_type)

    i = 0
    for file in files:
        img = Image.load_img(file, grayscale=grayscale, target_size=image_shape)
        x = Image.img_to_array(img, data_format=data_format)
        images[i] = x
    
        for dim in range(len(label_shape)):
            lbls[i][dim] = labels[i][dim]

        i += 1

    return (images, lbls)

def load_images_from_folder(folder_path, pattern, data_format=None, grayscale=False, verbose=0):
    """
        Browse a given repertory, load all images corresponding to given base name.
        Return a numpy array containing the images.

        :param folder_path: Path to the image folder.
        :param pattern: Pattern of the filename to look for.
        :param data_format: Image channel format, recommended to leave default behavior.
        :param grayscale: Convert images to grayscale.
        :param verbose: Verbose behavior.
        :return: The loaded images and corresponding filenames.
        :type folder_path: string
        :type pattern: string
        :type data_format: string
        :type grayscale: boolean
        :type verbose: int
        :rtype: Numpy array
    """

    if verbose > 0: print('Starting load_images_from_folder:')

    if data_format is None:
        data_format = K.image_data_format()
        if verbose > 0: print('-- Data format is None, initializing to: {}'.format(data_format))

    images = []
    fnames = []

    for filename in os.listdir(folder_path):
        if fnmatch.fnmatch(filename, pattern):
            if verbose > 0: print('-- Loading image: {}'.format(filename))
            img = Image.load_img(os.path.join(folder_path, filename), grayscale=grayscale)
            x = Image.img_to_array(img, data_format=data_format)
            images.append(x)
            fnames.append(filename)

    return images, fnames


def shuffle_dataset(images, labels):
    """
        Return a shuffled version of the dataset given as input.
        Correspondance between images and labels are maitained.

        :param images: 
        :param labels:
        :return: Shuffled arrays images and labels.
        ;type images:
        :type labels:
        :rtype: Same type as input.
    """

    permutation = np.random.permutation(len(images))

    perm_imgs = images[permutation]
    perm_lbls = labels[permutation]

    return perm_imgs, perm_lbls

def _deprec_discretize_labels(labels, lbls_type_in, lbls_type_out, tresholds, verbose=0):
    """
        Discretize the values contained in labels in accord with the given tresholds.
        

        :param labels: Array of labels to dicretize.
        :param tresholds: Array of array of boundaries.
        :param lbls_types_in:
        :param lbls_types_in:
        :param tresholds: Thresholds values for each labels dimension.
        :param verbose: Verbose behavior.
        :return: The label array with discretized values for the labels.
        :type labels: numpy array [[lbl1_value, lbl2_value, ...], ]
        :type lbls_types_in:
        :type lbls_types_out:
        :type tresholds: [[dim1_tr1, dim1_tr2, ...], [dim2_tr1, ...]]
        :type verbose: int [default=0, 1]
        :rtype: [[lbl1_value, lbl2_value, ...], ]
    """

    lbls = np.zeros(labels.shape, dtype=lbls_type_out)

    if verbose > 0:
        print('-Discretising labels:')
        ccounts = np.zeros((len(tresholds),), dtype=object)
        for dim in range(len(tresholds)):
            ccounts[dim] = np.zeros((len(tresholds[dim]) + 1,), dtype=int)

    for lbl in range(len(labels)):
        for dim in range(len(tresholds)):
            cl = __get_label_class(labels[lbl][dim], tresholds[dim])
            lbls[lbl][dim] = cl
            if verbose > 0:
                ccounts[dim][cl] += 1

    if verbose > 0:
        print('-Processed {} labels, ratios per class = {}.'.format(len(labels), np.divide(ccounts, float(len(labels)))))

    return lbls

def discretize_labels(labels, tresholds, verbose=0):
    """
        Discretize the labels given in inputs into classes defined in the tresholds.
        Be careful this function is hardly dependent on the labels format.
        In this case the labels have to be single dimensioned.

        :param labels: List of labels to discretize.
        :param tresholds: List of tresholds to define the classes.
            Each value defining two classes, above and under.
        :param verbose: Verbose behavior.
        :return: List of discretized labels.
        :type labels: float list
        :type tresholds: float list
        :type verbose: int
        :rtype: int list
    """

    lbls = np.zeros((len(labels), len(tresholds)+1), dtype=int)

    if verbose > 0:
        print('-Discretising labels:')
        ccounts = np.zeros((len(tresholds) + 1, ), dtype=object)
        
    for lbl in range(len(labels)):
        cl = __get_label_class(labels[lbl], tresholds)
        lbls[lbl,cl] = 1
        if verbose > 0:
            ccounts[cl] += 1

    if verbose > 0:
        print('-Processed {} labels, ratios per class = {}.'.format(len(labels), np.divide(ccounts, float(len(labels)))))

    return lbls

def __get_label_class(lbl, tresholds):
    """
        Return the class associated with the given label.
    """
    lbl_class = 0

    while (lbl_class < len(tresholds)) and (lbl > tresholds[lbl_class]) :
        lbl_class += 1

    return lbl_class

def _count_tresholds_1D(labels, tresholds, dimension):
    """
        Count the number of lbls inside each bounded box defined by the given tresholds.
    """
    class_counters = np.zeros(len(tresholds[dimension]) + 1)

    for lbl in labels:
        lbl_class = __get_label_class(lbl[dimension], tresholds[dimension])
        class_counters[lbl_class] += 1

    return class_counters

def equalize_dataset_continuous_1D(images, labels, tresholds, ratios, dimension, eq_ratios=False, verbose=0):
    """
        Equalize the ratio of data between classes.
        Tresholds must be in crescent order.
        The given dimension is the one used to compute the pick proportion.
        Whenever several dimensions should be used, for now only solution is to call this function sequentially.

        :param images: Array of images.
        :param labels: Array of labels.
        :param tresholds: Tresholds to use to distinguish classes.
        :param ratios: Ratios correspondings to the given tresholds.
        :param dimension: Dimension of the label to consider.
        :param eq_ratios: Equalize the picking ratios to fit the input ratios [default=False, True].
        :param verbose: Verbose behavior.
        :return: The equaliezd datasets (images, labels)
        :type images: np.array
        :type labels: np.array
        :type tresholds: float list
        :type ratios: float list (must of dimension len(tresholds) + 1)
        :type dimension: int
        :type eq_ratios: boolean
        :rtype: (np.array, np.array)
    """
    
    # compute ratios proportions
    rprob_n = np.divide(ratios, float(np.sum(ratios)))
    rprob = rprob_n

    if eq_ratios:
        rprob_e = np.zeros(len(ratios))
        class_counters = _count_tresholds_1D(labels, tresholds, dimension=dimension)
        class_ratios = np.divide(class_counters, float(np.sum(class_counters)))

        for i in range(len(ratios)):
            rprob_e[i] = rprob_n[i] / class_ratios[i]
        rprob_e = np.divide(rprob_e, np.sum(rprob_e))
        rprob = rprob_e


    imgs = []
    lbls = []

    if verbose > 0:
        print('--Tresholds= {}, ratios= {}'.format(tresholds, ratios))
        print('--Normalized ratios= {}'.format(rprob_n))
        if eq_ratios:
            print('--Equalized ratios= {}'.format(rprob_e))

    for i in range(len(images)):
        lbl = __get_label_class(labels[i][dimension], tresholds[dimension])
        pick = random.random() <= rprob[lbl]
        if pick:
            imgs.append(images[i])
            lbls.append(labels[i])

    return np.array(imgs), np.array(lbls)