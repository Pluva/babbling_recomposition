from __future__ import print_function

#python
import csv
import matplotlib.pyplot as plt
import os

#keras
from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.models import load_model
# from keras.optimizers import SGD
# from keras import applications

#numpy
import numpy as np

#local

# ---------------------- ***** ---------------------- #
# ---------------------- TOOLS ---------------------- #
# ---------------------- ***** ---------------------- #

# ---------------------- HISTORY ---------------------- #

def initialize_headers_filepath(param_headers, filepath):
    """
        Initialize the given file with the given headers.

        :param param_headers: List of headers.
        :param filepath: Path to the file.
        :type param_headers: list ['header1', 'header2', ... ]
        :type filepath: string
    """

    with open(filepath, 'w') as csv_file:
        initialize_headers_file(param_headers, csv_file)
    return;

def initialize_headers_file(param_headers, csv_file):
    """
        Initialize the given opened file object with the given headers.

        :param param_headers: List of headers.
        :param csv_file: File to write to.
        :type param_headers: list ['header1', 'header2', ... ]
        :type csv_file: opened file
    """

    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(param_headers)
    return;

def log_history_to_csv_filepath(metaparams, training_perfs, filepath):
    """ 
        Write the perfs of a model into the given filepath,
        open and close the file at each call,
        use log_history_to_csv_file for cross-validation usage.

        :param metaparams: Values of the metaparameters used for training.
        :param history: Training performances (see keras doc for details).
        :param filepath: Absolute path to the csv file.
        :type metaparams: list [metaparam1_value, metaparam2_value, ... ]
        :type history: keras training history object (see keras doc for details).
        :type filepath: string
    """

    with open(filepath, 'wa') as csv_file:
        log_history_to_csv_file(metaparams, training_perfs, csv_file)
    return;

def log_history_to_csv_file(metaparams, history, csv_file):
    """
        Write the perfs of a model into the given file, the file needs to be already open.

        :param metaparams: Values of the metaparameters used for training.
        :param history: Training performances (see keras doc for details).
        :param csv_file: Opened csv file to write to.
        :type metaparams: list [metaparam1_value, metaparam2_value, ... ]
        :type history: keras training history object (see keras doc for details).
        :type csv_file: file instance
    """
    
    csv_writer =  csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(history.history['loss'])):
        training_perfs = [i+1, history.history['loss'][i], history.history['acc'][i], history.history['val_loss'][i], history.history['val_acc'][i]]
        csv_writer.writerow(metaparams + training_perfs)
    return;

# ---------------------- MODELS ---------------------- #

def randomize_layers(nb_layers, old_model, model_type='Model'):
    """
        Randomize the n top layers of a model.
        In lack of a better solution, for now this function generate a new model, 
        and then copy the weigts of the old model

        :param nb_layers:
        :param old_model:
        :param model_type:
        :return: A newly instanciated model.
    """
    
    config = old_model.get_config()
    if model_type=='Model':
        new_model = Model.from_config(config)
    elif model_type=='Sequential':
        new_model = Sequential.from_config(config)
    else:
        print('Wrong parameter, model can only be Sequential or Model.')

    if nb_layers==-1:
        nb_layers = len(new_model.layers)
    else:
        nb_layers = min(nb_layers, len(new_model.layers))

    # Copy the weights of the non-randomized layers.
    for layer_i in range(len(new_model.layers) - nb_layers):
        new_model.layers[layer_i].set_weights(old_model.layers[layer_i].get_weights())

    del old_model

    return new_model

# Shuffle weights method from @jkleint 'MANY THANKS'
def shuffle_weights(model, layers_to_shuffle, weights=None):
    """
        Randomly permute the weights in `model`, or the given `weights`.
        This is a fast approximation of re-initializing the weights of a model.
        Assumes weights are distributed independently of the dimensions of the weight tensors
          (i.e., the weights have the same distribution along each dimension).

        :param Model model: Modify the weights of the given model.
        :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
        :param integer layers_to_shuffle: Number of layers to reinitialise starting from the top.
          If `None`, permute the model's current weights.

    """

    if weights is None:
        weights = model.get_weights()
    random_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]

    if layers_to_shuffle >= 0:
        starting_layer = max(0, len(weights) - layers_to_shuffle - 1)
    else:
        starting_layer = 0

    for i in range(starting_layer, len(weights)):
        weights[i] = random_weights[i]

    model.set_weights(weights)


def heatmaps_softmax(heatmaps):
    """
        Return the softmax values of the given heatmaps.
        Each heatmap must be the predicted value of one class.
        Therefore the returned maps will be the softmax predicted value for each class.
        The softmax function is applied along all class, by pixel.

        :param heatmaps: Array of heatmaps.
        :return: Array of updated heatmaps.
        :type heatmaps: Numpy array of dimension (w, h, nb_cls).
        :rtype: Numpy array of dimension (w, h, nb_cls).
    """

    ret = np.zeros(heatmaps.shape, heatmaps.dtype)

    width, height, _ = heatmaps.shape

    for w in range(width):
        for h in range(height):
            x = heatmaps[w,h]
            e = np.exp(x - np.max(x))
            ret[w,h] = e / np.sum(e, axis=0)

    return ret
