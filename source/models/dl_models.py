
#python
from __future__ import print_function
import numpy as np

#keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import convolutional, MaxPooling2D
from keras.layers import *
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import applications

#local


# ---------------------- ****** ---------------------- #
# ---------------------- MODELS ---------------------- #
# ---------------------- ****** ---------------------- #

# Global constant relating to predefined models
DL_MODEL_CIFAR10 = 1
DL_MODEL_VGG16 = 2
DL_MODEL_VGG16_K = 4
DL_MODEL_6CONV = 6
DL_MODEL_8CONV = 7

DL_MODEL_NAMES = {
    1: 'cifar10',
    2: 'vgg16',
    4: 'vgg16_k',
    6: '6conv',
    7: '8conv'
}

def construct_model_by_id(model_type, nb_classes, input_shape, last_layers_sizes, name_top):
    """
        Construction function to call for easier code handling.
        Construct and returns a model from the given parameters.

        :param model_type: Type of model to construct.
        :param nb_classes: Number of classes.
        :param input_shape: Image shape (rows, cols, channels)
        :param last_layer_sizes: Number of parameters for the last dense layers.
        :param name_top: Prefix to put as name of the last dense layers.
        :type model_type: int
        :type nb_classes: int
        :type input_shape: (int, int, int)
        :type last_layer_sizes: [int dense1, int dense2, ...]
        :type name_top: string
        :return: The Keras Model
        :rtype: keras.model
    """

    if model_type == DL_MODEL_CIFAR10:
        model = construct_model_CIFAR10(nb_classes=nb_classes,
            input_shape=input_shape, last_layers_sizes=last_layers_sizes, name_top=name_top)

    elif model_type == DL_MODEL_VGG16:
        model = construct_model_VGG16(nb_classes=nb_classes,
            input_shape=input_shape, last_layers_sizes=last_layers_sizes, name_top=name_top)

    elif model_type == DL_MODEL_VGG16_K:
        model = keras_model_VGG16(nb_classes=nb_classes,
            input_shape=input_shape, last_layers_sizes=last_layers_sizes, name_top=name_top)
 
    elif model_type == DL_MODEL_6CONV:
        model = construct_model_6CONV(nb_classes=nb_classes,
            input_shape=input_shape, last_layers_sizes=last_layers_sizes, name_top=name_top)

    elif model_type == DL_MODEL_8CONV:
        model = construct_model_8CONV(nb_classes=nb_classes,
            input_shape=input_shape, last_layers_sizes=last_layers_sizes, name_top=name_top)

    else:
        raise ValueError('Model should be one of the following: ' + str(DL_MODEL_NAMES))

    return model


#
# --------------------------- MODELS ---------------------------
#

def construct_model_CIFAR10(nb_classes, input_shape, last_layers_sizes=[256], name_top='fc'):
    """
        Create and returns a CIFAR10 model.

        :param nb_classes: Number of classes to predict.
        :param input_shape: Shape of the input data.
        :param last_layers_sizes: Size of the last dense layer at the top of the network.
        :param name_top: Prefix to put as name of the last dense layers.
        :type nb_classes: int
        :type input_shape: tuple
        :type last_layers_sizes: [int, ... ]
        :type name_top: string
        :return: VGG16 keras model.
    """

    i = Input(input_shape)
    
    # Block1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.25)(x)

    # Block2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.25)(x)

    # Top
    x = Flatten()(x)
    _i = 1
    for layer_size in last_layers_sizes:
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        # x = Dense(layer_size, activation='relu')(x)
        x = Dropout(0.5)(x)
        _i += 1
    x = Dense(nb_classes, activation='softmax', name=name_top + str(_i))(x)
    # x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=i, outputs=x)

    return model

def construct_model_VGG16(nb_classes, input_shape, last_layers_sizes=[4096,4096], name_top='fc'):
    """
        Construct and returns a functionnal VGG16 model.

        :param nb_classes: Number of classes.
        :param input_shape: Image shape (rows, cols, channels)
        :param last_layer_sizes: Number of parameters for the last dense layers.
        :param name_top: Prefix to put as name of the last dense layers.
        :type nb_classes: int
        :type input_shape: (int, int, int)
        :type last_layer_sizes: [int dense1, int dense2, ...]
        :type name_top: string
        :return: The Keras Model
        :rtype: keras.model
    """

    # input
    i = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)

    _i = 1
    for layer_size in  last_layers_sizes:    
        # x = Dense(layer_size, activation='relu', name='fc' + str(_i))(x)
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        x = Dropout(0.5)(x)
        _i += 1

    # x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    x = Dense(nb_classes, activation='softmax', name=name_top + str(_i))(x)

    model = Model(inputs=i, outputs=x, name='vgg16')

    return model

def keras_model_VGG16(nb_classes, input_shape, last_layers_sizes=[4096,4096], name_top='fc',
    weights='imagenet', input_tensor=None):
    """ 
        Generate and returns the predifined VGG16 network from keras.

        :param nb_classes: Number of classes.
        :param input_shape: Image shape (rows, cols, channels)
        :param last_layer_sizes: Number of parameters for the last dense layers.
        :param name_top: Prefix to put as name of the last dense layers.
        :param weights: Weights to be pre-loaded in the network (see keras documentations for details). 
        :param input_tensor: Input Tensor if specific input shape.
        :type nb_classes: int
        :type input_shape: (int, int, int)
        :type last_layer_sizes: [int dense1, int dense2, ... ]
        :type name_top: string
        :type weights: string
        :type input_tensor: Keras tensor
        :return: The keras model, uncompiled.
        :rtype: keras.model
    """

    model = applications.VGG16(include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape)

    x = model.output
    x = Flatten(name='flatten')(x)

    _i = 1
    for layer_size in  last_layers_sizes:    
        # x = Dense(layer_size, activation='relu', name='fc' + str(_i))(x)
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        x = Dropout(0.5)(x)
        _i += 1

    # x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    x = Dense(nb_classes, activation='softmax', name=name_top + str(_i))(x)

    model = Model(inputs=model.input, outputs=x, name='vgg16')

    return model;




def construct_model_6CONV(nb_classes, input_shape, last_layers_sizes=[256], name_top='fc'):
    """
        Create and returns a custom 6 convolutionnal layers model.

        :param nb_classes: Number of classes to predict.
        :param input_shape: Shape of the input data.
        :param last_layers_sizes: Size of the last dense layer at the top of the network.
        :param name_top: Prefix to put as name of the last dense layers.
        :type nb_classes: int
        :type input_shape: tuple
        :type last_layers_sizes: [int, ... ]
        :type name_top: string
        :return: The constructed keras model, uncompiled.
        :rtype: keras.model
    """

    i = Input(input_shape)
    
    # Block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.25)(x)

    # Block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.25)(x)

    # Block3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
    x = Dropout(0.25)(x)

    # Top
    x = Flatten()(x)
    _i = 1
    for layer_size in last_layers_sizes:
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        # x = Dense(layer_size, activation='relu')(x)
        x = Dropout(0.5)(x)
        _i += 1
    x = Dense(nb_classes, activation='softmax', name=name_top + str(_i))(x)
    # x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=i, outputs=x)

    return model

def construct_model_8CONV(nb_classes, input_shape, last_layers_sizes=[256], name_top='fc'):
    """
        Create and returns a custom 8 convolutionnal layers model.

        :param nb_classes: Number of classes to predict.
        :param input_shape: Shape of the input data.
        :param last_layers_sizes: Size of the last dense layer at the top of the network.
        :param name_top: Prefix to put as name of the last dense layers.
        :type nb_classes: int
        :type input_shape: tuple
        :type last_layers_sizes: [int, ... ]
        :type name_top: string
        :return: The constructed keras model, uncompiled.
        :rtype: keras.model
    """

    i = Input(input_shape)
    
    # Block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.25)(x)

    # Block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.25)(x)

    # Block3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
    x = Dropout(0.25)(x)

    # Block4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='block4_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(x)
    x = Dropout(0.25)(x)

    # Top
    x = Flatten()(x)
    _i = 1
    for layer_size in last_layers_sizes:
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        # x = Dense(layer_size, activation='relu')(x)
        x = Dropout(0.5)(x)
        _i += 1
    x = Dense(nb_classes, activation='softmax', name=name_top + str(_i))(x)
    # x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=i, outputs=x)

    return model




def construct_model_FCN():
    """
        Construct and return a fully convolutionnal network.
    """

    

    return model

def construct_model_babbling_recomp_1(input_shape, nb_blocks, last_layers_sizes, name_top='fc'):
    """
        Construct and return a model for babbling recomposition.
        Version 1. Basically equivalent to 8Conv network.

        :param input_shape: Shape of the input data.
        :param last_layers_sizes: Size of the last dense layers at the top of the network.
        :param name_top: Prefix to put as name of the last dense layers.
        :return: The constructed keras model, uncompiled.
        :type input_shape: tuple
        :type last_layers_sizes: [int, ... ]
        :type name_top: string
        :rtype: keras.model
    """

    i = Input(input_shape)
    
    if nb_blocks > 0:
        # Block1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
        x = Conv2D(64, (3, 3), activation='relu', name='block1_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
        x = Dropout(0.25)(x)

    if nb_blocks > 1:
        # Block2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', name='block2_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
        x = Dropout(0.25)(x)

    if nb_blocks > 2:
        # Block3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)
        x = Dropout(0.25)(x)

    if nb_blocks > 3:
        # Block4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='block4_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(x)
        x = Dropout(0.25)(x)

    # Top
    x = Flatten()(x)
    _i = 1
    for layer_size in last_layers_sizes:
        x = Dense(layer_size, activation='relu', name=name_top + str(_i))(x)
        x = Dropout(0.25)(x)
        _i += 1
    x = Dense(1, activation='linear', name=name_top + str(_i))(x)

    model = Model(inputs=i, outputs=x)

    return model



#
# --------------------------- LOADING ---------------------------
#

def load_pretrained_model(model_path, reset_layer=0, reset_method='shuffle', remove_top=0):
    """
        Load and returns a model from a given save file.

        :param model_path: Absolute path to the saved model.
        :param reset_layer: Number of layers to reset, starting from top [default 0].
        :param reset_method: Method to use for reseting the weights ['shuffle', 'randomize'].
        :param remove_top: Remove the n top layers, do nothing if 0 (default).
        :type model_path: string
        :type reset_layer: int
        :type reset_method: string
        :type remove_top: int
        :return: The loaded keras model, not compiled.
        :rtype: Keras Model
    """

    model = load_model(model_path)

    # Remove top
    if (remove_top > 0):
        model = Model(inputs=model.input, outputs=model.layers[-remove_top-1].output)

    if reset_layer > 0:
        if reset_method == 'shuffle':
            shuffle_weights(model, reset_layer)
        elif reset_method == 'randomize':
            model = randomize_layers(nb_layers=reset_layer, old_model=model)
        else:
            raise ValueError('Reset method should be one of ["shuffle", "randomize"].')
    return model

def load_pretrained_weights(model, weights_path):
    """
        Load the weights saved in weights_paths in the given model.
        Loads the weights by name of layers.

        :param weights_path: Absolute path to the saved weights.
        :type weights_path: string
    """
    
    model.load_weights(weights_path, by_name=True)
