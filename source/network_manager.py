"""
    Network Manager Class.
    Handle one Keras network to be trained during an experiment.
"""

#python
import random
import os

#keras
import keras.backend as K

#local
from keras_tools.data_generators import *
from models.dl_models import dl_models
from tools import global_tools

class NetworkManager:
    """
        Network Manager Class.
        Handle one Keras network to be trained during an experiment.
    """

    def __init__(self,
        model_type,
        input_shape,
        nb_classes,
        last_layers_sizes=None,
        name_top='fc',
        batch_size=32,
        data_augmentation=False,
        data_aug_params=None):
        """
            Initialisation

            :param model_type: Type of deep network to create.
            :param input_shape: Shape of the input data used to train the network.
            :param batch_size: Batch size to use during training.
            :param nb_classes: Number of expected classes, just for a matter of initialisation, as it should change dynamically.
            :param data_augmentation: Enable data augmentation during training.
            :param data_aug_params: Parameters for the data augmentation, see keras doc ofr details.
            :type model_type: int
            :type input_shape: tuple
            :type batch_size: int
            :type nb_classes: int
            :type data_augmentation: boolean
            :type data_aug_params: {'param_key':value, ...}
        """
        

        self._model_type = model_type
        self._img_rows, self._img_cols, self.img_channels = input_shape
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._nb_classes = nb_classes
        self._data_augmentation = data_augmentation

        if data_augmentation and (data_aug_params is None):
            # default initialisation
            self.set_data_aug_params()
        elif not data_augmentation:
            self.data_aug_params = {}

        # Initialize model
        self.initialize_model(model_type=model_type, input_shape=input_shape, nb_classes=nb_classes,
            last_layers_sizes=last_layers_sizes, name_top=name_top)
        






    # ------------------ </TOOLS> ------------------ #

    

    # ------------------ <TOOLS/> ------------------ #

    

    # ------------------ </INNER METHODS> ------------------ #

    def __generate_random_train_valid_from_classes(self, classes, ratio=0.7, shuffle=True):
        """
            Generate a train and validation sets from given labels and classes.
            Should be replaced to handle an initialisation file rahter than this hard linked version.

            :param ratio: Ratio of data to put in train (1-ratio in validation).
            :param shuffle: Shuffle the data.
            :type ratio: float
            :type shuffle: boolean
            :return: The train and validation dict.
            :rtype: ({('object_key', label), ...}, {('object_key', label), ...})
        """

        # inverse mapping
        inv_classes = global_tools.inverse_mapping(mapping=classes, mode=global_tools._INV_MAP_UKTL)
        
        # initialise
        train = {}
        inv_train = {}
        valid = {}
        inv_valid = {}

        # compute size train / valid
        max_per_label = {k: len(inv_classes[k]) for k in inv_classes.keys()}

        # build inv_train and inv_valid
        for label, ids in inv_classes.iteritems():
            if shuffle:
                random.shuffle(ids)
            inv_train[label] = ids[0 : int(round(max_per_label[label] * ratio))]
            inv_valid[label] = ids[int(round(max_per_label[label] * ratio)) : ]

        # reverse to get train and valid
        train = global_tools.inverse_mapping(inv_train, mode=global_tools._INV_MAP_LTUK)
        valid = global_tools.inverse_mapping(inv_valid, mode=global_tools._INV_MAP_LTUK)
        
        return train, valid


    def __load_generator_experiment(self,
        path, classes, labels,
        target_size, shuffle=True,
        batch_size=32, data_aug_params={}):
        """
            *** DEPRECATED ***
            The custom has strange behavior, don't use for now.
            Replaced by the __load_generator_experiment_build_temp function.


            Load and return the train dataset containing the data pointed in classes.

            :param path: Directory where to look for the data.
            :param classes: Dictionnary of the data to load with their respective labels.
            :param labels: List of the labels.
            :param target_size: Image size
            :param shuffle: Shuffle the data before returning.
            :type path: string
            :type classes: python.dict {'instance_path': label, ...}
            :type labels: list
            :type target_size: (int, int)
            :type shuffle: Boolean
            :return: The image generator
            :rtype: CustomDirectoryIterator
        """

        datagen = CustomImageDataGenerator(**data_aug_params)

        generator = datagen.custom_flow_from_directory(
            directory=path,
            classes=classes,
            labels=labels,
            shuffle=shuffle,
            target_size=target_size,
            batch_size=batch_size)

        return generator

    def __load_generator_experiment_build_temp(self,
        path, train_classes, valid_classes, labels,
        target_size, shuffle=True,
        batch_size=32, data_aug_params={}):
        """
            Slight modification of the existing function.
            The dataset is first split into two temporary folders, then DataGenerators are initialized.

        """
        
        path_to_temp = 'temp_split'
        path_to_split = os.path.dirname(path) + '/' + path_to_temp
        dataset_tools.splitting_dataset(path=path,
            train_classes=train_classes, valid_classes=valid_classes, labels=labels, path_to_temp=path_to_temp)

        train = ImageDataGenerator(**data_aug_params)

        train_gen = train.flow_from_directory(path_to_split+'/train',
            target_size=target_size,
            batch_size=batch_size)

        valid = ImageDataGenerator(rescale=1./255)

        valid_gen = valid.flow_from_directory(path_to_split+'/validation',
            target_size=target_size,
            batch_size=batch_size)

        return train_gen, valid_gen

    # ------------------ <INNER METHODS/> ------------------ #

    # ------------------ </METHODS> ------------------ #

    def initialize_model(self, model_type, nb_classes, input_shape, last_layers_sizes, name_top):
        """
            Initialize the inner model.

            :param model_type: Type of model to create and return.
            :type model_type: string
            :return: The initialised model.
            :rtype: Keras.models
        """

        # if model_type == dl_models.DL_MODEL_VGG16:
        #     self.model = dl_models.construct_model_VGG16(nb_classes=nb_classes, input_shape=input_shape,
        #         last_layers_sizes=last_layers_sizes, name_top=name_top)
        # elif model_type == dl_models.DL_MODEL_CIFAR10:
        #     self.model = dl_models.construct_model_CIFAR10(nb_classes=nb_classes, input_shape=input_shape,
        #         last_layers_sizes=last_layers_sizes, name_top=name_top)
        # elif model_type == dl_models.DL_MODEL_VGG16_K:
        #     self.model = dl_models.keras_model_VGG16(nb_classes=nb_classes, input_shape=input_shape,
        #         last_layers_sizes=last_layers_sizes, name_top=name_top)
        # else:
        #     mdls = str(dl_models.DL_MODEL_NAMES)
        #     raise(ValueError('This model [' + str(model_type) + '] does not exist. Existing models are ' + mdls))

        self.model = dl_models.construct_model_by_id(model_type=model_type, nb_classes=nb_classes, input_shape=input_shape,
            last_layers_sizes=last_layers_sizes, name_top=name_top)

    def _load_weights(self, weights_path):
        """
            Load the weights into the current inner model, based on the name of the layers.

            :param weights_path:
            :type weights_path:
        """
        dl_models.load_pretrained_weights(model=self.model, weights_path=weights_path)

    def load_model(self, model_type, model_path, nb_classes, include_top, top_size=0):
        """
            Load a previously trained and or initialised model.
            NEEDS TO BE REWORKED

            :param mode_path: Path to the model.
            :type model_path: string
            :return: The loaded model.
            :rtype: Keras.models
        """

        if model_type == dl_models.DL_MODEL_VGG16:
            # self.model = construct_model_VGG16(nb_classes=nb_classes, input_shape=input_shape)
            self.model = None
        elif model_type == dl_models.DL_MODEL_CIFAR10:
            base_model = dl_models.load_model_CIFAR10(model_path=model_path, include_top=include_top, top_size=6)

            x = base_model.output
            x = Flatten(name='top_flat1')(x)
            x = Dense(32, name='top_d1')(x)
            x = Activation('relu', name='top_a1')(x)
            x = Dropout(0.5, name='top_drop1')(x)
            x = Dense(nb_classes, name="top_d2")(x)
            x = Activation('softmax', name='top_a2')(x)

            self.model = Model(inputs=base_model.input, outputs=x)

        else:
            mdls = str(dl_models.DL_MODEL_NAMES)
            raise(ValueError('This model [' + str(model_type) + '] does not exist. Existing models are ' + mdls))
        

    def compile_model(self, optimizer=None, loss='categorical_crossentropy', metrics=['accuracy']):
        """
            Compile the inner model with the given parameters.
            See Keras documentation for details about them.

            :param optimizer: Keras optimizer
            :param loss: Keras loss evaluation
            :param metrics: Performance evaluation
            :type optimizer: Keras.Optimizer
            :type loss: string
            :type metrics: ['param1', ... ]
        """

        # defaulting
        if optimizer is None:
            optimizer = SGD(lr=0.0001, decay=0.0001, momentum=0.9, nesterov=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def set_nb_classes(self, nb_classes):
        """
            Set the number of classes to predict and update the network accordingly.
            Assume the last layer is a dense layer.

            :param nb_classes: Number of classes.
            :type nb_classes: int
        """

        self._nb_classes = nb_classes
        # get last layer index and config
        ind = self.model.layers.__len__() - 1
        config = self.model.get_layer(index=ind).get_config()
        # update config
        config['units'] = nb_classes
        # remove last layer
        self.model.layers.pop()
        # add new updated layer
        output = self.model.layers[ind-1].output
        output = Dense.from_config(config)(output)
        self.model = Model(inputs=self.model.input, outputs=output)

    def set_data_aug(self, data_augmentation=True):
        """
            Set the data augmentation behavior.

            :param data_augmentation: True by default.
            :type data_augmentation: boolean
        """
        self._data_augmentation = data_augmentation

    def set_data_aug_params(self,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255):
        """
            Set the inner parameters used for data augmentation.
            See Keras documentation about parameters details [ImagePreprocessing]. 
        """

        self.data_aug_params = {
            'samplewise_center':samplewise_center,
            'featurewise_std_normalization':featurewise_std_normalization,
            'samplewise_std_normalization':samplewise_std_normalization,
            'zca_whitening':zca_whitening,
            'rotation_range':rotation_range,
            'width_shift_range':width_shift_range,
            'height_shift_range':height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'channel_shift_range':channel_shift_range,
            'fill_mode':fill_mode,
            'cval':cval,
            'horizontal_flip':horizontal_flip,
            'vertical_flip':vertical_flip,
            'rescale':rescale}

    def set_trainable_layers(self, nb_layers):
        """
            Set the number of layers to be trained, starting from the top.

            :param nb_layers: If -1 all layers will be set to trainable. Otherwise the nb_layers's top layers will be set as trainable.
            :type nb_layers: int
        """
        if nb_layers >= 0:
            layers_to_freeze = max(0, len(self.model.layers) - nb_layers)
            for layer in self.model.layers[:layers_to_freeze]:
                layer.trainable = False
        else:
            for layer in self.model.layers:
                layer.trainable = True

    def __train_model(self, generator, nb_epochs, verbose, callbacks=None, valid_generator=None):
        """
            Train the inner model with the given image generator.

            :param generator: Image generator for training.
            :param nb_epochs: Number of epochs, 1 by default.
            :param verbose: Verbose behaviour.
            :param callbacks:
            :param valid_generator:
            :param use_multiprocessing:
            :type generator: CustomDirectoryIterator
            :type nb_epochs: int
        """

        if not(valid_generator is None):
            valid_steps = valid_generator.samples / self._batch_size
        else:
            valid_steps = 0

        print(valid_steps)

        history = self.model.fit_generator(generator=generator,
            steps_per_epoch=generator.samples / self._batch_size,
            epochs=nb_epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=valid_generator,
            validation_steps=valid_steps)

        return history

    # def train_model(self, classes, labels, data_path, valid_classes=None, nb_epochs=1, verbose=0):
    #     """
    #         Builds the generator from the given classes and data path, then train the inner network.

    #         :param classes: Dictionnary of objects labels.
    #         :param data_path: Path where to find the data.
    #         :param valid_classes: Validation data
    #         :param nb_epochs: Number of epochs (default 1).
    #         :param verbose: Verbose behavior (0 none, 1 verbose)
    #         :type classes: {'obj_id': label, ... }
    #         :type data_path: string
    #         :type valid_classes: {'obj_id': label, ... }
    #         :type nb_epochs: int
    #         :type verbose: int
    #     """

    #     # load the train generator
    #     args = {'path':data_path, 'classes':classes, 'labels':labels, 'target_size':(self._img_rows, self._img_cols)}
    #     if self._data_augmentation:
    #         args['data_aug_params'] = self.data_aug_params
    #     train_generator = self.__load_generator_experiment(**args)

    #     # load the validation generator
    #     if not(valid_classes is None):
    #         valid_generator = self.__load_generator_experiment(path=data_path, classes=valid_classes,
    #             nb_classes=self._nb_classes, target_size=(self._img_rows, self._img_cols))

    #     # train model
    #     self.__train_model(generator=train_generator, nb_epochs=nb_epochs, verbose=verbose)

    def train_model_directory(self, train_classes, valid_classes, labels, data_path, nb_epochs=1, verbose=0):
        """
            Builds the generator from the given classes and data path, then train the inner network.
            Variant that uses the ImageGenerator from keras, as there seems to be some mishaps in the custom one.

            :param classes: Dictionnary of objects labels.
            :param data_path: Path where to find the data.
            :param valid_classes: Validation data
            :param nb_epochs: Number of epochs (default 1).
            :param verbose: Verbose behavior (0 none, 1 verbose)
            :type classes: {'obj_id': label, ... }
            :type data_path: string
            :type valid_classes: {'obj_id': label, ... }
            :type nb_epochs: int
            :type verbose: int
        """

        # Set data augmentation parameters
        args = {'path':data_path, 'train_classes':train_classes, 'valid_classes':valid_classes,
        'labels':labels, 'target_size':(self._img_rows, self._img_cols)}
        if self._data_augmentation:
            args['data_aug_params'] = self.data_aug_params

        (train_gen, valid_gen) = self.__load_generator_experiment_build_temp(**args)
        history = self.__train_model(generator=train_gen, valid_generator=valid_gen, nb_epochs=nb_epochs, verbose=verbose)


    def __evaluate_model(self, generator, verbose):
        """
            Evaluate the inner model on the given data generator.
        """
        ret = self.model.evaluate_generator(generator=generator, steps=generator.samples / self._batch_size)
        if verbose:
            print('Evaluation performance:')
            print(ret)

    def evaluate_model(self, classes, labels, data_path, verbose=0):
        """
            Evaluate the inner model on the given classes and dataset.

            :param classes: Dictionnary associating objects ids to labels.
            :param nb_classes: Number of classes.
            :param data_path: Path to the dataset.
            :type classes: {'object_id': label, ...}
            :type nb_classes: int
            :type data_path: string
            :return: The performance.
            :rtype: 
        """

        # load the validation generator
        validation_generator = self.__load_generator_experiment(path=data_path, classes=classes, labels=labels,
            target_size=(self._img_rows, self._img_cols))
        self.__evaluate_model(generator=validation_generator, verbose=verbose)

    # ------------------ </SIMULATION> ------------------ #

    def __load_dataset_simulation(self, ratio=0.7, shuffle=True):
        """
            Load a train_generator and validation_generator from a hard linked dataset and labels.
            Arguments kwargs are the one passed to the train_generator, see Keras documentation for details.

            :param path: Path where to find the object dataset.
            :param ratio: Percentage of data to use for training.
            :param shuffle: Shuffle the data before selecting training.
            :type path: string
            :type ratio: int
            :type shuffle: boolean
            :return: train and validation generators.
            :rtype: (CustomDirectoryIterator, CustomDirectoryIterator)
        """

        # Static objects label for simulation and testing purposes. To be removed.
        __obj_ids_by_labels = { 1: ['1', '4', '8', '9', '10', '11', '13', '14', '15', '16', '18', '20', '25', '26', '30', '32', '35'],
            2: ['2', '3', '5', '6', '7', '12', '17', '19', '21', '22', '23', '24', '27', '28', '29', '31', '33', '34', '36', '37']}
        __labels = [1, 2]
        # Set hardlink path
        path = '/home/eze/kinect_datasets/build/Data'
        nb_classes = 2
        # Create classes
        classes = global_tools.inverse_mapping(__obj_ids_by_labels, mode=global_tools._INV_MAP_LTUK)
        classes = global_tools.add_prefix_to_keys(classes, 'object')
        # Generate random datasets
        (train, valid) = self.__generate_random_train_valid_from_classes(classes=classes, ratio=ratio, shuffle=shuffle)
        # ----- Training dataset
        train_generator = self.__load_generator_experiment(path=path, classes=train, labels=__labels,
            target_size=(self._img_rows, self._img_cols), shuffle=shuffle, batch_size=32)
        # ----- Validation dataset
        validation_generator = self.__load_generator_experiment(path=path, classes=valid, labels=__labels,
            target_size=(self._img_rows, self._img_cols), shuffle=shuffle)

        return (train_generator, validation_generator)

    def _simulation_(self):
        train_generator, valid_generator = self.__load_dataset_simulation(ratio=0.7, shuffle=True)
        self.__train_model(generator=train_generator, nb_epochs=60, verbose=1)

    


    # ------------------ <SIMULATION/> ------------------ #




# ------------------ </TESTING> ------------------ #
# To be removed #

def network_manager_test_training():
    """
        Function to test the underlying functions.
    """
    nm = NetworkManager(dl_models.DL_MODEL_CIFAR10, input_shape=(32,32,3), nb_classes=2, data_augmentation=False)
    # nm.load_model(model_type=DL_MODEL_CIFAR10,
    #     model_path='/home/eze/python_ws/saved_models/dl_cifar10/dl_cifar10_100_sgd.h5', include_top=False, nb_classes=2)

    optimizer = SGD(lr=0.001, decay=0.0001, momentum=0.9, nesterov=True)
    nm.compile_model(optimizer=optimizer)
    nm._simulation_()
    

# ------------------ <TESTING/> ------------------ #
