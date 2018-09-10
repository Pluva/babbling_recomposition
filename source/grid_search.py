"""
    Grid search functions to be called from the main.py.


"""

#python
import os

#numpy
import numpy as np

#keras
from keras.preprocessing import image as Image
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#matplotlib
import matplotlib.pyplot as plt

#opencv
import cv2

#local
from models import dl_models
from models.tools import data_generators, dl_utilities, heatmaps_tools
import expert_classifier
from tools import image_tools


def test_trainModel():
    # model = dl_models.keras_model_VGG16(nb_classes=2, input_shape=(128,128,3),last_layers_sizes=[128,32])
    # data_path = '/data/kinect_datasets/general_objs/DL_Ready/rollable_data_source_2c'
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # train_generator = train_datagen.flow_from_directory(
    #     data_path,
    #     target_size=(128,128),
    #     batch_size=32)

    # # ----- Validation dataset
    # test_datagen = ImageDataGenerator(
    #     rescale=1./255)
    # validation_generator = test_datagen.flow_from_directory(
    #     data_path,
    #     target_size=(128,128),
    #     batch_size=32)
    # for layer in model.layers[:len(model.layers)-6]:
    #     layer.trainable = False
    # print(model.summary())

    # optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=False, decay=0.0001)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(generator=train_generator,
    #     steps_per_epoch=train_generator.samples / 32,
    #     epochs=15,
    #     verbose=1,
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.samples / 32)

    # model.save('/data/kinect_datasets/Videos/saved_model1')
    return

def test():

    image_folder = '/data/kinect_datasets/Videos/'
    save_folder = '/data/kinect_datasets/Videos/'
    save_filename = 'pred_tableset1.mp4'
    video_path = '/data/kinect_datasets/Videos/tableset1.mp4'
    background = '/data/kinect_datasets/Videos/tableset1_back.jpg'

    # /data/kinect_datasets/Videos/IMG_20180904_172127.jpg

    image_background = cv2.imread(background, 0)
    image_background = cv2.resize(image_background, (1920, 1080))

    model = dl_models.load_pretrained_model('/data/kinect_datasets/Videos/saved_model1')

    expert_classifier.predict_on_video_stream(video_path=video_path, background=image_background,
        model=model, sv_path=save_folder, sv_filename=save_filename, fps_out=15, verbose=1)
    # plt.imsave('{}tryingMasks.jpg'.format(save_folder), image1_color)
    # heatmaps_tools.heatmaps_evaluation_video(fcn_model=None, video_path='/data/kinect_datasets/Videos/vid_desktop1.mp4', verbose=1)

def test2():
    model_path = '/data/python_ws/saved_models/dl_vgg16/DL_PT_flowers_AR_MDMS_VGG16_d32_224x224.h5'
    video_path = '/data/kinect_datasets/Videos/tableset1.mp4'
    sv_path = '/data/kinect_datasets/Videos/'
    sv_filename = 'hms_tableset1.mp4'

    model = dl_models.load_pretrained_model('/data/kinect_datasets/Videos/saved_model1')
    print(model.summary())
    fcn_model = dl_models.convert_to_fcn(model)
    print(fcn_model.summary())

    heatmaps_tools.heatmaps_evaluation_video(fcn_model, 0, video_path, sv_path, sv_filename,
        fps_out=10, verbose=1)
    


def test3():
    """
        Testing difference in fcn conversion.
    """
    img_shape=(45,45,3)
    lbl_shape=(1,)
    lbl_type=float

    nb_blocks = 3; last_layers_sizes = [128, 64]

    tresholds = [[0.7]]
    ratios = [3,2]; ratios_valid=[0,1]; eq_ratios = True
    nb_classes = len(ratios)
    # nb_classes = 1

    # Create model
    model = dl_models.construct_model_babbling_recomp_1(input_shape=img_shape, nb_classes=nb_classes,
        nb_blocks=nb_blocks, last_layers_sizes=last_layers_sizes)
    print(model.summary())
    fcn_model = dl_models.convert_to_fcn(model)
    print(fcn_model.summary())
    return


def _train(compiled_model, data_generator, train_data, valid_data, verbose=0):
    """
        Train a single model, returns the training history.
    """

    img_shape=(45,45,3)
    lbl_shape=(1,)
    lbl_type=float

    nb_blocks = 3; last_layers_sizes = [128, 64]

    tresholds = [[0.7]]
    ratios = [3,2]; ratios_valid=[0,1]; eq_ratios = True
    nb_classes = len(ratios)
    # nb_classes = 1

    # Create model
    model = dl_models.construct_model_babbling_recomp_1(input_shape=img_shape, nb_classes=nb_classes,
        nb_blocks=nb_blocks, last_layers_sizes=last_layers_sizes)
    # model = dl_models.construct_model_8CONV(nb_classes=nb_classes, input_shape=img_shape, last_layers_sizes=[64])

    datagen = Image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    # predefined datasets locations, to remove after testing
    data_folder_1 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_background/build'
    data_folder_docker_1 = '/data/kinect_datasets/pr2_3cars_background/build'
    data_folder_2 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_background2/build'
    data_folder_docker_2 = '/data/kinect_datasets/pr2_3cars_background2/build'
    data_folder_3 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_nobackground/build'
    data_folder_docker_3 = '/data/kinect_datasets/pr2_3cars_nobackground/build'

    # Load all datasets.
    xf1, yf1 = data_generators.read_numpy_data_file(
        directory=data_folder_docker_1, filename='images_labels.txt', types=[float], fnames_format='relative')
    xf2, yf2 = data_generators.read_numpy_data_file(
        directory=data_folder_docker_2, filename='images_labels.txt', types=[float], fnames_format='relative')
    xf3, yf3 = data_generators.read_numpy_data_file(
        directory=data_folder_docker_3, filename='images_labels.txt', types=[float], fnames_format='relative')

    x = xf3; y = yf3
    # x = xf1 + xf2 + xf3; y = yf1 + yf2 + yf3

    # Generate numpy array from loaded datasets filenames and labels.
    x, y = data_generators.generate_numpy_arrays(files=x, labels=y,
        image_shape=img_shape, label_shape=lbl_shape, label_type=lbl_type, data_format=None, grayscale=False)
    # Equalize the datasets regarding certain ratios per tresholds.
    x, y = data_generators.equalize_dataset_continuous_1D(images=x, labels=y,
        tresholds=tresholds, dimension=0, ratios=ratios, eq_ratios=eq_ratios, verbose=1)
    # Shuffle datasets
    x, y = data_generators.shuffle_dataset(x, y)

    # Discretize labels
    if tresholds != None:
        y = data_generators.discretize_labels_1D(y, tresholds=tresholds[0], verbose=1)

    # Split dataset
    split_ratio = 0.7
    if split_ratio >= 1:
        print('Not splitting dataset !')
        x_train = x_val = x; y_train = y_val = y
    else:
        n = int(len(x) * split_ratio)
        x_train = x[:n, :]
        x_val = x[n:, :]
        y_train = y[:n, :]
        y_val = y[n:, :]


    # Training, several steps
    # optimizer = SGD(lr=0.001, momentum=0.9, nesterov=False, decay=0.0001)
    # # optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # ## continuous
    # # model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    # ## discret
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
    #   steps_per_epoch=len(x_train) / 32,
    #   # steps_per_epoch=100,
    #   epochs=20,
    #   validation_data=(x_val, y_val))

    optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=False, decay=0.0001)
    # optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.001)
    ## continuous
    # model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    ## discret
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train) / 32,
        # steps_per_epoch=100,
        epochs=50,
        validation_data=(x_val, y_val))

    # optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=False, decay=0.0001)
    # # optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
    #   steps_per_epoch=len(x_train) / 32,
    #   # steps_per_epoch=100,
    #   epochs=10,
    #   validation_data=(x_val, y_val))

    # optimizer = SGD(lr=0.000001, momentum=0.9, nesterov=False, decay=0.0001)
    # # optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
    #   steps_per_epoch=len(x_train) / 32,
    #   # steps_per_epoch=100,
    #   epochs=10,
    #   validation_data=(x_val, y_val))



    # ---- Heatmap vizualisation
    fcn_model = dl_models.convert_to_fcn(model)
    image_folder = '/data/kinect_datasets/pr2_3cars_nobackground/'
    sizes = '';
    for s in last_layers_sizes:
        sizes += str(s)
    save_folder = '/data/kinect_datasets/pr2_3cars_nobackground/hm_{}_{}/'.format(nb_blocks, sizes)
    images, fnames = data_generators.load_images_from_folder(image_folder, pattern='*_color.jpg', rm_extension=True, verbose=0)

    heatmaps_tools.heatmaps_evaluation(fcn_model=fcn_model, images=images, sv_path=save_folder, sv_names=fnames)

    return history




def grid_search(verbose=0):
    """
        Do grid search on a given set of parameters.

        :param verbose: Verbose behavior.
        :type verbose: int
    """


    return