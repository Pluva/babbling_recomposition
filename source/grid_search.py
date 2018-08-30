"""
	Grid search functions to be called from the main.py.


"""

#python


#numpy
import numpy as np

#keras
from keras.preprocessing import image as Image
from keras.optimizers import SGD, RMSprop
from keras import backend as K

#local
from models import dl_models
from models.tools import data_generators, dl_utilities

#matplotlib
import matplotlib.pyplot as plt

# def test():
# 	image_folder = '/data/kinect_datasets/pr2_3cars_background/'
# 	data_generators.load_images_from_folder(image_folder, pattern='*_color.jpg', verbose=1)



def _train(compiled_model, data_generator, train_data, valid_data, verbose=0):
	"""
		Train a single model, returns the training history.
	"""

	img_shape=(50,50,3)
	lbl_shape=(1,)
	lbl_type=float

	tresholds = [[0.7]]
	ratios = [1,1]; ratios_valid=[0,1]; eq_ratios = True
	nb_classes = len(ratios)
	# nb_classes = 1

	

	model = dl_models.construct_model_babbling_recomp_1(input_shape=img_shape, nb_classes=nb_classes, nb_blocks=3, last_layers_sizes=[64, 32])
	# model = dl_models.construct_model_8CONV(nb_classes=nb_classes, input_shape=img_shape, last_layers_sizes=[64])


	

	datagen = Image.ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True)

	data_folder_1 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_background/build'
	data_folder_docker_1 = '/data/kinect_datasets/pr2_3cars_background/build'
	data_folder_2 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_background2/build'
	data_folder_docker_2 = '/data/kinect_datasets/pr2_3cars_background2/build'
	data_folder_3 = '/home/luce_vayrac/kinect_datasets/pr2_3cars_nobackground/build'
	data_folder_docker_3 = '/data/kinect_datasets/pr2_3cars_nobackground/build'

	data_folder_train = data_folder_docker_2
	x, y = data_generators.read_numpy_data_file(
		directory=data_folder_train, filename='images_labels.txt', types=[float], fnames_format='relative')
	x, y = data_generators.generate_numpy_arrays(
		files=x, labels=y, image_shape=img_shape, label_shape=lbl_shape, label_type=lbl_type, data_format=None, grayscale=False)
	x, y = data_generators.equalize_dataset_continuous_1D(
		x, y, tresholds=tresholds, dimension=0, ratios=ratios, eq_ratios=eq_ratios, verbose=1)
	x, y = data_generators.shuffle_dataset(x, y)
	x_train = x; y_train = y

	# Use two different datasets
	data_folder_valid = data_folder_docker_2
	x_val, y_val = data_generators.read_numpy_data_file(
		directory=data_folder_valid, filename='images_labels.txt', types=[float], fnames_format='relative')
	x_val, y_val = data_generators.generate_numpy_arrays(
		files=x_val, labels=y_val, image_shape=img_shape, label_shape=lbl_shape, label_type=lbl_type, data_format=None, grayscale=False)
	x_val, y_val = data_generators.equalize_dataset_continuous_1D(
		x_val, y_val, tresholds=tresholds, dimension=0, ratios=ratios_valid, eq_ratios=eq_ratios, verbose=1)

	# Split single dataset
	# ratio = 0.7; n = int(len(x) * ratio)
	# x_train = x[:n, :]
	# x_val = x[n:, :]
	# y_train = y[:n, :]
	# y_val = y[n:, :]


	y_train = data_generators.discretize_labels(y_train, tresholds=tresholds[0], verbose=1)
	y_val = data_generators.discretize_labels(y_val, tresholds=tresholds[0], verbose=1)



	optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=False)
	# optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
	## continuous
	# model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
	## discret
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
		steps_per_epoch=len(x_train) / 32,
		# steps_per_epoch=100,
		epochs=50,
		validation_data=(x_val, y_val))


	optimizer = SGD(lr=0.00001, momentum=0.9, nesterov=False)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
		steps_per_epoch=len(x_train) / 32,
		# steps_per_epoch=100,
		epochs=50,
		validation_data=(x_val, y_val))


	# ---- Heatmap vizualisation
	fcn_model = dl_models.convert_to_fcn(model)

	image_folder = '/data/kinect_datasets/pr2_3cars_background2/'
	images, fnames = data_generators.load_images_from_folder(image_folder, pattern='*_color.jpg', verbose=0)

	k = 0
	for image in images:
		im = np.zeros((1, )+image.shape, dtype=K.floatx())
		im[0] = image
		out = fcn_model.predict(im)

		plt.imsave('{}hm_cl1_{}.png'.format(image_folder, fnames[k].split('.')[0]), out[0,:,:,0])
		plt.imsave('{}hm_cl2_{}.png'.format(image_folder, fnames[k].split('.')[0]), out[0,:,:,1])
		k += 1

	return history




def grid_search(verbose=0):
	"""
		Do grid search on a given set of parameters.

		:param verbose: Verbose behavior.
		:type verbose: int
	"""


	return