#!/usr/bin/env python

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

if __name__ == '__main__':

	K.set_image_dim_ordering('tf')

	img_shape=(40,40,3)
	lbl_shape=(1,)
	lbl_types=[('d1', float)]

	model = dl_models.construct_model_babbling_recomp_1(input_shape=img_shape, nb_blocks=3, last_layers_sizes=[64])

	optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=False)
	# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

	model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

	datagen = Image.ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True)

	x, y = data_generators.read_numpy_data_file(directory='/home/luce_vayrac/kinect_datasets/pr2_3cars_background2/build', filename='images_labels.txt', types=[float], fnames_format='relative')

	x, y = data_generators.generate_numpy_arrays(files=x, labels=y, image_shape=img_shape, label_shape=lbl_shape, label_types=lbl_types, data_format=None, grayscale=False)

	print(y[:,0])
	print(np.average(y[:,0], axis=None))
	# print(y.median())

	# x_train = x[:10000, :]
	# x_val = x[10000:, :]
	# y_train = y[:10000, :]
	# y_val = y[10000:, :]



	# model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
	# 	steps_per_epoch=len(x) / 32,
	# 	# steps_per_epoch=100,
	# 	epochs=10)

	# score = model.evaluate(x_val, y_val, batch_size=32)
	# print(score)