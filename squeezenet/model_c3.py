# Squeezenet Architecture (Papers 17)
# add dropout and regularization later on if needed

# Libraries
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras import regularizers
import keras
from keras.models import Model 
import keras.backend as K 

def fire(x, s, e):
	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	x = Conv2D(s, (1,1), activation = 'relu', padding = 'valid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
	x = Dropout(0.1)(x)
	l = Conv2D(e, (1,1), activation = 'relu', padding = 'valid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
	r = Conv2D(e, (3,3), activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)

	x = keras.layers.concatenate([l, r], axis = channel_axis)

	return x

def squeezenet(num_classes):
	inp = Input(shape = (224, 224, 3))

	conv1 = Conv2D(64, (3,3), strides = (2,2), activation = 'relu', padding = 'valid')(inp)
	maxpool1 = MaxPooling2D(pool_size = (3,3), strides = 2)(conv1)

	fire2 = fire(maxpool1, 16, 64)
	fire2 = Dropout(0.1)(fire2)
	fire3 = fire(fire2, 16, 64)
	fire3 = Dropout(0.1)(fire3)
	fire4 = fire(fire3, 32, 128)
	fire4 = Dropout(0.1)(fire4)

	maxpool4 = MaxPooling2D(pool_size = (3,3), strides = 2)(fire4)
	fire5 = fire(maxpool4, 32, 128)
	fire5 = Dropout(0.1)(fire5)	
	fire6 = fire(fire5, 48, 192)
	fire6 = Dropout(0.1)(fire6)
	fire7 = fire(fire6, 48, 192)
	fire7 = Dropout(0.1)(fire7)
	fire8 = fire(fire7, 64, 256)
	fire8 = Dropout(0.1)(fire8)

	maxpool8 = MaxPooling2D(pool_size = (3,3), strides = 2)(fire8)
	fire9 = fire(maxpool8, 64, 256)
	conv10 = Conv2D(num_classes, (1,1), activation = 'relu', padding = 'valid')(fire9)
	# # dr = Dropout(0.9)(conv10)

	avgpool = AveragePooling2D(pool_size = (13,13), strides = 1)(conv10)

	avgpool = Flatten()(conv10)
	avgpool = Dense(4096, activation = 'relu')(avgpool)
	avgpool = Dropout(0.1)(avgpool)
	avgpool = Dense(4096, activation = 'relu')(avgpool)
	avgpool = Dropout(0.1)(avgpool)
	out = Dense(num_classes, activation = 'softmax')(avgpool)

	model = Model(inputs = inp, outputs = out)
	print(model.summary())

	return model

if __name__ == '__main__':
	num_classes = 3
	model = squeezenet(num_classes)

