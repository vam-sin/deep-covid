# Paper 12
# Predicting two classes only (Normal v COVID-19)

from numpy.random import seed
import math
seed(42)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(42)
# libraries
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.models import Model 
from keras.layers import Dense, Input, Dropout, Flatten
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf 

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# data
bs = 1
gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1./255)

train_data = gen.flow_from_directory('../data/squeezenet/train', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', classes=['COVID-19', 'Normal', 'Pneumonia'])
test_data = gen.flow_from_directory('../data/squeezenet/test', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', shuffle=False, classes=['COVID-19', 'Normal', 'Pneumonia'])

# for i in range(len(train_data)):
# 	X, y = next(train_data)
# 	print(y)

print(train_data.class_indices)
# model
num_classes = 3
densenet201 = keras.applications.DenseNet201(include_top = False, weights = "imagenet")

_input = Input(shape=(224, 224, 3), name = 'image_input')
#Use the generated model 
doutput = densenet201(_input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(doutput)
x = Dense(1024, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(inputs=_input, outputs=x)

# sgd 
opt = keras.optimizers.SGD(learning_rate = 1e-3)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('densenet201_c3.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

weights = {0: 1., 1: 1.}
# with tf.device('/cpu:0'):
# 	model.fit_generator(train_data, steps_per_epoch = len(train_data), epochs = 10, validation_data = test_data, validation_steps = len(test_data), callbacks = callbacks_list)

# Evaluate
with tf.device('/cpu:0'):
	print("Evaluating")
	model = keras.models.load_model('densenet201_c3.h5')
	y_pred = model.predict_generator(test_data, math.ceil(len(test_data) // bs), workers = 1, pickle_safe = True, verbose = 1)
	y_pred = np.argmax(y_pred, axis=1)
	print(len(y_pred))
	y_true = test_data.classes

	print('Confusion Matrix')
	cm = confusion_matrix(y_true, y_pred)
	print(cm)

	# accuracy, precision, recall, f1 score, sensitivity, specificity, auc
	print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

	print('\nClassification Report\n')
	print(classification_report(y_true, y_pred, target_names=['COVID-19', 'Normal', 'Pneumonia'], output_dict = True)) 

	sens = (cm[0][0] + cm[1][1] + cm[2][2])/(cm[0][0] + cm[1][1] + cm[2][2] + cm[1][0] + cm[2][0] + cm[0][1] + cm[2][1] + cm[0][2] + cm[1][2])
	spec = (cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])/(cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] + cm[1][0] + cm[2][0] + cm[0][1] + cm[2][1] + cm[0][2] + cm[1][2] )
	prec = (cm[0][0] + cm[1][1] + cm[2][2])/(cm[0][0] + cm[1][1] + cm[2][2] + cm[1][0] + cm[2][0] + cm[0][1] + cm[2][1] + cm[0][2] + cm[1][2])

	print("Per class Sensitivity is the same as Per class Recall")

	# Sensitivity
	tpc = cm[0][0]
	tpn = cm[1][1]
	tpp = cm[2][2]

	fpc = cm[1][0] + cm[2][0]
	fpn = cm[0][1] + cm[2][1]
	fpp = cm[0][2] + cm[1][2]

	fnc = cm[0][1] + cm[0][2]
	fnn = cm[1][0] + cm[1][2]
	fnp = cm[2][0] + cm[2][1]

	tnc = np.sum(cm) - tpc - fpc - fnc
	tnn = np.sum(cm) - tpn - fpn - fnn
	tnp = np.sum(cm) - tpp - fpp - fnp

	print("\nSensitivity\n")
	csens = tpc/(tpc + fnc)
	nsens = tpn/(tpn + fnn)
	psens = tpp/(tpp + fnp)
	weighted_sens = (csens + nsens + psens)/3

	print("COVID-19 Sensitivity: ", csens)
	print("Normal Sensitivity: ", nsens)
	print("Pneumonia Sensitivity: ", psens)
	print("Weighted Sensitivity: ", weighted_sens)

	# Specificity
	print("\nSpecificity\n")
	cspec = tnc/(tnc + fpc)
	nspec = tnn/(tnn + fpn)
	pspec = tnp/(tnp + fpp)
	w_spec = (cspec + nspec + psens)/3

	print("COVID-19 Specificity: ", cspec)
	print("Normal Specificity: ", nspec)
	print("Pneumonia Specificity: ", pspec)
	print("Weighted Specificity: ", w_spec)

'''
Results: 
loss: 0.0897 - accuracy: 0.9732 - val_loss: 1.1692e-04 - val_accuracy: 0.9962
'''
