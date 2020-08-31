import keras
import math
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np

bs = 1
gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1./255)

train_data = gen.flow_from_directory('data/data/train', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', classes=['COVID-19', 'Normal'])
test_data = gen.flow_from_directory('data/data/test', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', shuffle=False, classes=['COVID-19', 'Normal'])

with tf.device('/cpu:0'):
	print("Loading Models")
	X, y = next(test_data)

	squeezenet = keras.models.load_model('squeezenet/squeezenet_fulldata_c2.h5')
	vgg19 = keras.models.load_model('vgg19/vgg19_c2.h5')
	squeezenet_last = keras.models.Model(inputs = squeezenet.inputs, outputs = squeezenet.layers[-2].output)
	vgg19_last = keras.models.Model(inputs = vgg19.inputs, outputs = vgg19.layers[-2].output)

	inp = keras.layers.Input(shape = (224, 224, 3))

	x1 = squeezenet_last(inp)
	x2 = vgg19_last(inp)

	x1 = keras.layers.Dense(1024, activation = 'relu')(x1)

	x_concat = keras.layers.concatenate([x1, x2])
	x_concat = keras.layers.Dropout(0.7)(x_concat)

	out = keras.layers.Dense(2, activation = 'softmax')(x_concat)

	model = keras.models.Model(inputs = inp, outputs = out)

	print(model.summary())

	opt = keras.optimizers.Adam(learning_rate = 1e-6)
	model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

	mcp_save = keras.callbacks.callbacks.ModelCheckpoint('ensemble_c2.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
	reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	callbacks_list = [mcp_save, reduce_lr]

	weights = {0: 1., 1: 1.}
	model.fit_generator(train_data, steps_per_epoch = 1000, epochs = 100, validation_data = test_data, validation_steps = len(test_data), callbacks = callbacks_list)

	# Evaluate
	# with tf.device('/cpu:0'):
	print("Evaluating")
	model = keras.models.load_model('ensemble_c2.h5')
	y_pred = model.predict_generator(test_data, math.ceil(len(test_data) // bs), workers = 1, pickle_safe = True, verbose = 1)
	y_pred = np.argmax(y_pred, axis=1)
	print(len(y_pred))
	y_true = test_data.classes

	print('Confusion Matrix')
	cm = confusion_matrix(y_true, y_pred)
	print("Confusion Matrix: ", cm)

	tn, fp, fn, tp = cm.ravel()

	acc = (tp + tn)/np.sum(cm)

	sens = tp/(tp + fn)

	spec = tn/(fp + tn)

	prec = tp/(tp + fp)

	rec = tp/(tp + fn)

	f1 = (2*prec*rec)/(prec + rec) 

	auc = roc_auc_score(y_true, y_pred)

	print("Accuracy: ", acc)
	print("Sensitivity: ", sens)
	print("Specificity: ", spec)
	print("Precision: ", prec)
	print("Recall: ", rec)
	print("F1 Score: ", f1)
	print("AUC Score: ", auc)

''' VGG19 and SqueezeNet (Last Layer Features Ensemble)
ensemble_best_c2.h5
Confusion Matrix:  [[98  2]
 [ 1 99]]
Accuracy:  0.985
Sensitivity:  0.99
Specificity:  0.98
Precision:  0.9801980198019802
Recall:  0.99
F1 Score:  0.9850746268656716
AUC Score:  0.985
'''
