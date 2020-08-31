import keras
import math
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np

bs = 1
gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1./255)

test_data = gen.flow_from_directory('data/data/test', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', shuffle=False, classes=['COVID-19', 'Normal'])

with tf.device('/cpu:0'):
	print("Loading Models")

	squeezenet = keras.models.load_model('squeezenet/squeezenet_fulldata_c2.h5')
	vgg19 = keras.models.load_model('vgg19/vgg19_c2.h5')
	mobilenetv2 = keras.models.load_model('mobilenetv2/mobilenetv2_c2.h5')
	
	print("Evaluating")
	
	print("Predicting SqueezeNet")
	y_pred_squeezenet = squeezenet.predict_generator(test_data, math.ceil(len(test_data) // bs), workers = 1, pickle_safe = True, verbose = 1)
	print("Predicting VGG 19")
	y_pred_vgg19 = vgg19.predict_generator(test_data, math.ceil(len(test_data) // bs), workers = 1, pickle_safe = True, verbose = 1)
	
	y_pred = []
	for i in range(len(y_pred_vgg19)):
		pred1 = (y_pred_vgg19[i][0] + y_pred_squeezenet[i][0])/2
		pred2 = (y_pred_vgg19[i][1] + y_pred_squeezenet[i][1])/2
		lis = [pred1, pred2]
		y_pred.append(lis)

	y_pred = np.asarray(y_pred)
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

''' VGG19 and SqueezeNet (Average Ensemble)

Confusion Matrix:  [[96  4]
 [ 1 99]]
Accuracy:  0.975
Sensitivity:  0.99
Specificity:  0.96
Precision:  0.9611650485436893
Recall:  0.99
F1 Score:  0.9753694581280788
AUC Score:  0.975


'''
