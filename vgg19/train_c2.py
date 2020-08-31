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

train_data = gen.flow_from_directory('../data/data/train', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', classes=['COVID-19', 'Normal'])
test_data = gen.flow_from_directory('../data/data/test', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', shuffle=False, classes=['COVID-19', 'Normal'])

# for i in range(len(train_data)):
# 	X, y = next(train_data)
# 	print(y)

print(train_data.class_indices)
# model
num_classes = 2
vgg19 = keras.applications.VGG19(include_top = False, weights = "imagenet")

_input = Input(shape=(224, 224, 3), name = 'image_input')
#Use the generated model 
doutput = vgg19(_input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(doutput)
x = Dense(1024, activation='relu', name='fc1')(x)
# x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(inputs=_input, outputs=x)
print(model.summary())

# sgd 
opt = keras.optimizers.Adam(learning_rate = 1e-5)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('vgg19_c2.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

weights = {0: 1., 1: 1.}
# with tf.device('/cpu:0'):
# 	model.fit_generator(train_data, steps_per_epoch = len(train_data), epochs = 100, validation_data = test_data, validation_steps = len(test_data), callbacks = callbacks_list)

# Evaluate
# with tf.device('/cpu:0'):
print("Evaluating")
model = keras.models.load_model('vgg19_c2.h5')
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

'''
Results:
loss: 0.0113 - accuracy: 0.9971 - val_loss: 2.1458e-05 - val_accuracy: 1.0000
Confusion Matrix:  [[95  5]
 [ 5 95]]
Accuracy:  0.95
Sensitivity:  0.95
Specificity:  0.95
Precision:  0.95
Recall:  0.95
F1 Score:  0.9500000000000001
AUC Score:  0.95
'''
