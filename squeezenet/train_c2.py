# don't have all the data as the author collected data from a local hospital and hasn't released that segment.
# Predicting two classes only

from numpy.random import seed
import math
seed(42)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(42)
# libraries
from keras.preprocessing.image import ImageDataGenerator
from model_c2 import squeezenet
import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
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

train_data = gen.flow_from_directory('../data/squeezenet/train', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', classes=['COVID-19', 'Normal'])
test_data = gen.flow_from_directory('../data/squeezenet/test', target_size = (224, 224), batch_size = bs, class_mode = 'categorical', shuffle=False, classes=['COVID-19', 'Normal'])

# for i in range(len(train_data)):
# 	X, y = next(train_data)
# 	print(y)

print(train_data.class_indices)
# model
num_classes = 2
model = squeezenet(num_classes)

opt = keras.optimizers.Adam(learning_rate = 1e-6)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('squeezenet_new_c2.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

weights = {0: 1., 1: 1.}
# model.fit_generator(train_data, steps_per_epoch = len(train_data), epochs = 30, validation_data = test_data, validation_steps = len(test_data), callbacks = callbacks_list)

# Evaluate
model = keras.models.load_model('squeezenet_new_c2.h5')
y_pred = model.predict_generator(test_data, math.ceil(len(test_data) // bs), workers = 1, pickle_safe = True)
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



'''Tasks
1. Set seed - Done
2. Weighted - Done

Best:
loss: 0.0934 - accuracy: 0.9699 - val_loss: 0.0121 - val_accuracy: 0.9925
Precision:  0.9629629629629629
Sensitivity:  0.9629629629629629
Specificity:  0.9923076923076923

Benchmark:
accuracy: 99.7%
precision: 99.7%
sensitivity: 99.7%
specificity: 99.55%
'''
