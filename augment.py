from keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1./255, preprocessing_function=add_noise)
image = gen.flow_from_directory('/home/vamsi/Internships/Ljubljana/SOTA/data/data/train',target_size=(224,224),save_to_dir='/home/vamsi/Internships/Ljubljana/SOTA/data/data/aug_COVID-19',class_mode='binary',save_prefix='N',save_format='jpeg',batch_size=32)

for i in range(395):
	image.next()

####### COVID-19 #######
# Increase samples by a factor of 4 (Minaee)
# 12 times (Ucar)
# 1,152 total COVID-19 images. Only CXR, removed CT. Removed overlapping images from two of the datasets. 
# 100 of these were placed in the test set. 
# 1,052 training COVID images augmented to 12,596
####### Pneumonia ####### 
# 10,000 Pneumonia from NIH Chest X-Rays (Kaggle)
# 2,696 Pneumonia from Chest X-Ray Images (Pneumonia) on Kaggle.  
# Chest X-Ray Images (Pneumonia) on Kaggle.
####### Normal #######
# 1,583 Normal from Chest X-Ray Images (Pneumonia) on Kaggle.
# 4,999 Normal from NIH Chest X-Rays (Kaggle)
# 6,114 from CheXpert
