from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, rescale=1./255, validation_split = 0.1)
image = gen.flow_from_directory('/home/vamsi/Internships/Ljubljana/SOTA/data/squeezenet/COVID',target_size=(224,224),save_to_dir='/home/vamsi/Internships/Ljubljana/SOTA/data/squeezenet/resized',class_mode='binary',save_prefix='N',save_format='jpeg',batch_size=32)

for i in range(42):
	image.next()