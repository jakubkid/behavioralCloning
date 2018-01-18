import numpy as np
import csv
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('dataPath', '', "Patch to driving_log.csv and IMG folder seperate more with ,")
flags.DEFINE_string('epochs', '10', "Number of training epochs ")
flags.DEFINE_string('offset', '0.15', "Steering angle offset for left (+) and right (-) ")
flags.DEFINE_string('output', 'drop', "Output name .h5 will be added")
dataPaths = FLAGS.dataPath.split(',')
images = []
measurements = []
print('convert!!!!')
for dataPath in dataPaths:
	print(dataPath)
	lines = []
	with open(dataPath+'\\driving_log.csv', mode='r') as infile:
			reader = csv.reader(infile)
			for line in reader:
				lines.append(line)

	for line in lines:
		for i in range(3):    
			sourcePath = line[i]
			filename = sourcePath.split('\\')[-1]
			bgr_image = cv2.imread(dataPath+'\\IMG\\'+filename)
			b,g,r = cv2.split(bgr_image)
			rgb_img = cv2.merge([r,g,b]) # switch it to rgb
			images.append(rgb_img)
		steering_center = float(line[3])
		# create adjusted steering measurements for the side camera images
		correction = float(FLAGS.offset)
		steering_left = steering_center + correction
		steering_right = steering_center - correction
		measurements.append(steering_center)
		measurements.append(steering_left) 
		measurements.append(steering_right)


images, measurements = shuffle(images, measurements) # shuffling list uses much less RAM then array, 8GB RAM was suffisient 
X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train,y_train,nb_epoch=int(FLAGS.epochs), validation_split=0.2, shuffle=True)

model.save(FLAGS.output + '.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()