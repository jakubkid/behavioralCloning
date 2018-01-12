import numpy as np
import csv
import cv2
import tensorflow as tf

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
flags.DEFINE_string('dataPath', '', "Patch to driving_log.csv and IMG folder")
flags.DEFINE_string('epochs', '10', "Number of training epochs ")
print(FLAGS.dataPath)
lines = []
with open(FLAGS.dataPath+'\\driving_log.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
 
images = []
measurements = []
for line in lines:
    sourcePath = line[0]
    filename = sourcePath.split('\\')[-1]
    image = cv2.imread(FLAGS.dataPath+'\\IMG\\'+filename)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    #flip the images and measurements to generate more training data and teach car to turn right
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 10, 10))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.9))
model.add(Activation('relu'))
model.add(Convolution2D(16, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))
'''
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Flatten())
model.add(Dense(1))
'''


model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train,y_train,nb_epoch=int(FLAGS.epochs), validation_split=0.2, shuffle=True)

model.save('model.h5')

