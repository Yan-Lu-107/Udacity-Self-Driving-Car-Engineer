import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


car_images=[]
steering_angles =[] 
lines=[]


file_path='/GitHub/Udacity-Self-Driving-Car-Engineer/CarND-Behavioral-Cloning-P3/data/'
file_names=["1","2","reverse1","reverse2","reverse3","back2center1","back2center2","back2center3"]

for file_name in file_names:
    lines=[]
    folder_path = file_path+file_name+'/IMG/'
    with open(file_path+file_name+'/driving_log.csv') as csvfile:
        reader =csv.reader(csvfile)
        for line in reader:
            line.append(folder_path)
            lines.append(line)  
 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]            
            for batch_sample in batch_samples:
                image_center_path = batch_sample[7]+batch_sample[0].split('/')[-1]
                image_left_path = batch_sample[7]+batch_sample[1].split('/')[-1]
                image_right_path = batch_sample[7]+batch_sample[2].split('/')[-1]

                image_center = cv2.imread(str(image_center_path))
                image_left = cv2.imread(str(image_left_path))
                image_right = cv2.imread(str(image_right_path))
                
                car_images.append(image_center)
                car_images.append(image_left)
                car_images.append(image_right)             
                
                correction = 0.02 # this is a parameter to tune
                steering_center=float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                    
                steering_angles.append(steering_center)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

ch, row, col = 3, 160, 320  # camera format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x:x/255.0 - 0.5,
            input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))#crop useless background
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#model.summary() 


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,\
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=4, verbose=1)
model.save('model.h5')
