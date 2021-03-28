#navita
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

car_images=[]
steering_angles =[] 
lines=[]
X_train=[]
y_train=[]
file_path="data_local/"
file_names=["1","2","back2center1","back2center2","back2center3","back2center4","back2center5","reverse1","reverse2","reverse3"]

for file_name in file_names:
    folder_path = file_path+file_name+'/IMG/'
    with open(file_path+file_name+'/driving_log.csv') as csvfile:
        reader =csv.reader(csvfile)
        for line in reader:
            line.append(folder_path)
            lines.append(line) 
               
def pre_process_image(image):
    # Since cv2 reads the image in BGR format and the simulator will send the image in RGB format
    # Hence changing the image color space from BGR to RGB
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Cropping the image
    return colored_image
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Set our batch size
batch_size=128
def generator(lines, batch_size):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]   
            car_images=[]
            steering_angles =[] 
            for batch_sample in batch_samples:
                image_center_path = batch_sample[7]+batch_sample[0].split('/')[-1]
                image_left_path = batch_sample[7]+batch_sample[1].split('/')[-1]
                image_right_path = batch_sample[7]+batch_sample[2].split('/')[-1]

                image_center = cv2.imread(str(image_center_path))
                image_left = cv2.imread(str(image_left_path))
                image_right = cv2.imread(str(image_right_path))

                image_center=pre_process_image(image_center)
                image_left=pre_process_image(image_left)
                image_right=pre_process_image(image_right)                
                
                
                correction = 0.3 # this is a parameter to tune
                steering_center=float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
       
                car_images.append(image_center)
                car_images.append(image_left)
                car_images.append(image_right) 


                steering_angles.append(steering_center)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)
                
            augmented_images,augmented_angles=[],[]
            for image,angle in zip(car_images, steering_angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*(-1.0))

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

train_number = len(train_samples)
validation_number  = len(validation_samples)

print("Number of training examples =", train_number)
print("Number of validation examples =", validation_number)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

row, col, ch = 160,320,3  # camera format
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))#crop useless background
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Dropout(0.25))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Dropout(0.25))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Dropout(0.25))

model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(500))
model.add(Dropout(0.25))

model.add(Dense(100))
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary() 


model.compile(loss='mse', optimizer="adam")
history_object=model.fit_generator(train_generator,\
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=7,verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('lossimage.png')


#cd /home/workspace/CarND-Behavioral-Cloning-P3
#python model.py
#python drive.py model.h5
#python drive.py model.h5 run1
#python video.py run1
#file_names=["1","2","OnEdge","back2center1","back2center2","back2center3","reverse1","reverse2","reverse3"]
