# Behavior Cloning 
# import libraries
import csv
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# read data
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)                
        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        image = cv2.imread(current_path)
        correction = 0.2
        if image == None:
            pass;
        else:
            if i == 0:
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)

            if i == 1:
                images.append(image)
                measurement_left = float(line[3]) + correction
                measurements.append(measurement_left)

            if i == 2:
                images.append(image)
                measurement_right = float(line[3]) - correction
                measurements.append(measurement_right)

# Data augmentation flip images
augmented_images, augmented_measurements = [], []
for images,measurements in zip(images,measurements):
    augmented_images.append(images)
    augmented_measurements.append(measurements)
    augmented_images.append(cv2.flip(images,1))
    augmented_measurements.append(measurements*-1.0)

print(len(augmented_images))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Neural Network Architecture 
model = Sequential()
# Normalize
model.add(Lambda(lambda x: (x/250.0)-0.5, input_shape=(160,320,3)))
# Cropping image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Convolutional Layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(BatchNormalization())
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
# Output Layer
model.add(Dense(1))

# Compile
model.compile(loss='mse', optimizer='adam')

# Training model
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)

### Printing the keys contained in the history object
print(history_object.history.keys())
np.savetxt("file_loss.csv",history_object.history['loss'], delimiter=",", fmt='%s')
np.savetxt("file_val_loss.csv",history_object.history['val_loss'], delimiter=",", fmt='%s')

# Save Model
model.save('model1.h5')
exit()
