Project4: Behavioral Cloning

STEP1: DATA
* Data was acquired by driving the car in training modsel, by keeping the car in the center of the lane.
* The files containg the images and the log file is read and stored.
* To increase the data, images from side view were considered with correction factor=0.2
* Further to avoid left_bias of the steering, image was flipped and change of factor -1 was made to steering_measurements 

STEP2: Pre-processing data
* Normalization to image was done by dividing image array by 250 and adjusting it to center by subtracting it by 0.5
* Cropping - Image was cropped to capture onlyu the road and not the background, this would reduce the load while training the data

STEP3: Model Architecture
I followed the architecture suggested in the videos of NVIDEA paper.
Firstly, I implemented the model proposed on paper and later to improve the model performance I added Dropout and Maxpooloing functions.

* Layer 1: Lambda layer with a lambda function to normalize data.
* Layer 2: Convolution layer using 24 of size 5*5 filter with ELU activation
* Layer 3: Convolution layer using 36 of size 5*5 filters with ELU activation
* Layer 4: Convolution layer using 48 of size 5*5 filters with activation, which gave an output of (5,37,48)
* Layer 5,6,7: Convolution layer using 64 of size 3*3 filters with ELU activation, which gave an output of (1,33,64)
***************************
The next layers were changed from the proposed paper to make it fit for this case
* Layer 8: Dropout function was later introduced at this layer 
* Layer 9: Flatten Layer 
* Layer 10: Dense Layer of vector transformation 100 nurons
* Layer 11: Dense Layer of vector transformation 50 neurons
* Layer 12: Dense Layer of vector transformation 10 neurons
* Layer 13: Dense Layer of vector transformation 1 neuron
* Layer 14: Compilation--> loss function: mean squared error; Optimizer: adam; validation split:0.2

STEP4: Training the model
* model was trained with epochs=10 --> resulted in oscillating validation_loss after an initial drop
* model trainied with reduced epochs = 5 to avoid overfitting --> resulting in similar overfitting
* A dropout fiunction with rate = 0.5 was created after the convolution layers
* In order to increase the data - I drove the car for 2 laps and edge lane recovery to give more generalized data to train 
* After these changes, model was trained for epochs =5 
* This resulted in appropriate result of decreasing validation_loss, but only a slight spike in the last epoch

STEP5: Testing the model
* Model was tested in autonomous mode for 1 lap
* Video file named "run1" is created with fps = 60 
* It is observed that the car is able to stay in the center of the road throughout the lap
