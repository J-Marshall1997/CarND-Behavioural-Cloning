import csv
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, Reshape
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

###########################################################################
samples = []
with open('mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


image_paths = []
measurements = []
for line in samples:
    ##Remove steering values == 0
    if float(line[3]) != 0:
        #Centre Image
        source_path = line[0]
        path, filename = os.path.split(source_path)
        current_path = 'C:\\Users\\James\\Documents\\GitHub\\CarND-Behavioral-Cloning-P3\\mydata\\IMG' + '\\' + filename
        current_path.strip()
        image_paths.append(current_path)
        measurements.append(float(line[3]))

        #Left Image
        source_path = line[1]
        path, filename = os.path.split(source_path)
        current_path = 'C:\\Users\\James\\Documents\\GitHub\\CarND-Behavioral-Cloning-P3\\mydata\\IMG' + '\\' + filename
        current_path.strip()
        image_paths.append(current_path)
        measurements.append(float(line[3]) + 0.25)
        #Right Image
        source_path = line[2]
        path, filename = os.path.split(source_path)
        current_path = 'C:\\Users\\James\\Documents\\GitHub\\CarND-Behavioral-Cloning-P3\\mydata\\IMG' + '\\' + filename
        current_path.strip()
        image_paths.append(current_path)
        measurements.append(float(line[3]) - 0.25)
############################################################################
num_bins = 25
avg_samples_per_bin = len(measurements)/num_bins

hist, bins, _ = plt.hist(measurements, num_bins)
#plt.show()
#############################################################################
keep_probs = []
##Set this as the target samples in bins greater than this amount
target = avg_samples_per_bin * 1.3
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(measurements)):
    for j in range(num_bins):
        if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
image_paths = np.delete(image_paths, remove_list, axis=0)
measurements = np.delete(measurements, remove_list)

# print histogram again to show more even distribution of steering angles
plt.hist(measurements, num_bins)
print('SAMPLES =', len(measurements))
#plt.show()
##########################################################################
def preprocess_image(img):
    #Apply a Gaussian Blur
    new_img = cv2.GaussianBlur(img, (3,3), 0)
    #Following NVidias advice, resize to 66x200x3
    #Crop first
    new_img = img[50:140,:,:]
    new_img = cv2.resize(new_img, (200, 66), interpolation = cv2.INTER_AREA)
    #Again following NVidias advice on the model being used, convert to YUV colour space
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img
#############################################################################
#Randomly distort the training images to help the model generalise better
def random_distort(img):
    #Convert to float to allow distortion
    new_img = img.astype(float)

    #Adjust brightness of image
    value = np.random.randint(-28, 28)
    #valid_mask will be false if the brightness adjustment would set the value out of the (0, 255) range
    #If false, the brightness will not be adjusted
    if value > 0:
        valid_mask = (new_img[:,:,0] + value) > 255
    else:
        valid_mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(valid_mask, 0, value)

    #Randomly shadow the image on a random rectangle in the image from top to bottom
    height ,width = new_img.shape[0:2]
    mid = np.random.randint(0, width)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:width, 0] *= factor

    #Randomly warp the horizon to generalise inclines
    height ,width = new_img.shape[0:2] #Redundant but helps for clarity
    horizon = 2*height / 5
    vertical = np.random.randint(-height/8, height/8)
    src = np.float32([[0, horizon], [width, horizon], [0, height], [width, height]])
    dst = np.float32([[0, horizon+vertical], [width, horizon+vertical], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    new_img = cv2.warpPerspective(new_img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

    #Remember to return the image as a uint8
    return new_img.astype(np.uint8)
###########################################################################
image_path_train, image_path_valid, measurements_train, measurements_valid = train_test_split(image_paths, measurements,
                                                                                                test_size=0.2)
############################################################################
def generator(image_paths, measurements, batch_size=32, valid_flag=False):

    image_paths, measurements = shuffle(image_paths, measurements)
    X_train = []
    y_train = []

    while 1:

        for i in range(len(measurements)):

            image = cv2.imread(image_paths[i])
            image = preprocess_image(image)
            measurement = measurements[i]

            if valid_flag == False:
                    image = random_distort(image)

            X_train.append(image)
            y_train.append(measurement)

            if len(X_train) == batch_size:
                yield (np.array(X_train), np.array(y_train))
                X_train = []
                y_train = []
                image_paths, measurements = shuffle(image_paths, measurements)

            #If the steering angle is significant (> 0.33), flip the image and add it to the training set
            if abs(measurement)  > 0.33:
                flipped_image = cv2.flip(image, 1)
                measurement *= -1
                X_train.append(flipped_image)
                y_train.append(measurement)
                if len(X_train) == batch_size:
                    yield (np.array(X_train), np.array(y_train))
                    X_train = []
                    y_train = []
                    image_paths, measurements = shuffle(image_paths, measurements)
#############################################################################
##Generators
train_generator = generator(image_path_train, measurements_train, batch_size=64, valid_flag=False)
validation_generator = generator(image_path_valid, measurements_valid, batch_size=64, valid_flag=True)
##############################################################################
model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape=(66,200,3)))
model.add(Convolution2D(3, kernel_size = 1, strides = 1, padding="same"))

##First set of convolutions
model.add(Convolution2D(32, kernel_size = 7, strides = 2, padding="same", kernel_regularizer=l2(0.001)))
model.add(ELU(alpha=0.1))
model.add(Convolution2D(32, kernel_size = 7, strides = 2, kernel_regularizer=l2(0.0001)))
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))

##Second set of convolutions
model.add(Convolution2D(64, kernel_size = 5, strides = 1, padding="same", kernel_regularizer=l2(0.001)))
model.add(ELU(alpha=0.1))
model.add(Convolution2D(64, kernel_size = 3, strides = 1, kernel_regularizer=l2(0.001)))
model.add(ELU(alpha=0.1))
model.add(Dropout(0.5))

##Flatten Layer
model.add(Flatten())

##Fully connected layers
model.add(Dense(128))
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(ELU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(ELU(alpha=0.1))

model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.0001))

##Checkpoints and early stopping
checkpoint = ModelCheckpoint(filepath="./models/best-{val_loss:.4f}.h5", monitor='val_loss', verbose=0)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2)

history_object = model.fit_generator(train_generator,
                    samples_per_epoch= len(measurements_train),
                    validation_data=validation_generator,
                    nb_val_samples=len(measurements_valid),
                    callbacks =[checkpoint, early_stopping], nb_epoch=15)

model.save('./models/model-{:.4f}.h5'.format(history_object.history['val_loss'][-1]))
