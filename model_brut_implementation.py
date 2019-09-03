import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave

from data_generator import DataGenerator

from segmentation_models import Unet
from keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.layers import Input, Conv2D, concatenate, Dropout, MaxPooling2D, UpSampling2D
from keras.models import Model

#%%

# Parameters
params = {'img_suffix': '.tiff',
          'mask_suffix': '_segmentation.png',
          'dim': (320, 320),  # with the module segmentation_models, dim must be divisible by factor 32
          'batch_size': 2,
          'n_classes': 2,
          'n_channels': 5,
          'shuffle': True}

# Data sets
data_path = './ISIC2018_data320/'
IDs = os.listdir(data_path)
n = len(IDs)
for k in range(n):
    IDs[k] = IDs[k].replace(params['img_suffix'], '')
    IDs[k] = IDs[k].replace(params['mask_suffix'], '')

IDs = np.unique(IDs)

n = len(IDs)

indices = np.random.permutation(n)
train_idx, validation_idx = indices[:int(0.8*n)], indices[int(0.2*n):]

partition = {'train': np.array(IDs)[train_idx],
             'validation': np.array(IDs)[validation_idx]
             }

# Generators
training_generator = DataGenerator(path=data_path, list_IDs=partition['train'], **params)
validation_generator = DataGenerator(path=data_path, list_IDs=partition['validation'], **params)

#%%

# Design model
inp = Input(shape=(*params['dim'], params['n_channels']))
l1 = Conv2D(3, (1, 1))(inp)  # map n_channels channels data to 3 channels

conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(l1)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input=inp, output=conv10)

model.compile(optimizer=Adam(lr=0.0002), loss='binary_crossentropy', metrics=[iou_score])

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=len(training_generator)//training_generator.batch_size,
                              epochs=1,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator)//validation_generator.batch_size
                              )

model.save('unet_model4.h5')
model.save_weights('unet_weights4.h5')

# Get training and test loss histories
training_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#%%
#from keras.models import load_model

#model = load_model('unet_model3.h5')
#model.load_weights('unet_weights3.h5')
#preds = model.predict_generator(validation_generator)
