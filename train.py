#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.layers import Input, Conv2D
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam

from data_generator import DataGenerator, retrieve_ids
import utils


# settings for the training
img_suffix = '.tiff'
mask_suffix = '_segmentation.png'
im_size = (320, 320)
batch_size = 1
n_classes = 2
n_channels = 5
shuffle = True

nb_epochs = 100
training_ratio = 0.8
learning_rate = 2e-4
optimizer = Adam(lr=learning_rate)
loss = bce_jaccard_loss  # loss = binary_crossentropy

params = {'img_suffix': img_suffix,
          'mask_suffix': mask_suffix,
          'dim': im_size,  # with the module segmentation_models, dim must be divisible by factor 32
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': shuffle
          }
infos = {'loss': loss,
         'optimizer': optimizer,
         'learning_rate': learning_rate,
         'training_ratio': training_ratio,
         'batch_size': batch_size,
         'nb_epochs': nb_epochs
         }

# data sets and generators
train_data_path = './ISIC2018_data/train/'
test_data_path = './ISIC2018_data/test/'

IDs_train = retrieve_ids(train_data_path, params['img_suffix'], params['mask_suffix'])
IDs_test = retrieve_ids(test_data_path, params['img_suffix'], params['mask_suffix'])

n_train, n_test = len(IDs_train), len(IDs_test)
partition = {'train': IDs_train,
             'validation': IDs_test
             }

training_generator = DataGenerator(path=train_data_path, list_IDs=partition['train'], **params)
validation_generator = DataGenerator(path=test_data_path, list_IDs=partition['validation'], **params)

print('Size training set: {}'.format(len(training_generator)))
print('Size test set: {}'.format(len(validation_generator)))

# design model
base_model = Unet()
inp = Input(shape=(*params['dim'], params['n_channels']))
l1 = Conv2D(3, (1, 1))(inp)  # map n_channels channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)
model.compile(optimizer=optimizer, loss=loss, metrics=[iou_score])

# train model on dataset
steps_per_epoch = (n_train * batch_size) // nb_epochs
validation_steps = 50

print('\nInformations\n')
print(infos)
history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=nb_epochs-1,
                              validation_data=validation_generator,
                              validation_steps=validation_steps
                              )

# for the last epoch, we compute precision on the full validation set
t1 = time.time()
history2 = model.fit_generator(generator=training_generator,
                               steps_per_epoch=steps_per_epoch,
                               epochs=1,
                               validation_data=validation_generator,
                               validation_steps=n_test  # we test on the full dataset
                               )
t2 = time.time()
print('Last epoch computer in {} seconds.'.format(int(t2-t1)))

# save the network
networks_name = 'segmentation_models'
utils.save_network(networks_name,
                   model,
                   history,
                   infos
                   )

# test
some_tests = partition['validation'][np.random.permutation(n_test)[:9]]

im_names1 = some_tests[:3]
im_names2 = some_tests[3:6]
im_names3 = some_tests[6:9]
images_lists = [im_names1, im_names2, im_names3]

for im_names in images_lists:
    predictions, bin_predictions = utils.display_some_results(model=model,
                                                              im_names=im_names,
                                                              threshold=0.3,
                                                              dim=im_size)

# save network
utils.save_network(networks_name,
                   model,
                   history2,
                   infos
                   )
