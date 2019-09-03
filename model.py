#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% IMPORTS

import numpy as np
import os

from skimage.io import imread
from skimage.io import imsave

from data_generator import DataGenerator

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Input, Conv2D
from keras.models import Model

import time
import inspect


import useful_functions


#%% SETTINGS FOR THE TRAINING

im_size = (320, 320)
batch_size = 1
nb_epochs = 100

training_data_ratio= 0.9
optimizer_name = "Adam" # please choose among ["Adam"]

learning_rate = 0.0002

loss_name = "Jaccard" # please choose among ["Jaccard", "binary_crossentropy"]


#%% prepare optimizer, loss and infos dictionnary
if optimizer_name == "Adam":
    optimizer = Adam(lr=learning_rate)

if loss_name == "Jaccard":
    loss = bce_jaccard_loss
elif optimizer_name == "binary_crossentropy":
    loss = binary_crossentropy

infos = {"loss":loss_name,
         "optimizer":optimizer_name,
         "learning_rate":learning_rate,
         "training_ratio":training_data_ratio,
         "batch_size":batch_size,
         "nb_epochs":nb_epochs
        }

#%% set parameters and prepare datasets
params = {'img_suffix': '.tiff',
          'mask_suffix': '_segmentation.png',
          'dim': im_size,  # with the module segmentation_models, dim must be divisible by factor 32
          'batch_size': batch_size,
          'n_classes': 2,
          'n_channels': 5,
          'shuffle': True}

# Data sets
train_data_path = '../ISIC2018_data/train/'
test_data_path = '../ISIC2018_data/test/'


IDs_train = os.listdir(train_data_path)
IDs_test = os.listdir(test_data_path)

for k_train in range(len(IDs_train)):
    IDs_train[k_train] = IDs_train[k_train].replace(params['img_suffix'], '')
    IDs_train[k_train] = IDs_train[k_train].replace(params['mask_suffix'], '')
for k_test in range(len(IDs_test)):
    IDs_test[k_test] = IDs_test[k_test].replace(params['img_suffix'], '')
    IDs_test[k_test] = IDs_test[k_test].replace(params['mask_suffix'], '')

IDs_train = np.unique(IDs_train)
IDs_test = np.unique(IDs_test)

n_train = len(IDs_train)
n_test = len(IDs_test)

partition = {'train': np.array(IDs_train),
             'validation': np.array(IDs_test)
             }


#%% Generators
training_generator   = DataGenerator(path=train_data_path, list_IDs=partition['train'], **params)
validation_generator = DataGenerator(path=test_data_path, list_IDs=partition['validation'], **params)

print("taille jeu d'entrainement :", len(training_generator))
print("taille jeu de test :", len(validation_generator))


#%% Design model
base_model = Unet()

inp = Input(shape=(*params['dim'], params['n_channels']))
l1 = Conv2D(3, (1, 1))(inp)  # map n_channels channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)

model.compile(optimizer=optimizer, loss=loss, metrics=[iou_score])


#%% Train model on dataset

steps_per_epoch = (n_train*batch_size)//nb_epochs
validation_steps = 50

print("\n ##### INFOS ##### \n")
print(infos)
print("\n ##### INFOS ##### \n")
history = model.fit_generator(generator        = training_generator,
                              steps_per_epoch  = steps_per_epoch,
                              epochs           = nb_epochs - 1,
                              validation_data  = validation_generator,
                              validation_steps = validation_steps
                              )

# for the last epoch, we compute precision on the full validation set
print("last epoch :")
t1 = time.time()
history2 = model.fit_generator(generator        = training_generator,
                               steps_per_epoch  = steps_per_epoch,
                               epochs           = 1,
                               validation_data  = validation_generator,
                               validation_steps = n_test # we test on the full dataset
                               )
t2 = time.time()
print("(computed in ", int(t2-t1), " seconds.)")

#%% SAVE THE NETWORK

networks_name = inspect.getfile(inspect.currentframe())
networks_name = networks_name.split("/")[-1] # remove path (keep only the name of the file)
networks_name = networks_name.split(".py")[0] # remove .py

networks_name = "model_other_params"
useful_functions.save_network(networks_name, 
                              model, 
                              history, 
                              infos)


#%% test

some_tests = partition['validation'][np.random.permutation(n_test)[:9]]

im_names1 = some_tests[:3]
im_names2 = some_tests[3:6]
im_names3 = some_tests[6:9]

images_lists = [im_names1, im_names2, im_names3]


for im_names in images_lists :
    predictions, bin_predictions = useful_functions.display_some_results(model     = model,
                                                                        im_names  = im_names,
                                                                        threshold = 0.3,
                                                                        dim       = im_size)

#%% save network
useful_functions.save_network(networks_name, 
                              model, 
                              history2, 
                              infos)


#%% PREDICT AND SAVE PREDICTIONS
nb_images_to_predict = n_test
images_to_predict = partition['validation'][np.random.permutation(n_test)[:nb_images_to_predict]]

useful_functions.display_some_results(model = model,
                                      im_names  = images_to_predict,
                                      threshold = 0.3,
                                      dim       = im_size,
                                      display   = False,
                                      save      = True)







