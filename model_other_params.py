#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import os
from skimage.io import imread

from data_generator import DataGenerator

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from keras.layers import Input, Conv2D
from keras.models import Model

import useful_functions


#%% SETTINGS

im_size = (320, 320)
batch_size = 1
nb_epochs = 100

training_data_ratio= 0.9
optimizer_name = "Adam" # please choose among ["Adam"]

learning_rate = 0.0002

loss_name = "Jaccard" # please choose among ["Jaccard", "binary_crossentropy"]



#%%
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

#%%
params = {'img_suffix': '.tiff',
          'mask_suffix': '_segmentation.png',
          'dim': im_size,  # with the module segmentation_models, dim must be divisible by factor 32
          'batch_size': batch_size,
          'n_classes': 2,
          'n_channels': 5,
          'shuffle': True}

# Data sets
data_path = '../ISIC2018_data/'
IDs = os.listdir(data_path)
n = len(IDs)

for k in range(n):
    IDs[k] = IDs[k].replace(params['img_suffix'], '')
    IDs[k] = IDs[k].replace(params['mask_suffix'], '')

IDs = np.unique(IDs)
n = len(IDs)

indices = np.random.permutation(n)
train_idx, validation_idx = indices[:int(training_data_ratio*n)], indices[int(training_data_ratio*n):]

partition = {'train': np.array(IDs)[train_idx],
             'validation': np.array(IDs)[validation_idx]
             }


#%% Generators
training_generator   = DataGenerator(path=data_path, list_IDs=partition['train'], **params)
validation_generator = DataGenerator(path=data_path, list_IDs=partition['validation'], **params)


#%% Design model
base_model = Unet()

inp = Input(shape=(*params['dim'], params['n_channels']))
l1 = Conv2D(3, (1, 1))(inp)  # map n_channels channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)

model.compile(optimizer=optimizer, loss=loss, metrics=[iou_score])


#%% Train model on dataset
steps_per_epoch = (len(train_idx)*batch_size)//nb_epochs
validation_steps = 30

print("\n ##### INFOS ##### \n")
print(infos)
print("\n ##### INFOS ##### \n")
history = model.fit_generator(generator        = training_generator,
                              steps_per_epoch  = steps_per_epoch,
                              epochs           = nb_epochs,
                              validation_data  = validation_generator,
                              validation_steps = validation_steps
                              )



#%% SAVE THE NETWORK

"""
import inspect
networks_name = inspect.getfile(inspect.currentframe())
networks_name = networks_name.split("/")[-1] # remove path (keep only the name of the file)
networks_name = networks_name.split(".py")[0] # remove .py
"""
networks_name = "model_other_params"
useful_functions.save_network(networks_name, 
                              model, 
                              history, 
                              infos)


#%% test

im_names1 = ["hflip_ISIC_0000000",
            "hflip_ISIC_0000001",
            "hflip_ISIC_0000008"]
im_names2 = ["hflip_ISIC_0000023",
            "hflip_ISIC_0000041",
            "hflip_ISIC_0000074"]
im_names3 = ["hflip_ISIC_0000079",
            "hflip_ISIC_0000276",
            "hflip_ISIC_0000279"]


images_lists = [im_names1, im_names2, im_names3]


for im_names in images_lists :
    predictions = useful_functions.display_some_results(model     = model,
                                                                im_names  = im_names,
                                                                threshold = 0.3,
                                                                dim       = im_size)






