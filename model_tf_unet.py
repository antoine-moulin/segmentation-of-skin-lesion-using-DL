import os
import numpy as np

from data_generator import DataGenerator
from data_generator import Image5DataProvider

from tf_unet import unet

from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
from skimage.transform import resize

# %%

# Parameters
params = {'img_suffix': '.tiff',
          'mask_suffix': '_segmentation.png',
          'dim': (320, 320),  # with the module segmentation_models, dim must be divisible by factor 32
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Data sets
data_path = './ISIC2018_rgb320/'
IDs = os.listdir(data_path)
n = len(IDs)

for k in range(n):
    IDs[k] = IDs[k].replace(params['img_suffix'], '')
    IDs[k] = IDs[k].replace(params['mask_suffix'], '')

IDs = np.unique(IDs)

n = len(IDs)

indices = np.random.permutation(n)
train_idx, validation_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

partition = {'train': np.array(IDs)[train_idx],
             'validation': np.array(IDs)[validation_idx]
             }

# Generators
training_generator = DataGenerator(path=data_path, list_IDs=partition['train'], **params)
validation_generator = DataGenerator(path=data_path, list_IDs=partition['validation'], **params)

# %%
output_path = './unet_trained'
training_provider = Image5DataProvider(data_path + 'train/*', mask_suffix=params['mask_suffix'],
                                       data_suffix=params['img_suffix'])

net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(training_provider, output_path, training_iters=len(train_idx), epochs=1)

#%%

# preds = model.predict_generator(validation_generator)
# imsave('pred0.png', preds[0])

validation_provider = Image5DataProvider(data_path + 'val/*', mask_suffix=params['mask_suffix'],
                                       data_suffix=params['img_suffix'])

#%%

x_test, y_test = validation_provider(1)
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
prediction = net.predict("./results/tfunet_trained/model.ckpt", x_test)
#%%
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.49
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
fig.tight_layout()
fig.show() #("./prediction/.png")