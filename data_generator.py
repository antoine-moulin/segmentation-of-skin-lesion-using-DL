import cv2 as cv
import numpy as np
import os
from PIL import Image
from skimage.io import imread

import keras


def retrieve_ids(path, img_suffix, mask_suffix):
    """
    Retrieve the IDs of images in a folder.

    Parameters
    ----------
    path: string
      Path to the folder that contains images.

    img_suffix: string
      Suffix of the images, e.g. '.tiff'.

    mask_suffix: string
      Suffix of the masks, e.g. '.png'.

    Returns
    -------
    IDs: ndarray, shape (n_images,)
      IDs of the images in the folder.
    """

    IDs = os.listdir(path)
    for k in range(len(IDs)):
        IDs[k] = IDs[k].replace(img_suffix, '')
        IDs[k] = IDs[k].replace(mask_suffix, '')

    return np.unique(IDs)


class DataGenerator(keras.utils.Sequence):
    """
    Object for fitting to a dataset of images with multiple channels (e.g. 5 channels instead of the RGB channels).
    Implement the `__getitem__` and the `__len__` methods from the Keras' Sequence class.

    Attributes
    ----------
    path: string
      Path to the folder containing the dataset.

    list_IDs: ndarray, shape (n_strings,)
      Images' IDs that are to be used for the generator (e.g. images from the training set).
      n_strings == number of IDs.

    img_suffix: string
      Suffix of the images used (e.g. '.jpg' or '.tiff').

    mask_suffix: string
      Suffix of the masks used (e.g. '_segmentation.png').

    batch_size: int
      Size of the batches for the generator.

    dim: tuple, (width, height)
      Dimensions of the images (width and height).

    n_channels: int
      Number of channels in the images.

    n_classes: int
      Number of classes (e.g. for segmentation, 0 if the pixel is outside the mask and 1 otherwise).

    shuffle: boolean
      True if the dataset has to be shuffled, False otherwise.
    """

    def __init__(self, path, list_IDs, img_suffix, mask_suffix, batch_size=1, dim=(320, 320), n_channels=5,
                 n_classes=2, shuffle=True):
        """
        Initialization of a DataGenerator object. See documentation of the class for a description of the attributes.
        """

        self.path = path
        self.list_IDs = list_IDs
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            new_im = imread(self.path + ID + self.img_suffix)
            im_resized = cv.resize(new_im, self.dim)
            X[i, ] = im_resized.copy()

            # Store class
            new_mask = imread(self.path + ID + self.mask_suffix)
            mask_resized = cv.resize(new_mask, self.dim)
            y[i] = np.expand_dims(mask_resized, axis=2) / 255

        return X, y
