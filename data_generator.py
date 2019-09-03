### IMPORTS ###
import numpy as np
from skimage.io import imread
from PIL import Image
import cv2 as cv

import keras
from keras.utils import to_categorical

from tf_unet.image_util import BaseDataProvider
import glob


class DataGenerator(keras.utils.Sequence):
    """
    Object for fitting to a dataset of images with multiple channels (e.g. 5 channels instead of the RGB channels).
    Implement the `__getitem__` and the `__len__` methods from the Keras' Sequence class.

    Attributes:
        path: string, path to the folder containing the dataset.
        list_IDs: array of strings, images' IDs that are to be used for the generator (e.g. images from the training set).
        img_suffix: string, suffix of the images used (e.g. '.jpg' or '.tiff').
        mask_suffix: string, suffix of the masks used (e.g. '_segmentation.png').
        batch_size: int, size of the batches for the generator.
        dim: tuple, dimensions of the images (width and height).
        n_channels: int, number of channels in the images.
        n_classes: int, number of classes (e.g. for segmentation, 0 if the pixel is outside the mask and 1 otherwise).
        shuffle: boolean, True if the dataset has to be shuffled, False otherwise.
    """

    def __init__(self, path, list_IDs, img_suffix, mask_suffix,
                 batch_size=1, dim=(320, 320), n_channels=5, n_classes=2, shuffle=True):
        """
        Initialization of DataGenerator.
        """

        self.dim = dim
        self.batch_size = batch_size
        self.path = path
        self.list_IDs = list_IDs

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

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
        """  # X: (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            new_im = imread(self.path + ID + self.img_suffix)
            im_resized = cv.resize(new_im, self.dim)
            X[i,] = im_resized.copy()

            # Store class
            new_mask = imread(self.path + ID + self.mask_suffix)
            mask_resized = cv.resize(new_mask, self.dim)
            y[i] = np.expand_dims(mask_resized, axis=2) / 255

        return X, y


class Image5DataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif',
                 shuffle_data=True, n_class=2):
        super(Image5DataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class

        self.data_files = self._find_data_files(search_path)

        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        return imread(path).astype(dtype)  # np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)

        return img, label
