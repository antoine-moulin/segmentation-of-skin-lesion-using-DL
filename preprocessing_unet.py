# imports
import numpy as np
import os
import cv2 as cv

from skimage.io import imread
from skimage.io import imsave


# %%

def make_gaussian(size, fwhm=125, center=None):
    """
    Make a square gaussian kernel.

    Usage:
    gauss = make_gaussian(size)

    :param size: length of a side of the square
    :param fwhm: full-width-half-maximum, which can be thought of as an effective radius
    :param center: position of the center of the gaussian, default is at the center of the image
    :return: an image that contains a 2D Gaussian

    The code comes from here: https://gist.github.com/andrewgiessel/4635563.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def preprocessing_unet(im_id,
                       mask_bool=True,
                       img_dir='./ISIC2018_Task1-2_Training_Input/',
                       mask_dir='./ISIC2018_Task1_Training_GroundTruth/',
                       img_suffix='.jpg',
                       mask_suffix='_segmentation.png',
                       largest_dimension=250,
                       desired_size=320):
    """
    From a RGB image, create a 5-channel image that contains :
        - RGB channels after a histogram equalization has been done on the intensity channel in the HSI space
        - the original intensity channel
        - a 2D gaussian centered on the image
    Besides, we resize the image following the method indicated by the paper.

    Usage:
    im, mask = preprocessing_unet(im_id) # if training set
    im = preprocessing_unet(im_id, mask=False) # if test set

    :param im_id: id of the image
    :param mask_bool: indicates if there is a mask to process (e.g. for the training set), default is True
    :param img_dir: folder that contains the original images
    :param mask_dir: folder that contains the ground truth masks
    :param img_suffix: suffix for the image, default is .jpg
    :param mask_suffix: suffix for the mask, default is _segmentation.png
    :param largest_dimension: the largest dimension of the image before padding
    :param desired_size: pad the image so it is a square image whose dimensions have the desired size
    :returns: the preprocessed image and, if mask_bool is True, also the mask

    """

    new_im = imread(img_dir + im_id + img_suffix)
    if mask_bool:
        new_mask = imread(mask_dir + im_id + mask_suffix)

    # resize so that the largest dimension is 250 -------------------------------------------
    rows, columns, _ = new_im.shape

    if rows >= columns:
        percent = largest_dimension / float(rows)
        csize = int((float(columns) * float(percent)))
        new_im = cv.resize(new_im, (csize, largest_dimension))
        if mask_bool:
            new_mask = cv.resize(new_mask, (csize, largest_dimension))

    else:
        percent = largest_dimension / float(columns)
        rsize = int((float(rows) * float(percent)))
        new_im = cv.resize(new_im, (largest_dimension, rsize))
        if mask_bool:
            new_mask = cv.resize(new_mask, (largest_dimension, rsize))

    # convert RGB image to HSI image -------------------------------------------------------
    im_hsi = cv.cvtColor(new_im, cv.COLOR_RGB2HLS)

    # original intensity channel -----------------------------------------------------------
    original_intensity = im_hsi[:, :, 1]

    delta_w = desired_size - new_im.shape[1]
    delta_h = desired_size - new_im.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # we pad this image
    original_intensity = cv.copyMakeBorder(original_intensity, top, bottom, left, right, cv.BORDER_CONSTANT,
                                           value=[255])
    original_intensity = np.expand_dims(original_intensity, axis=2)

    # histogram equalization on the intensity channel then convert back to RGB -------------
    im_hsi[:, :, 1] = cv.equalizeHist(im_hsi[:, :, 1])
    new_im = cv.cvtColor(im_hsi, cv.COLOR_HLS2RGB)

    # we pad the image
    new_im = cv.copyMakeBorder(new_im, top, bottom, left, right, cv.BORDER_CONSTANT, value=[255, 255, 255])

    if mask_bool:
        # we also pad the mask
        new_mask = cv.copyMakeBorder(new_mask, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0])

    # 2D gaussian --------------------------------------------------------------------------
    gauss = make_gaussian(desired_size)
    gauss = np.expand_dims(gauss, axis=2)

    # concatenation of the different channels ----------------------------------------------
    im_5ch = np.concatenate((new_im / 255, original_intensity / 255, gauss), axis=2)

    if mask_bool:
        return im_5ch, new_mask
    else:
        return im_5ch


# %%
# the data set is available here:
# https://challenge2018.isic-archive.com/task1/training/

def build_training_set(output_path='./ISIC2018_data/',
                       img_dir='./ISIC2018_Task1-2_Training_Input/',
                       mask_dir='./ISIC2018_Task1_Training_GroundTruth/',
                       largest_dimension=250,
                       desired_size=320,
                       img_suffix='.jpg',
                       mask_suffix='_segmentation.png',
                       specific_ids=['ISIC_0000031', 'ISIC_0000060', 'ISIC_0000073', 'ISIC_0000074', 'ISIC_0000121', 'ISIC_0000166', 'ISIC_0000355', 'ISIC_0000395', 'ISIC_0009944', 'ISIC_0010047', 'ISIC_0016064'],
                       seed=42):
    """
    Build the preprocessed data set from the ISIC data set. The preprocessing applies the method indicated in the paper.

    Usage:
    Download and unzip the ISIC data set (https://challenge2018.isic-archive.com/task1/training/)
    build_training_set()

    :param output_path: folder in which we save the data set
    :param img_dir: folder that contains the images from ISIC
    :param mask_dir: folder that contains the masks from ISIC
    :param largest_dimension: the largest dimension of the image before padding
    :param desired_size: pad the image so it is a square image whose dimensions have the desired size
    :param img_suffix: extension of the images
    :param mask_suffix: extension of the masks
    :param specific_ids: make sure the ids in this list are in the test set
    :param seed: seed used for the split

    """

    # we create the folders for the data set
    if os.path.isdir(output_path) == 0:
        os.mkdir(output_path)
    if os.path.isdir(output_path + 'train') == 0:
        os.mkdir(output_path + 'train')
    if os.path.isdir(output_path + 'test') == 0:
        os.mkdir(output_path + 'test')

    # names of the images
    list_ids = os.listdir(img_dir)

    # we remove the .txt files
    if 'LICENSE.txt' in list_ids:
        list_ids.remove('LICENSE.txt')
    if 'ATTRIBUTION.txt' in list_ids:
        list_ids.remove('ATTRIBUTION.txt')

    # we only keep the ids
    for k in range(len(list_ids)):
        list_ids[k] = list_ids[k].replace(img_suffix, '')

    n = len(list_ids)

    # we split our data set
    indices = np.random.RandomState(seed=seed).permutation(n)
    train_idx, validation_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    partition = {'train': np.array(list_ids)[train_idx],
                 'test': np.array(list_ids)[validation_idx]
                 }

    # we check that the ids we want to test are in the test set
    if specific_ids:
        for k in range(len(partition['train'])):
            for id_ in specific_ids:
                if id_ == partition['train'][k]:
                    rd_idx = np.random.randint(len(partition['test']))
                    partition['train'][k] = partition['test'][rd_idx]
                    partition['test'][rd_idx] = id_

    # we create the training set
    for k in range(len(partition['train'])):
        im_path = partition['train'][k]

        im, mask = preprocessing_unet(im_path, True, img_dir, mask_dir, img_suffix, mask_suffix, largest_dimension,
                                      desired_size)

        hflip_im, hflip_mask = cv.flip(im, 0), cv.flip(mask, 0)
        vflip_im, vflip_mask = cv.flip(im, 1), cv.flip(mask, 1)
        rot_im, rot_mask = cv.flip(im, -1), cv.flip(mask, -1)

        imsave(output_path + 'train/' + partition['train'][k] + '.tiff', im)
        imsave(output_path + 'train/' + partition['train'][k] + mask_suffix, mask)

        imsave(output_path + 'train/' + 'hflip_' + partition['train'][k] + '.tiff', hflip_im)
        imsave(output_path + 'train/' + 'hflip_' + partition['train'][k] + mask_suffix, hflip_mask)

        imsave(output_path + 'train/' + 'vflip_' + partition['train'][k] + '.tiff', vflip_im)
        imsave(output_path + 'train/' + 'vflip_' + partition['train'][k] + mask_suffix, vflip_mask)

        imsave(output_path + 'train/' + 'rot_' + partition['train'][k] + '.tiff', rot_im)
        imsave(output_path + 'train/' + 'rot_' + partition['train'][k] + mask_suffix, rot_mask)

    # we create the test set
    for k in range(len(partition['test'])):
        im_path = partition['test'][k]

        im, mask = preprocessing_unet(im_path, True, img_dir, mask_dir, img_suffix, mask_suffix, largest_dimension, desired_size)

        imsave(output_path + 'test/' + partition['test'][k] + '.tiff', im)
        imsave(output_path + 'test/' + partition['test'][k] + mask_suffix, mask)
