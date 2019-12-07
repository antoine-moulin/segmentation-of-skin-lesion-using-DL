#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import copy
import datetime
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from skimage import measure
from skimage.io import imread, imsave
from skimage.morphology import black_tophat, dilation, disk


# add folders to the path
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = parent_folder_path
results_dir = main_dir + "/saved_models"
predictions_dir = main_dir + "/predictions"
if main_dir not in sys.path:
    sys.path.append(main_dir)


def remove_hair(im):
    """
    Segmentation of the hairs in an image in order to remove them.

    Usage:
     im = remove_hair(im)

    Parameters
    ----------
    im: ndarray, shape (width, height, channels)
      Image that contains hairs.

    Returns
    -------
    new_im: ndarray (width, height, channels)
      An image with hairs removed.
    """

    n, m, _ = im.shape

    # step 1: looking for the mask using a threshold on the RGB image
    bool_im = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if im[i, j, 0] < 20 or im[i, j, 1] < 20 or im[i, j, 2] < 20:
                bool_im[i, j] = 1
            else:
                bool_im[i, j] = 0

    # step 2: looking for the mask using a morphological operation and then a threshold
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    tophat = black_tophat(gray_im, disk(20)) / 255
    tophat = (tophat > 0.15).astype(np.int)  # 0.2 is used in the paper

    # step 3: take the union of both results
    mask = dilation((bool_im.astype(np.bool) + tophat.astype(np.bool)) * 1)

    # step 4: if there are not enough hairs, no need to preprocess
    if 100 * sum(sum(mask)) / (n * m) < 5:
        return im

    # step 5: we remove the hairs as indicated in the paper
    new_im = im.copy()

    for i in range(n):
        for j in range(m):
            if mask[i, j] == 1:
                i_index = [max(i - 30, 0), min(i + 30, n)]
                j_index = [max(j - 30, 0), min(j + 30, m)]
                temp = im[i_index[0]:i_index[1], j_index[0]:j_index[1]]

                new_im[i, j, 0] = np.median(temp[:, :, 0])
                new_im[i, j, 1] = np.median(temp[:, :, 1])
                new_im[i, j, 2] = np.median(temp[:, :, 2])

    return new_im


def save_network(networks_name, classifier, history, infos_dict):
    """
    Saving a trained network with its parameters and its scores in a new folder.

    Parameters
    ----------
    networks_name: string
      The name of the network (for the saving).

    classifier: Model
      The classifier which has been trained.

    history: keras History
      A keras History object (e.g. returned by fit_generator function from keras).

    infos_dict: dict
      Dictionnary containing useful information about the network and its training params.
    """

    # date
    date = str(datetime.datetime.now())
    date = date.replace(' ', '_')
    date = date.replace(':', '-')
    date = date.split('.', 1)[0]

    networks_dir = results_dir + '/' + networks_name

    # recover subfolders names to check if it already exists
    subfolders = [f.name for f in os.scandir(results_dir) if f.is_dir()]

    if networks_name not in subfolders:
        # if it doesn't already exists, create a subfolder for this network
        os.makedirs(networks_dir)
        print('subfolder [', networks_dir, '] built')

    # create the test folder (with the date of the test)
    new_test_dir = networks_dir + '/' + date
    if os.path.exists(new_test_dir):
        print('\n\nWARNING : THE FOLDER ', new_test_dir, ' ALREADY EXISTS.\n\n')
        new_test_dir = new_test_dir + '_copy'
    os.makedirs(new_test_dir)

    # save model and weights
    model_json = classifier.to_json()
    with open(new_test_dir + '/model.json', 'w') as json_file:
        json_file.write(model_json)
    classifier.save_weights(new_test_dir + '/weights.h5')

    # save infos_dict
    with open(new_test_dir + '/infos.json', 'w') as json_file:
        json_file.write(json.dumps(infos_dict))

    # save history
    save_history_figure(history, new_test_dir)


def save_history_figure(history, path):
    """
    Save the history's graph in a file.

    Parameters
    ----------
    history: keras History
      A keras History object (e.g. returned by fit_generator function from keras).

    path: string
      Indicates the path of the test folder.
    """

    # save history
    with open(path + '/history_obj.json', 'w') as json_file:
        json.dump(history.history, json_file)

    fig = plt.figure()
    plt.plot(history.history['iou_score'])
    if 'val_iou_score' in history.history.keys():
        plt.plot(history.history['val_iou_score'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if 'val_iou_score' in history.history.keys():
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.savefig(path + '/history.pdf')
    plt.show()


def display_some_results(model, im_names, im_path, im_suffix, mask_suffix, threshold=0.3, dim=(320, 320), display=True,
                         save=False):
    """
    Compute predictions with a model and display and/or save them.

    Parameters
    ----------
    model: Model
      The classifier which has been trained.

    im_names: list of strings
      Contains the images names.

    im_path: string
      Path that contains the images.

    im_suffix: string
      Suffix of the images, e.g. '.tiff'.

    mask_suffix: string
      Suffix of the masks, e.g. '.png'.

    threshold: float
      Number used to threshold grayscale image. Default is 0.3.

    dim: tuple, (width, height)
      Refers to the dimensions of the images. Default is (320, 320).

    display: boolean
      States if images will be displayed. Default is True.

    save: boolean
      States if images will be saved. Default is True.

    Returns
    -------
    predictions: list of ndarrays, shape (width, height)
      Raw masks predicted by the model.

    bin_predictions: list of ndarrays, shape (width, height)
      Masks predicted by the model after thresholding.
    """

    data_path = im_path
    nb_images = len(im_names)
    fig = plt.figure()
    rows = nb_images
    columns = 4
    ax = []
    predictions = []
    bin_predictions = []    
    
    if save:
        # date
        date = str(datetime.datetime.now())
        date = date.replace(' ', '_')
        date = date.replace(':', '-')
        date = date.split('.', 1)[0]

        # create the new test folder
        saving_path = predictions_dir + '/' + date
        if os.path.exists(saving_path):
            print('\n\nWARNING: THE FOLDER ', saving_path, ' ALREADY EXISTS.\n\n')
            saving_path = saving_path + '_copy'
        os.makedirs(saving_path)

    for k in range(nb_images):
        im_id = im_names[k]

        # test and original images
        image_test = imread(data_path + im_id + im_suffix)
        original_image = image_test[:, :, :3]

        # ground truth
        ground_truth = imread(data_path + im_id + mask_suffix)

        # grayscale prediction
        prediction = model.predict(np.expand_dims(cv2.resize(image_test, dim), axis=0))
        grayscale_pred = np.squeeze(prediction)
        predictions.append(grayscale_pred)

        # binary prediction
        grayscale_pred_bin = (copy.deepcopy(grayscale_pred) >= threshold).astype(np.uint8)

        bin_predictions.append(grayscale_pred_bin)

        if display:
            ax.append(fig.add_subplot(rows, columns, k * 4 + 1))
            ax[-1].set_title('Original image')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(original_image, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 2))
            ax[-1].set_title('Grayscale prediction')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(grayscale_pred, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 3))
            ax[-1].set_title('Binary prediction')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(grayscale_pred_bin, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 4))
            ax[-1].set_title('Ground truth')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(ground_truth, cmap='gray')

        if save:
            final_name = saving_path + '/' + im_id + '.jpg'
            imsave(final_name, grayscale_pred_bin * 255, cmap='gray')

    if display:
        plt.subplots_adjust(wspace=0, hspace=1)

    return predictions, bin_predictions


def draw_contour_on_image(img_path, mask_path, prediction, output_dir):
    """
    Plot the skin lesion with the contours of both the ground truth mask and the mask predicted, in order to compare the
    results.

    Usage:
     img_path = ...
     mask_path = ...
     prediction = model.predict(original_image)
     draw_contour_on_image(img_path, mask_path, prediction, output_dir)

    Parameters
    ----------
    img_path: string
      Path to the original image.

    mask_path: string
      Path to the original mask (ground truth).

    prediction: ndarray, shape (width, height)
      The mask predicted by a model.

    output_dir: string
      The folder in which to store the result.
    """

    original_image = imread(img_path)
    original_mask = imread(mask_path)

    # find all the contours
    gt_contours = measure.find_contours(original_mask, 0.5)
    pred_contours = measure.find_contours(prediction, 0.5)

    # iterate over all the contours to find the longest one
    best_gt_contour = np.array([])
    best_gt_length = 0
    best_pred_contour = np.array([])
    best_pred_contour2 = np.array([])
    best_pred_length = 0

    for n, contour in enumerate(gt_contours):
        if len(contour[:, 1]) > best_gt_length:
            best_gt_length = len(contour[:, 1])
            best_gt_contour = contour

    plt.figure()

    plt.imshow(original_image, interpolation='nearest')

    for n, contour in enumerate(pred_contours):
        if len(contour[:, 1]) > best_pred_length:
            best_pred_length = len(contour[:, 1])
            best_pred_contour2 = best_pred_contour
            best_pred_contour = contour
        plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)

    plt.plot(best_gt_contour[:, 1], best_gt_contour[:, 0], color='green', linewidth=1, label='Ground Truth')
    plt.plot(best_pred_contour[:, 1], best_pred_contour[:, 0], color='blue', linewidth=1, label='Prediction')

    if best_pred_contour2.shape[0] != 0:
        if best_pred_contour2.shape[0] > 0.6*best_pred_contour.shape[0]:
            plt.plot(best_pred_contour2[:, 1], best_pred_contour2[:, 0], color='blue', linewidth=1)

    plt.title('Skin lesion with the contours of the masks')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best')

    plt.savefig(output_dir + img_path.split('/')[-1])
