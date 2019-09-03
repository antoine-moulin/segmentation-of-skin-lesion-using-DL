#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% SYSTEM IMPORTS ###
import sys
import os
import inspect

### ADDING FOLDERS TO THE PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = parent_folder_path
results_dir = main_dir + "/saved_models"
predictions_dir = main_dir + "/predictions"
if not main_dir in sys.path :
    sys.path.append(main_dir)


#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import measure
from skimage.io import imread
from skimage.io import imsave
from skimage.morphology import disk
from skimage.morphology import black_tophat
from skimage.morphology import dilation

import datetime
import json
import copy




#%% USEFUL FUNCTIONS
def remove_hair(im):
    """
    Segmentation of the hairs in an image in order to remove them.

    Usage:
    im = remove_hair(im)

    :param im: image that contains hairs.
    :return: an image with hair removed

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
    Saving a trained network with its parameters and its scores in a new folder

    :param networks_name: the name of the network (for the saving)
    :param classifier: the classifier which has been trained
    :param history: a keras History object (e.g. : returned by fit_generator function from keras)
    :param infos_dict: dictionnary containing useful information about the network and its training params

    """

    # date
    date = str(datetime.datetime.now())
    date = date.replace(" ", "_")
    date = date.replace(":", "-")
    date = date.split(".", 1)[0]

    networks_dir = results_dir + "/" + networks_name

    # recover subfolders names to check if it already exists
    subfolders = [f.name for f in os.scandir(results_dir) if f.is_dir()]

    if not networks_name in subfolders:
        # if it doesn't already exists, create a subfolder for this network
        os.makedirs(networks_dir)
        print("subfolder [", networks_dir, "] built")

    # create the test folder (with the date of the test)
    new_test_dir = networks_dir + "/" + date
    if os.path.exists(new_test_dir):
        print("\n\nWARNING : THE FOLDER ", new_test_dir, " ALREADY EXISTS.\n\n")
        new_test_dir = new_test_dir + "_copy"
    os.makedirs(new_test_dir)

    # save model and weights
    model_json = classifier.to_json()
    with open(new_test_dir + "/model.json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights(new_test_dir + "/weights.h5")

    # save infos_dict
    with open(new_test_dir + "/infos.json", "w") as json_file:
        json_file.write(json.dumps(infos_dict))

    # save history
    save_history_figure(history, new_test_dir)


def save_history_figure(history, path):
    """

    :param history: a keras History object (e.g. : returned by fit_generator function from keras)
    :param path: string which indicates the path of the test folder

    """

    # save history
    with open(path + "/history_obj.json", 'w') as json_file:
        json.dump(history.history, json_file)

    fig = plt.figure()
    plt.plot(history.history['iou_score'])
    if "val_iou_score" in history.history.keys():
        plt.plot(history.history['val_iou_score'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if "val_iou_score" in history.history.keys():
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.savefig(path + "/history.pdf")
    plt.show()


def display_some_results(model, im_names, threshold=0.3, dim=(320, 320), display=True, save=False):
    
    """

    :param model: the classifier which has been trained
    :param im_names: list of strings (image names)
    :param threshold: number used to threshold grayscale image
    :param dim: tuple which refers to the dimensions of the images
    :param display: boolean which states if images will be displayed
    :param save: boolean which states if images will be saved
    :return: tuple with list of predictions and list of binary predictions

    """
    data_path = '../ISIC2018_data/test/'

    nb_images = len(im_names)
    fig = plt.figure()
    columns = 4
    rows = nb_images
    ax = []
    predictions = []
    bin_predictions = []    
    
    if save:
        # date
        date = str(datetime.datetime.now())
        date = date.replace(" ", "_")
        date = date.replace(":", "-")
        date = date.split(".", 1)[0]

        # create the new test folder (avec la date et l'heure du test)
        saving_path = predictions_dir + "/" + date
        if os.path.exists(saving_path):
            print("\n\nWARNING : THE FOLDER ", saving_path, " ALREADY EXISTS.\n\n")
            saving_path = saving_path + "_copy"
        os.makedirs(saving_path)

    for k in range(len(im_names)):

        im_name_without_ext = im_names[k]

        # original image
        original_image_name = im_name_without_ext + ".tiff"
        original_image = imread(data_path + original_image_name)[:, :, :3]

        # ground truth
        ground_truth_name = im_name_without_ext + "_segmentation.png"
        ground_truth = imread(data_path + ground_truth_name)

        # grayscale prediction
        im_name = im_name_without_ext + ".tiff"
        image_test = imread(data_path + im_name)
        im_resized = np.expand_dims(cv2.resize(image_test, dim), axis=0)
        prediction = model.predict(im_resized)
        grayscale_pred = np.squeeze(prediction)
        predictions.append(grayscale_pred)

        # binary prediction
        grayscale_pred_bin = copy.deepcopy(grayscale_pred)
        for i in range(grayscale_pred_bin.shape[0]):
            for j in range(grayscale_pred_bin.shape[1]):
                if grayscale_pred_bin[i, j] < threshold:
                    grayscale_pred_bin[i, j] = 0
                else:
                    grayscale_pred_bin[i, j] = 1
        bin_predictions.append(grayscale_pred_bin)

        if display:
            ax.append(fig.add_subplot(rows, columns, k * 4 + 1))
            ax[-1].set_title('original image')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(original_image, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 2))
            ax[-1].set_title('grayscale prediction')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(grayscale_pred, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 3))
            ax[-1].set_title('binary prediction')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(grayscale_pred_bin, cmap='gray')
            ax.append(fig.add_subplot(rows, columns, k * 4 + 4))
            ax[-1].set_title('ground truth')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            plt.imshow(ground_truth, cmap='gray')

        if save:
            final_name = saving_path + "/" + im_name_without_ext + ".jpg"
            imsave(final_name, grayscale_pred_bin)

    if display:
        plt.subplots_adjust(wspace=0, hspace=1)

    return predictions, bin_predictions




def draw_contour_on_image(img_name, img_dir, ground_truth, prediction, output_dir):
    """
    Plot the skin lesion with the contours of both the ground truth mask and the mask predicted, in order to compare the
    results.

    Usage:
    # the image to use is in the folder img_dir, and is named img_name
    # the result will be stored in output_dir

    ground_truth = imread(...)
    prediction = model.predict(original_image)
    draw_contour_on_image(img_name, img_dir, ground_truth, prediction, output_dir)

    :param img_name: name of an image of a skin lesion
    :param img_dir: folder that contains the original image
    :param ground_truth: the real mask of the image
    :param prediction: the mask predicted by a model
    :param output_dir: the folder in which to store the result
    :return: none

    """

    original_image = imread(img_dir + img_name)

    # find all the contours
    gt_contours = measure.find_contours(ground_truth, 0.5)
    pred_contours = measure.find_contours(prediction, 0.5)

    # iterate over all the contours to find the longest one
    best_gt_contour = np.array([])
    best_gt_length = 0

    best_pred_contour = np.array([])
    best_pred_length = 0

    best_pred_contour2 = np.array([])
    best_pred_length2 = 0

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

    plt.savefig(output_dir + img_name.replace('./', ''))