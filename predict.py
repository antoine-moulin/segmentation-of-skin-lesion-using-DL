import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.io import imsave
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.interpolation import zoom


from useful_functions import draw_contour_on_image
from preprocessing_unet import preprocessing_unet

from keras.models import model_from_json
from segmentation_models.metrics import iou_score


#%%
def predict_mask(model,
                 im_id,
                 img_dir,
                 img_suffix='.jpg',
                 mask_suffix='_segmentation.png',
                 largest_dimension=250,
                 desired_size=320):
    """
    Takes an image unprocessed as the input and returns the binary mask with the same dimensions predicted by the model.
    The postprocessing consists in applying a dilation and then a filling.

    Usage:
    pred = predict_mask(model, im, img_dir)

    :param model: the model used for the prediction. We suppose it has been trained using the segmentation-models
                  library
    :param im_id: id of the image
    :param img_dir: folder that contains the images for the segmentation (without any preprocessing)
    :param img_suffix: suffix of the original image, default is .jpg
    :param mask_suffix: suffix of the mask, default is .png
    :param largest_dimension: the largest dimension used for the preprocessing. In the paper, 250 is used
    :param desired_size: the desired size of the preprocessed image. In the paper, 320 is used
    :return: a binary mask of the skin lesion

    """

    # preprocess the image
    im = preprocessing_unet(im_id=im_id, mask_bool=False, img_dir=img_dir, mask_dir='', img_suffix=img_suffix,
                            mask_suffix=mask_suffix, largest_dimension=largest_dimension, desired_size=desired_size)
    im = np.expand_dims(im, axis=0)

    # keep the original image for proportions
    original_im = imread(img_dir + im_id + img_suffix)
    rows, columns, _ = original_im.shape

    if rows >= columns:
        percent = largest_dimension / float(rows)
        csize = int((float(columns) * float(percent)))
        original_im = cv2.resize(original_im, (csize, largest_dimension))

    else:
        percent = largest_dimension / float(columns)
        rsize = int((float(rows) * float(percent)))
        original_im = cv2.resize(original_im, (largest_dimension, rsize))

    delta_w = desired_size - original_im.shape[1]
    delta_h = desired_size - original_im.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # predict the mask
    pred = model.predict(im)
    pred = np.squeeze(pred)

    # # threshold
    pred = (pred > 0.3)

    # # post-processing
    pred = binary_dilation(pred).astype(int)
    pred = binary_fill_holes(pred).astype(int)
    pred = binary_dilation(pred).astype(int)
    pred = binary_dilation(pred).astype(int)
    pred = binary_fill_holes(pred).astype(int)

    # # crop the mask
    pred = pred[top:pred.shape[0]-bottom, left:pred.shape[1]-right]

    # # resize so the mask has the same dimensions as the original image
    return zoom(pred, [rows/pred.shape[0], columns/pred.shape[1]])


#%%
# load json and create model
model_dir = './saved_models/segmentation_models/'
model_id = '2019-04-25/'

json_file = open(model_dir + model_id + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_dir + model_id + 'weights.h5')

#%%

results_dir = './results/'

if os.path.isdir(results_dir + model_id) == 0:
    os.mkdir(results_dir + model_id)

specific_images = ['ISIC_0000031', 'ISIC_0000060', 'ISIC_0000073', 'ISIC_0000074', 'ISIC_0000121', 'ISIC_0000166',
                   'ISIC_0000355', 'ISIC_0000395', 'ISIC_0009944', 'ISIC_0010047', 'ISIC_0016064']

for im_test in specific_images:

    pred = predict_mask(loaded_model, im_test, './ISIC2018_Task1-2_Training_Input/')

    original_image = imread('./ISIC2018_Task1-2_Training_Input/' + im_test + '.jpg')
    original_mask = imread('./ISIC2018_Task1_Training_GroundTruth/' + im_test + '_segmentation.png')

    draw_contour_on_image(im_test+'.jpg', './ISIC2018_Task1-2_Training_Input/', original_mask, pred, results_dir+model_id)
