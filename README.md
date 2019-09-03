# Segmentation of skin lesion images using deep learning techniques

This repository contains:

* <tt>preprocessing_unet.py</tt> to create the data set and preprocess the images
* <tt>data_generator.py</tt> to handle the data
* <tt>model_brut_implementation.py</tt> (using [<tt>Keras</tt>](https://keras.io/)), <tt>model.py</tt> (using [<tt>segmentation_models</tt>](https://github.com/qubvel/segmentation_models)), <tt>model_tf_unet.py</tt> (using [<tt>tf_unet</tt>](https://github.com/jakeret/tf_unet))
* <tt>predict.py</tt>
* <tt>useful_functions.py</tt>
* The folder <tt>results</tt> with skin lesions and the associated contours predicted by our model
* The folder <tt>saved_models</tt> with the weights of our model

The paper used for this project is : [Skin Lesion Segmentation: U-Nets versus Clustering](https://arxiv.org/pdf/1710.01248.pdf), written by Bill S. Lin, Kevin Michael,  Shivam Kalra, H.R. Tizhoosh.

# Usage

To use this project: 

1. Open the file <tt>preprocessing_unet.py</tt> and follow the instructions in the docstring of the function <tt>build_training_set</tt> to create the data set (download images of ISIC challenge etc.). The data set is too heavy to be stored on the git repository.
2. Go to the file <tt>predict.py</tt> to predict some masks. The results will be displayed in the folder </tt>results/model_id</tt>. The model provided here has the id <tt>2019-04-25_12-19-15</tt>.

# The different files

## <tt>preprocessing_unet.py

In this file, we transform the RGB images into 5-channels images as indicated in the paper. We just take a size of 320 x 320 instead of 342 x 342.

## <tt>data_generator.py</tt>

This file contains a class, <tt>DataGenerator</tt>, that allows us to manage the data set, which contains 5-channel images and binary masks.

## <tt>model.py</tt>

This file contains the code for our model. We did try to fully implement the U-Net or to use the library <tt>tf_unet</tt> but we focused on the <tt>segmentation_models</tt> library. Besides, this model takes a size divisible by 32!

## <tt>predict.py</tt>

To predict a mask, see the last cell of the file to use your own images. The original images must be in the folder <tt>ISIC2018_Task1-2_Training_Input</tt> and the ground truth mask, if available, in the folder <tt>ISIC2018_Task1_Training_GroundTruth</tt>.
