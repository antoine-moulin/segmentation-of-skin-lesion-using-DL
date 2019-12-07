# Segmentation of skin lesion images using deep learning techniques

The paper used for this project is: [Skin Lesion Segmentation: U-Nets versus Clustering](https://arxiv.org/pdf/1710.01248.pdf), written by Bill S. Lin, Kevin Michael,  Shivam Kalra, H.R. Tizhoosh.

# Usage

To use this project:

1. Open the file <tt>preprocessing_unet.py</tt> and follow the instructions in the docstring of the function <tt>build_training_set</tt> to create the data set (download images of ISIC challenge etc.). The data set is too heavy to be stored on the git repository but is available here:
2. Go to the file <tt>train.py</tt> to train a model. It will be saved in the <tt>saved_models</tt> folder.
3. Go to the file <tt>predict.py</tt> to predict some masks. The results will be displayed in the folder </tt>results/model_id</tt>. The model provided here has the id <tt>2019-04-25_12-19-15</tt>.

# Organization

* <tt>preprocessing.py</tt>

In this file, we transform the RGB images into 5-channels images as indicated in the paper. We just take a size of 320 x 320 instead of 342 x 342.

* <tt>data_generator.py</tt>

This file contains a class, <tt>DataGenerator</tt>, that allows us to manage the data set, which contains 5-channel images and binary masks.

* <tt>train.py</tt>

This file contains the code to train our model. We did try to fully implement the U-Net or to use the library [<tt>tf_unet</tt>](https://github.com/jakeret/tf_unet) but we focused on the [<tt>segmentation_models</tt>](https://github.com/qubvel/segmentation_models) library which yielded better results. Besides, this model takes a size divisible by 32!

* <tt>predict.py</tt>

To predict a mask, see the last cell of the file to use your own images. The original images must be in the folder <tt>ISIC2018_Task1-2_Training_Input</tt> and the ground truth mask, if available, in the folder <tt>ISIC2018_Task1_Training_GroundTruth</tt>.

* <tt>utils.py</tt>

This file contains some functions to save a model, to plot some results etc.
