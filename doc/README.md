# Project: Can you recognize the emotion from an image of a face?

### Doc folder
**Lingyi Cai, Yanan Li, Xiaotong Li, Yiwen Ma, Runzi Qiang**
We highly recommend you to use main.py in finalize_model folder instead of main.ipynd in this folder<br>
Our 10-fold cross validation results are 50.16%.<br>

All the scripts and data are in the `Fall2019-proj3-sec1--proj3-sec1-grp3/finalize_model` folder. 



Project Description: In this project, we will carry out model evaluation and selection for predictive analytics on image data. As data scientists, we often need to evaluate different modeling/analysis strategies and decide what is the best. Such decisions need to be supported by sound evidence in the form of model assessment, validation and comparison. In addition, we also need to communicate our decision and supporting evidence clearly and convincingly in an accessible fashion.

**Important:** Trained models have already saved in `/finalize_model/trained_models` folder, so users don't have to train them again. You can use the pretrained models to predict the images. 

***
**If you want to predict a bunch of new images:" <br>

Please run `Just_predict_main.py`. 



***
**If you want to train the models:**

## Phrase 0: Environment Setup
To replicate the result, please check the environment set up file to correctly set up the pytorch environment.


## Phrase 1: Data pre-processing 
In this step, we will be doing offline data augmentation. Which includes face detection, facial landmark extraction, face cropping, and width, height, shift, horizaontal flip, zooming and brightness change.<br>
1. `step0_faceCropper.py` is used for cropping the images from 750x1000x3 to 256x256x3, and reduce the influence of background, clothes and hair. After this step, you will generate a folder `data_0` contains 256x256x3 images in `data` folder. <br>

2. `step1_faceAlignment.py` is used for aligning the face and choose more precise emotions. The images will be cropped, and we will get 48x48 image size. After this step, you will generate a folder `data_1` contains 48x48x3 images in `data` folder. <br>

3. `step2_subFolders.py` is used for classifying the images to 22 subfolders based on the labels. After this step, you will generate a folder `data_2` contains 22 subfolders in `data` folder. In each subfolder, you will have ~100 48x48 grayscale images in a same class. 

4. `step3_H5preprocess.py` is used for generating `data.h5` file which will be used as our input data. After this step, you will get a `data.h5` file in `data` folder. 

## Phrase 2: Model training.
We are following the guidelines of this [github](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch). We have experimented many architectures possible by learning multiple resources.

The architecture we have experimented on are:

- VGG16 
- Resnet18
- Resnet34

1. `step4_mainpro.py` is used for training (90% for training and 10% for testing). We highly recommend you to run `step5_k_fold_train.py` directly which applied 10-fold cross validation. This will take longer time but will get a more similar accuracy with testing data. In addition, if you want to train a model by using all the images, please run `step4_mainpro_allData.py`. <br>
2. `step6_plot_CK+_confusion_matrix.py` is used for generating confusion matrix after 10-fold cross validation. Confusion matrix will be significant when you do the error analysis. <br>

## Phrase 3: Model Prediction
The best model we expect is Resnet34. But the training accuracy is more similar with testing accuracy by using VGG19 architecture. 

1. `step7_predPrivateTest.py` is used for generating the results. 
