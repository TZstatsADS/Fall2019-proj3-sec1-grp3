# Project: Can you recognize the emotion from an image of a face?

### Doc folder
**Lingyi Cai, Yanan Li, Xiaotong Li, Yiwen Ma, Runzi Qiang**

Our 10-fold cross validation results are 50.16%.<br>

Project Description: In this project, we will carry out model evaluation and selection for predictive analytics on image data. As data scientists, we often need to evaluate different modeling/analysis strategies and decide what is the best. Such decisions need to be supported by sound evidence in the form of model assessment, validation and comparison. In addition, we also need to communicate our decision and supporting evidence clearly and convincingly in an accessible fashion.

## Phrase zero: Environment Setup
To replicate the result, please check the environment set up file to correctly set up the pytorch environment.
Phrase one: Model Preparation
In this step, we will be doing offline data augmentation. Which includes face detection, facial landmark extraction, face cropping, and width, height, shift, horizaontal flip, zooming and brightness change.

Train data folder should be under data/images

## Phrase 2: Model training.
We are following the guidelines of this [github](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch). We have experimented many architectures possible by learning multiple resources.

The architecture we have experimented on are:

- VGG16 
- Resnet18
- Resnet34

The training model provided by the following command is using VGG19

## Phrase 3: Model Prediction
The best model we expect is Resnet18. We pretrained by using our trained data. When given test data during class, we use the following command and get prediction based on that.

Please run the following command to see how the images are being classified.
The eventual results for the images is stored in result.csv
For running the test set prediction, you need to renamed the image folder to be test_images and then stored it under the .\finalize_model\data folder.
