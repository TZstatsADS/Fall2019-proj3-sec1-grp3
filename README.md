# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2019

+ Team ##
+ Team members
	+ Cai, Lingyi lc3352@columbia.edu
	+ Li, Xiaotong xl2788@columbia.edu
	+ Li, Yanan yl4062@columbia.edu
	+ Ma, Yiwen ym2775@columbia.edu
	+ Qiang, Runzi rq2156@columbia.edu

+ Project summary: In this project, we created a classification engine for facial emotion recognition. For the baseline model gbm, the input data set was fiducial markers and the final accuracy was ~32%. For the final model, raw images are used as data input. In the preprocessing part, we applied face detection, 2D data alignment and data augmentation. Several deep neural network models were implemented and VGG19, ResNet18 and ResNet34 final architectures are used for final testing. 10-fold cross validation is implemented and the average accuracy reached ~50% on training dataset. For furture work, we can have more local data augmentation on low accuracy classes, increase the image size, and add landmarks in the training models.

+ Slides：https://docs.google.com/presentation/d/18b-wnVFckduUIE9Pi8Lmz4zYw1i6TVK1QgyV84W0Ve8/edit?ts=5db879e4#slide=id.p
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
