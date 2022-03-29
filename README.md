Automatic Behavior Recognition System (ABRS)

Copyright (c) 2019 Primoz Ravbar UCSB
Licensed under BSD 2-Clause [see LICENSE for details]
Written by Primoz Ravbar

Automatic Behavior Recognition System can annotate behaviors of freely moving flies, and possibly other animals, from video. It does NOT require alignment of frames, segmentation, anatomical information nor pose estimation. It can reliably recognize behavior in highly variable backgrounds, animal orientations, positions, light levels, movie qualities and other conditions. It does that by extracting spatio-temporal features from video. These spatio-temporal features can then be used with supervised machine learning (ML) to classify behaviors. 

There is now a fully functional GUI for annotation of videos. The output of the GUI is in a format that can be immediately accessible for the training. See the ABRS_GUI_label_maker.pdf (in 'GUI_ABRS' folder) for very clear instructions! The GUI code is 'ABRSLabelMakerGUI.ipynb' (in 'GUI_ABRS' folder).

The latest version of the ABRS (03/28/2022) uses residual neural network architecture - ResNet-50 for predictions. The ST-images and labels produced by the 'ABRSLabelMakerGUI.ipynb' can be used for the ResNet-50 training by ResNet50_ABRS.ipynb. With each new training session, the model should improve, which will make further annotation by the GUI faster. 

The ResNet-50 is too large to upload here so the initial labels can be made in the GUI by using any of the old models (in ‘Models’). Once the first labeled training set is created, a new ResNet-50 model should be trained with it. When the training is complete, start the GUI to annotate more data with the new model. 

To analyze new videos run the video_to_prob_batch.ipynb script. The output is the probabilities of behavioral classes (NumPy arrays). These can be used in downstream analysis, for example to create ethograms.

The most current real-time version utilizes a small convolutional neural network directly from the video with simplified pre-processing. It can classify behavior in real-time. This version can be tested by cloning the ABRS and running real_time_ABRS. A sample model (the trained convolutional network) used is: modelConv2ABRS_3C Other, better models, can be found in the "Model" folder. That's it. It will produce an ethogram (record of behavior) from a video. 

The pre-processing (production of ST-images) of video is crucial. It extracts features from raw video frames in three time-scales:

  1) Raw frame;
  2) Difference between two frames; and
  3) Spectral features extracted from a wider time window (typically .5 sec).
  
See ST_images_samle_anterior.png for an example of 3-channel spatio-temporal images (blue - raw frame; yellow/green - frame-to-frame difference; red - spectral features from broader time window). The fly in these images is involved in "leg rubbing" [1] and "head cleaning" behaviors [2]. The behavioral labels (classes) [1] and [2] above the images are automatically created by ABRS. Another example, SampleImagesWingCleanin.png, is showing several different flies, in various backgrounds, positions, orientations and light levels, all engaged in the same behavior (wing cleaning). Again, these highly variable images were classified as "wing cleaning" by the ABRS, illustrating good generalization.

Also see the paper: Ravbar, Primoz, Kristin Branson, and Julie H. Simpson. "An automatic behavior recognition system classifies animal behaviors using movements and their temporal context." Journal of Neuroscience Methods (2019): 108352. (https://www.sciencedirect.com/science/article/pii/S0165027019302092)


