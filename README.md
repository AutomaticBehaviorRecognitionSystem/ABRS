Automatic Behavior Recognition System (ABRS)

Copyright (c) 2019 Primoz Ravbar UCSB
Licensed under BSD 2-Clause [see LICENSE for details]
Written by Primoz Ravbar

Automatic Behavior Recognition System can annotate behaviors of freely moving flies and potentially other animals from video. It extracts spatio-temporal features from video. These spatio-temporal features can then be used with supervised machine learning (ML) to classify behaviors. 

The most current real-time version utilizes a small convolutional neural network directly from the video with simplified pre-processing. It can classify behavior in real-time. This version can be tested by cloning the ABRS and running real_time_ABRS. The latest model (the trained convolutional network) used is: modelConv2ABRS_3C  That's it. It will produce an ethogram (record of behavior) from a video. 

The pre-processing of video is crucial. It extracts features from raw video frames in three time-scales:

  1) Raw frame;
  2) Difference between two frames; and
  3) Spectral features extracted from a wider time window (typically .5 sec).

ConvNet training is implemented by ConvNet_training.ipynb The trained CNN graph and weights can be used for the real-time and batch implementations.

The batch implementation (to run multiple movies and produce ethograms) will be uploaded shortly.

The thouroughly tested older version is described below:

The extraction of the features is implemented in two steps: first, run video_to_ST_image_batch to produce "ST-images" (ST-images capture "shapes of movements" in a defined time-window), second, run ST_image_to_ST_feature_batch to reduce the dimensionality of the ST-images (to 30 spatio-temporal features - down from 80x80 = 6400 dimensions). Here the 30 dimensional spatio-temporal features ("STF") are stored in Data_demo/ST_features.

The STF files contain numpy matrices of the 30 spatio-temporal features calculated for every frame of the raw movies. They also contain max change of light intensity in the ST-image time-window and the total body displacement in the same window ("speed"). These features can be used in supervised or unsupervised ML steps. Here we provide an implementation based on LDA (supervised learning). The human labels are also provided.   
