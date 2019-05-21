# ABRS

Automatic Behavior Recognition System can annotate behaviors of freely moving flies from video. It extracts spatio-temporal features from video. These spatio-temporal features can then be used with supervised machine learning (ML) to classify behaviors. 

The most current version utilizes a small convolutional neural network directly from the video with simplified pre-processing. It can classify behavior in real-time. This version can be tested by cloning the ABRS an running real_time_ABRS. That's it. It will produce an ethogram (record of behavior) from a video. 

The thouroughly tested older version is described below:

The extraction of the features is implemented in two steps: first, run video_to_ST_image_batch to produce "ST-images" (ST-images capture "shapes of movements" in a defined time-window), second, run ST_image_to_ST_feature_batch to reduce the dimensionality of the ST-images (to 30 spatio-temporal features - down from 80x80 = 6400 dimensions). Here the 30 dimensional spatio-temporal features ("STF") are stored in Data_demo/ST_features.

The STF files contain numpy matrices of the 30 spatio-temporal features calculated for every frame of the raw movies. They also contain max change of light intensity in the ST-image time-window and the total body displacement in the same window ("speed"). These features can be used in supervised or unsupervised ML steps. Here we provide an implementation based on LDA (supervised learning). The human labels are also provided.   
