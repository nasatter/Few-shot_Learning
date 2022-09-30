# Few-shot_Learning
Few shot learning NN 
Currently 2-Class N-shot for pore detection
To apply other images simply place them in two folders in a single directory and modify lines 36 and with the folder names.
The code is set up for CUDA but CPU will work as well. Expect 8GB of GPU memory to be consumed during training.
Few-shot learning is accomplished by training a network to distinguish objects opposed to classifying them. Since the network extracts features to recognize what the object is as well what it is not, transfer learning can be applied to recycle the features on a very small set.

In the presented case, a set cross-section images from Sintering Assisted Additively Manufactured (SAAM) parts composed of Ni were used to train the initial model with the following:
10-shot
Training: 500x2
Validation: 380x2
Testing: 380x2

After training the resulting accuracy is high at 95%. We can then use this same network to train the SS image set with the following:
10-shot
Training: 10x2
Validation: 330x2
Testing: 330x2

The resulting accuracy is similarly at 91% after 100 epochs even though the training size is considerably small. It is also found that without the transfer learning, the model performs significantly less well at 85% at 300 epochs. This the method saves time and increases accuracy.

Some feature that need to be implemented:
1) Selecting the best comparison images 
Outlier selection will considreably decrease performance. We need to find images without a certain stand-deviaion distance from the cluster center. It is anticipated that multiple early training iterations could be utilized for this task.
2) N-Class learning
This can be performed with the current constrastive loss function, but the clustering locations will be arbitrary and may require excessive training. Alternatively the loss function can be modified to include all classes which will then maximize the distance from all classes during a single back propogation (opposed to multiple).
3) Multi-signal learning
It is theorized that the combination of visual and audio signals can be used to increase accuracy. To test this, work is planned to convert the images into audio signals and include these signals with the images during training.
