# Ground Truth Segmentation with GAN Model

## Notes
Clone the project on your machine with :

```
git clone https://github.com/zakariamejdoul/ground-truth-gan-segmentation.git
```

## Behaviour

Image segmentation is an image processing operation that aims to group pixels together according to predefined criteria. The pixels are thus grouped into regions, which constitute a paving or a partition of the image.<br><br>
Generative adversarial networks (GANs) are a class of unsupervised learning algorithms. These algorithms were introduced by Goodfellow et al. 2014. They are used to generate images with a high degree of realism.
<br><br>In this project, we will apply the DCGAN approach for ground truth segmentation operation of satellite images.

## Dataset

The dataset used is AIRS ([Aerial Imagery for Roof Segmentation - Resized Version)](https://www.kaggle.com/atilol/resized-aerialimageryforroofsegmentation) is a public dataset that aims to compare roof segmentation algorithms from very high resolution aerial images. 
The main characteristics of AIRS can be summarized as follows:
* Coverage of 457 km2 of orthorectified aerial imagery with more than 220,000 buildings
* Very high spatial resolution of the imagery (0.075m)
* Refined ground truths that strictly align with the roof contours

To reduce processing time and given the limited resources we have, we opted for a resized version of the database from (10000px × 10000px × 3) to (1024px × 1024px × 3), and for the dimensional constraint of the Pix2Pix model input we resized all images and labels to (256px × 256px × 3). 
<br><br>Summary on the dataset: 
* Number of images: 875(train) + 95(test) + 94(validation)
* Image size: (256 × 256 × 3)
* Each folder contains the images and the roof labels

The following figure shows an example of an image with its label :

![alt image](static/image_label.PNG)

## Data Processing (Image Loading)

Data preprocessing is a data mining technique that is used to transform raw data into a useful and efficient format.
<br>In our case, the following processing was applied:
* Image loading: The `load_image()` function is used to load, resize, and separate the dataset (train, test, validation); each dataset subset contains the satellite images and their labels (grounds truth).
* Contour extraction: The function `Extract_Contour()` with image processing operations (erosion and dilation) extracts the contours of the labels (of the binary images) that represent the footprint of the buildings, and then draws the extracted contours on the binary images or the labels.

The following figure shows an example of an image with its contoured label :

![alt image](static/image_label_contour.png)

## Model Architecture

### Model based on Pix2Pix architecture

Pix2Pix is based on conditional generative adversarial networks (CGAN) to learn a mapping function that maps an input image into an output image. <br>Pix2Pix like GAN, CGAN is also composed of two networks, the generator and the discriminator. The figure below shows a very high level view of the Pix2Pix architecture:

![alt image](static/pix2pix.PNG)

### The Generator Architecture

The purpose of the generator is to take an input image and convert it into the desired image (output or footprint) by implementing the necessary tasks. There are two types of generators, including the encoder-decoder and the U-Net. The final difference is to have skipped connections.
<br>Encoder-decoder networks translate and compress the input images into a low-dimensional vector presentation (Bottleneck). Then the process is reversed and the multitude of low-level information exchanged between the input and output can be used to execute all the necessary information through the network.<br>In order to bypass the Bottleneck part of the information, they added a hop connection between each layer i and n-i, where i is the total number of layers. Note that the shape of the generator with the hop connections looks like a U-Net. These images are shown below:

![alt image](static/encoder_decoder.png)

>**As can be seen in the U-Net architecture, information from earlier layers will be integrated into later layers and, thanks to the use of hop connections, they require no size changes or projections.**

### The Discriminator Architecture





