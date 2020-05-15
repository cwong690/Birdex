![HITMAKA HEADER](image)

# Bird Classification

Capstone II goals

- An MVP that demonstrates supervised or unsupervised learning (and maybe both).
- In the case of supervised learning, picking an appropriate metric to quantify performance, and then use of that metric in cross-validation to arrive at a model that generalizes as well as possible on unseen data. Be prepared for the request: "Describe the process you used to ensure your model was properly fit."
- In the case of unsupervised learning, picking an appropriate clustering method and metric to guide the choice of the number of hard or soft clusters. This should be followed up with a thoughtful discussion as to why the chosen number of clusters is "correct."
- In all cases, discussion of the cleaning and featurization pipeline and how raw data were transformed to the data trained on. Text processing especially requires discussion.
- In the case of classification and class imbalance, discussion of how the class imbalance was addressed. Did you use a default decision threshold, or did you pick a different threshold through out-of-model knowledge (e.g. a cost-benefit matrix and a profit curve.)

![badge](https://img.shields.io/badge/last%20modified-may%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)

## Table of Contents

- [Overview](#basic-overview)
  - [Context](#context)
  - [Goal](#goal)
- [Exploring Data](#exploring-data)
  - [Web Scraping](#web-scraping)
  - [Initial Intake](#initial-intake)
- [Language Processing](#language-processing)
  - [Tokenizing](#tokenizing)
  - [Visualizations](#visualizations)
- [**K O D A K  B O T  3 0 0 0**](#k-o-d-a-k--b-o-t--3-0-0-0)
- [Future Considerations](#future-considerations)
- [License](#license)

- <a href="https://github.com/cwong690/bird-classifcation">Introduction</a> 
- <a href="https://github.com/cwong690/bird-classifcation">Data Preparation</a> 

- <a href="https://github.com/cwong690/bird-classifcation">Neural Network</a> 
- <a href="https://github.com/cwong690/bird-classifcation">Summary and Key Findings</a>


## Overview

<!-- The data was pulled from the [The National UFO Reporting Center Online Database](http://www.nuforc.org/webreports.html).   -->

Concepts and Codes:

- BytesIO: reads bytes objects
    - s3 object bodies come back as a byte string, BytesIO helps read that

Images:

- contains around 50,000 images and almost 700 different species

Text Files:

- image_class_labels.txt: contains all images and the folder number it is in
- hierarchy.txt: contains the folder number and the class number(which kind of bird it is)
- classes.txt: contains the class number and what type of bird that is

- images.txt: comtains the image names without .jpg extension and the image file paths (starting with the folder names)

Day 1:

- watch tutorials
- read tutorials
- figure out models desired for training
- download new birds
- try to load into jupyter lab
- create bird images with RBG sets

Day 2:

- AWS
- AWS
- AWS
- AWS
- AWS
- figure out how to set up EC2 instance, connect an Elastic IP address, S3 bucket
- s3 keeps crashing, need a new way to load in the numerous birds

Day 3:

- create dataframes with text files that contains information about the labels and the folders it corresponds to
- figure out how to merge all as one for easy access
- figure out how to use as labels
- create dataframes of image information
- test small set of loading in images and displaying
    - saved as array
- upload all images to s3


## Planned Process

- load into s3 (many images) DONE
- use small sets of images first
- resize images (so model dont take too long)
- align images with labels
- get arrays
- normalize (keras normalize or simply divide by 255)
- make model
    - Sequential()
    - Flatten layer of inputs
    - dense layers of hidden layers
    - final dense layer: output
    
1. Baseline Model
- Random Forest

2. Deep Learning
- Neural Network
- CNN

## ISSUES/MISTAKES

- birds are labeled by species but also by gender and juvenile/adult. They DO all looke quite different especially the colors between the females and males
- A TON of labels (555 total), very sparse
- tried drag and drop with s3
- accidentally placed arguments in wrong spot and ran 50k images fail message
- another reason for hot garbage: birds dont have the same amount of images, some have 20 something, some has 120
    - checked inputs, y labels and x labels
    - checked images folders, different amounts of bird images
    - checked slicing and what images i am getting, turns out i could be slicing where each bird only has one image
        - fix by grabbing sequentially because all the birds in one folder are next to each other in dataframe
- model was awful, figured out one hot encoded the wrong numbers due to the fact that some numbers are missing and not in a perfect range

## Data Preparation and Exploratory Data Analysis

<!-- <details>
    <summary>summary</summary>
    <img alt="Data" src=''>
</details>
    
<details>
    <summary>summary</summary>
    <img alt="Data" src=''>
</details>     -->
    
<br> 

<!-- wesley's op drop down -->
<details>
  <summary>
    <b> Model Comparison Code </b>  
  </summary>
  
```python
code

```
  

</details>



<!-- <img alt="shapes" src='' style='width: 600px;'> -->


## Convolutional Neural Network

<!-- 
<img alt="" src=''>

<img alt="" src=''> -->

## Summary and Key Findings


<!-- <img alt="" src=''> -->

## Future Work

- KNN
- Better Model
- TensorBoard
- Transfer Learning
- SHAP