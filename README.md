![Duck, Duck, HAWK](graphs/bird_collage.png)

# Duck, Duck, HAWK

Capstone II goals

- In all cases, discussion of the cleaning and featurization pipeline and how raw data were transformed to the data trained on. Text processing especially requires discussion.
- In the case of classification and class imbalance, discussion of how the class imbalance was addressed. Did you use a default decision threshold, or did you pick a different threshold through out-of-model knowledge (e.g. a cost-benefit matrix and a profit curve.)

![badge](https://img.shields.io/badge/last%20modified-may%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Birds](#birds)
- [Convolutional Neural Network](#convolutional-neural-network)
- [Summary](#summary)
- [Issues Notes](#issues-notes)
- [Future Work](#future-work)



blargb;arg;abf




## Overview

The data was pulled from the [The Cornell Lab of Ornithology](https://www.birds.cornell.edu/home).  
It is a collection of about 48,000 images and more than 400 species of birds observed in North America. Birds are separated by male, female or juvenile since they look quite different. Text files are also included that contains image file names and their corresponding labels.


## Data Preparation

Since there are many images, Amazon S3 came into play. The images are loaded into a bucket and stored in separated folders of the bird species.
While the original goal is to classify around 555 species of birds with more than 40,000 images, as it will be shown later, it was not possible for now. The pivoted project goal is to identify 3 different types of birds: **ducks, finches, and hawks**.

A function is written to retrieve the images from the S3 bucket while also resizing them, convert to array, and append to a list. This is due to the need for the input of the neural network to be numpy arrays.

<details>
  <summary>
    <b> Load and Resize Image Code </b>  
  </summary>
  
```python
code

```


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

<details>
    <summary>summary</summary>
    <img alt="Data" src=''>
</details>
    
<details>
    <summary>summary</summary>
    <img alt="Data" src=''>
</details>    
    
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

<img alt="shapes" src='' style='width: 600px;'>

## Birds


<img alt="shapes" src='' style='width: 600px;'>


## Convolutional Neural Network

The one of the first models tested was on a small subset (~3,000) of the total images(~40,000). This is mainly to test that the inputs of features and labels are correct. Errors did occur the very first run.

Shape of training sets and testing sets.
<img alt="data shapes" src='graphs/data_shapes.png'>
    
This returned some pretty disturbing metrics which is was the turning point for the project goal.
<img alt="data shapes" src='graphs/data_shapes.png'>


## Metric Visualizations

<img alt="" src=''>

## Issues Notes

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


## Future Work

- KNN
- More birds
- Better Model
- TensorBoard
- Transfer Learning
- SHAP