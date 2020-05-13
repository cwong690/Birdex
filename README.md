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

- <a href="https://github.com/cwong690/bird-classifcation">Introduction</a> 
- <a href="https://github.com/cwong690/bird-classifcation">Data Preparation</a> 

- <a href="https://github.com/cwong690/bird-classifcation">Neural Network</a> 
- <a href="https://github.com/cwong690/bird-classifcation">Summary and Key Findings</a>


## Introduction

<!-- The data was pulled from the [The National UFO Reporting Center Online Database](http://www.nuforc.org/webreports.html).   -->


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

## Planned Process

- load into s3 (many images)
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


<!-- <img alt="shapes" src='' style='width: 600px;'> -->


## Neural Network




## KNN
<!-- 
<img alt="" src=''>

<img alt="" src=''> -->

## Summary and Key Findings


<!-- <img alt="" src=''> -->