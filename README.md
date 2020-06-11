<img alt="Birds Collage" src='graphs/bird_collage.png' height="600px" width="1000px" align='center'>

# BIRDEX
### Web based Flask app to predict the family group of birds from images using transfer learning.

![badge](https://img.shields.io/badge/last%20modified-june%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
    - [Birds](#birds)
- [Convolutional Neural Network](#convolutional-neural-network)
- [Birdex: Flask App](#birdex:-flask-app)
- [Summary](#summary)
- [Issues Notes](#issues-notes)
- [Future Work](#future-work)

## Overview



The data was pulled from the [The Cornell Lab of Ornithology](https://www.birds.cornell.edu/home).  
It is a collection of about 48,000 images and more than 400 species of birds observed in North America. Birds are separated by male, female or juvenile since they look quite different. Text files are also included that contains image file names and their corresponding labels.

So why is bird conservation import? Check out this post by the American Bird Conservancy:

[Why Bird Conservation is Important](https://abcbirds.org/about/mission-and-strategy/why-conserve-birds/)

Also, they're basically modern dinosaurs.

<img alt="birdfam" src='graphs/bird-fam-tree.png' height='400px' width='500px' align='right'>

This Berkeley articles on why birds are dinosaurs (but also shows the skeptical side):
[Are Birds Really Dinosaurs?](https://ucmp.berkeley.edu/diapsids/avians.html)




## Data Preparation

Since there are many images, Amazon S3 came into play. The images are loaded into a bucket and stored in separated folders of the bird species.
For this project, 21129 images will be used which includes 39 family groups of birds.

A function is written to retrieve the images from the S3 bucket while also resizing them, convert to array, and append to a list. This is due to the need for the input of the neural network to be numpy arrays.

<details>
  <summary>
    <b> Load and Resize Image Code </b>  
  </summary>
  
```python

def resize_images_array(img_dir, folders, bucket):
    # arrays of image pixels
    img_arrays = []
    labels = []
    
    # loop through the dataframe that is linked to its label so that all images are in the same order
    for folder in tqdm(folders):
        s3 = boto3.client('s3')
        enter_folder = s3.list_objects_v2(Bucket=bucket, Prefix=f'{img_dir}/{folder}')
        for i in enter_folder['Contents'][2:]:
            try:
                filepath = i['Key']
                obj = s3.get_object(Bucket=bucket, Key=f'{filepath}')
                img_bytes = BytesIO(obj['Body'].read())
                open_img = Image.open(img_bytes)
                arr = np.array(open_img.resize((299,299)))
                img_arrays.append(arr)
                labels.append(folder)
            except:
                print(filepath) # get file_path of ones that fail to load
                continue

    return np.array(img_arrays), np.array(labels)

```
</details>

### Birds EDA

The images have 3 different channels for the color which makes up the colors in the main image.
The shape of the images are **(299,299,3)**, the third one represent the number of channels. For greyscale, it'd be 1.

Let's check out some of the contestant within the data!

Contestant 1: Waterfowl    |  Contestant 2: Grosbeak   |     Contestant 3: Hawk
:-------------------------:|:-------------------------:|:-------------------------:
![](graphs/readme_waterfowl.png)| ![](graphs/readme_grosbeak.png) | ![](graphs/hawk1.png)

  
Here are the RGB Channels of three classes of birds seen in this dataset:


<img alt="RGB images" src='graphs/dhf_RGBplot.png' style='width: 600px;'>

The exploratory data analysis began with looking at the number of species in the Order group of the birds.

Since the interest is predicting birds based on family groups, a count plot for the number of species in each family group is created.

<img alt="fam countplot" src='graphs/readme_num_fam_group.png' style='width: 600px;'>


## Convolutional Neural Network

This first model was trained on a small subset (~3,000) of the total images(~40,000). This is mainly to test that the inputs of features and labels are correct. Errors did occur the very first run.

<details>
    <summary>Shape of training sets and testing sets.</summary>
    <img alt="data shapes" src='graphs/data_shapes.png'>
</details>

The model is pretty weak.
<img alt="weak model metrics" src='graphs/model1_bad.png'>

This is what the CNN layers look like generally:
<details>
    <summary>CNN Code</summary>
    <img alt="CNN Code" src='graphs/first_conv_code.png'>
</details>

After the first awful run, a simple model will be created using 3 types of birds: ducks, finches and hawks. This is to see if the amount of classes was causing the model to do so poorly. It will later be expanded to more.

### Simple CNN Model

<details>
    <summary>CNN Model Epochs</summary>
    <img alt="CNN Model epochs" src='graphs/modelx7_epochs.png'>
</details>

<details>
    <summary>CNN Model Accuracy/Loss Plots</summary>
    <img alt="CNN Model acc/loss plots" src='graphs/modelx7_acc_loss_overfit.png'>
</details>

<details>
    <summary>CNN Model Confusion Matrix</summary>
    <img alt="CNN Model conf_mat" src='graphs/modelx_7_conf_mat.png'>
    After a few runs, it finally captured the finches!
</details> 

### Transfer Learning using Xception Model

<details>
    <summary>Model Summary</summary>
    <img alt="Model Summary" src='graphs/transfer_learning.png'>
</details>

<details>
    <summary>Model Epochs</summary>
    <img alt="Model epochs" src='graphs/xception_epoch.png'>
</details>

<details>
    <summary>Model Accuracy/Loss Plots</summary>
    <img alt="Model acc plots" src='graphs/readme_xception_acc.png'>
    <img alt="Model loss plots" src='graphs/readme_xception_loss.png'>
</details>

<details>
    <summary>Model Confusion Matrix</summary>
    <img alt="Model conf_mat" src='graphs/readme_confusion_mat.png'>
</details>

## Birdex: Flask App

## Issues Notes

- birds are labeled by species but also by gender and juvenile/adult. They DO all looke quite different especially the colors between the females and males
- another reason for poor model: birds dont have the same amount of images, some have 20 something, some has 120
    - A TON of labels (555 total), very sparse, along with unbalanced amounts of bird images
    - checked inputs, y labels and x labels
    - checked images folders, different amounts of bird images
    - checked slicing and what images i am getting, turns out i could be slicing where each bird only has one image
        - fix by grabbing sequentially because all the birds in one folder are next to each other in dataframe
- model was awful, figured out one hot encoded the wrong numbers due to the fact that some numbers are missing and not in a perfect range


## Future Work

- [x] Better Model
- [x] Transfer Learning
- [ ] SHAP/LIME
- [ ] Clean up files
- [ ] Object Detection