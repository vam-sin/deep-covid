# Deep Covid

A deep learning tool for the detection of COVID-19 by analyzing Chest-Xrays

# Requirements

```python3
pip3 install tensorflow
pip3 install keras
```

# Data

In this study we present the DeepC19x dataset which was derived from 8 different publicly available datasets consisting of chest x-ray images. The DeepC19x dataset has a total of 1,104 COVID-19 chest x-ray images, 6,624 chest x-ray images of pneumonia and 6,624 normal chest x-ray images. The datasets used are as follows:

1. COVID-19 Chest X-ray Database 

The COVID-19 Chest X-ray Database is publicly available on Kaggle, designed by researchers at the Qatar University. This dataset has images belonging to three different classes, COVID-19, Normal and Pneumonia. To create the DeepC19x dataset we only chose the COVID-19 images from this dataset which were 219 in number.

2. SIRM COVID-19 Database 

The Società Italiana di Radiologia Medica e Interventistica (SIRM) has published an online report of COVID-19 cases. These case reports consist of various chest x-ray and Computed Tomography (CT) scan images. A total of 98 chest x-ray images were obtained from all the COVID-19 positive cases (Dated 25th July, 2020).

3. Figure1 COVID Chest X-Ray Dataset

The Figure 1 COVID-19 Chest X-ray Dataset Initiative was started by the COVID-Net team as a part of building the COVIDx dataset. 55 COVID-19 chest x-ray images were obtained from this Github repository (Dated 25th July, 2020). This dataset consisted of only COVID-19 chest x-ray images.

4. Actualmed COVID Chest X-Ray Dataset 

The Actualmed COVID-19 Chest X-ray dataset initiative was also started by the COVID-Net team as a part of building the COVIDx dataset. A total of 238 COVID-19 chest x-ray images were obtained from this Github repository (Dated 25th July, 2020). This dataset consisted of only COVID-19 chest x-ray images.

5. COVID Chest X-Ray Dataset  

The COVID chest x-ray dataset was published by Joseph Paul Cohen. This Github repository has images pertaining to COVID-19, Viral and Bacterial Pneumonia along with ARDS and SARS. A total of 466 COVID-19 chest x-rays were obtained from this repository (Dated 25th July, 2020).

6. Twitter Chest Imaging Dataset 

A Twitter thread created by the user ChestImaging has provided with COVID-19 chest x-rays from a hospital in Spain. A total of 28 images were obtained from this Twitter thread (Dated 25th July, 2020). 

7. NIH Chest X-rays 

The National Institute of Health (NIH) has published this dataset on Kaggle. This dataset has a total of 112,120 chest x-ray images for 12 different diseases. 4,999 normal and 6,624 pneumonia chest x-ray images were extracted from this dataset. 

8. Chest X-Ray Images (Pneumonia)

This Kaggle dataset was published by Paul Mooney. A total of 5,863 chest x-ray images pertaining to pneumonia and normal circumstances are present in this dataset. 1,625 normal images were extracted from this dataset. 

# Models

Five different State-Of-The-Art Convolutional Neural Network models were developed on the merged dataset of COVID-19 chest x-ray images. The networks are:

1. SqueezeNet
2. MobileNetv2
3. DenseNet 201
4. CheXNet
5. VGG 19

Two class classification of the images into COVID-19 and Non-COVID-19 was conducted and the two best performing models were the MobileNetv2 followed by the VGG 19. Two different fusion models, one with the features and the other with the predictions was developed.

To run the final fusion model, run the following code.

```python3
python3 feature_ensemble_c2.py
```
