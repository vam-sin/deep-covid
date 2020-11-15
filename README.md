# Deep Covid

A deep learning tool for the detection of COVID-19 by analyzing Chest-Xrays

Five different State-Of-The-Art Convolutional Neural Network models were developed on the merged dataset of COVID-19 chest x-ray images. The networks are:

1. SqueezeNet
2. MobileNetv2
3. DenseNet 201
4. CheXNet
5. VGG 19

Two class classification of the images into COVID-19 and Non-COVID-19 was conducted and the two best performing models were the MobileNetv2 followed by the VGG 19. Two different fusion models, one with the features and the other with the predictions was developed.

To run the final fusion model, run the following code.

```python3
python3 feature_ensemble.py
```
