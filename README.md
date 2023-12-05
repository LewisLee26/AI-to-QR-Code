# AI-to-QR-Code

## Overview
This project explores the limits of model compression and representation. The goal is to shrink a trained MNIST (handwritten digit recognition) model to fit within the constraints of a QR code. 

## Current Model
The model is made up of two convolutional layers and one dense layer.
It achievies a 89% accuracy on the MNIST dataset.

![](model.png)

*This QR-code doesn't currently work due to a bug in the qrcode python encoding module corrupting the gzip binary* 

