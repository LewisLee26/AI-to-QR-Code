# AI-to-QR-Code

## Overview
This project explores the limits of model compression and representation. The goal is to shrink a MNIST (handwritten digit recognition) model to fit within the constraints of a QR code (2,953 bytes). 

## Current Model
The model is made up of two convolutional layers and one dense layer.
When fully compressed the model is 2,640 bytes.
It achievies a 89% accuracy on the MNIST dataset.

![Model QRcode PNG](model.png)

*This QR-code doesn't currently work due to a bug in the qrcode python encoding module that is corrupting the gzip binary* 

