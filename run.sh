#!/bin/bash

echo "Building Unet and Generating Feature Maps..."
python3 pytorch_unet.py

echo "Training UNet..."
python3 train_unet.py