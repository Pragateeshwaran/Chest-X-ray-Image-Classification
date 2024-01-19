# Chest X-ray Image Classification

This project focuses on classifying chest X-ray images into two categories: NORMAL and PNEUMONIA. It utilizes deep learning techniques implemented with TensorFlow and Keras.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)


## Introduction

This project aims to develop a deep learning model for classifying chest X-ray images into two categories: NORMAL and PNEUMONIA. The goal is to assist in the early detection of pneumonia cases using automated image analysis.

## Dataset

The dataset is sourced from chest X-ray images and is divided into training, validation, and test sets. The images are organized into directories for NORMAL and PNEUMONIA classes. Preprocessing steps include data augmentation techniques to enhance the model's ability to generalize.

## Model Architecture

The neural network architecture used for image classification includes convolutional layers, max-pooling, dropout, and dense layers. The model is implemented using TensorFlow and Keras.

```python
# Model Summary
Model: "sequential"

Total params: 550,578
Trainable params: 550,578
Non-trainable params: 0
```

## Training

The model is trained using the training dataset, with data augmentation to improve robustness. The training process involves multiple epochs, and key hyperparameters are configured for optimal performance.

```python
# Model Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25
)
```

## Evaluation

The model's performance is evaluated on the validation and test datasets, measuring metrics such as accuracy and loss.

```python
# Model Evaluation
model.evaluate(test_ds)
# Output: [0.5589020848274231, 0.8301281929016113]
```

## Usage

To use the trained model for predictions, you can load it and apply it to new chest X-ray images.

```python
# Model Usage
from tensorflow.keras.models import load_model

loaded_model = load_model('final.h5')
# Make predictions using the loaded_model
```
