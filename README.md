# Alzheimer-s-Detection
Built a CNN in TensorFlow/Keras to classify MRI scans into 4 Alzheimer’s stages. Addressed class imbalance with SMOTE and enhanced robustness using batch norm, dropout, and LR scheduling. Achieved strong accuracy and AUC on the Kaggle Alzheimer’s dataset.

# Early Diagnosis of Alzheimer's Disease using Deep Learning

This repository contains a reproducible implementation of the project **Early Diagnosis of Alzheimer’s Disease using Deep Learning** (original project report and code from Sathyabama Institute). The model classifies MRI images into four categories: `NonDemented`, `VeryMildDemented`, `MildDemented`, and `ModerateDemented`.

## Key features
- Convolutional Neural Network (custom CNN blocks) implemented in TensorFlow / Keras.
- Imbalance handling via SMOTE after flattening image tensors.
- Training utilities: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- Plots for accuracy, loss and AUC, plus final model export
