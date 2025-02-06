# Bleeding Detection on Capsule Endoscopy

![Sample Image](path/to/your/image1.png)

## Overview

This repository presents a comprehensive approach to detecting bleeding in capsule endoscopy images, combining deep learning models with classical image processing techniques. The project aims to enhance diagnostic accuracy and provide a user-friendly interface for medical practitioners.

## Features

- **Deep Learning Models:** Implementation of fine-tuned MobileNetV2 and ResNet50 architectures for image classification tasks.
- **Classical Image Processing:** Utilization of redness-based analysis to identify potential bleeding regions.
- **Graphical User Interface (GUI):** An intuitive interface allowing users to upload images, select the desired detection model, and view results seamlessly.
- **Data Augmentation:** Application of various augmentation techniques to address class imbalance and improve model robustness.
- **Comprehensive Evaluation:** Assessment using metrics such as precision, recall, F1-score, and area under the precision-recall curve.

## Dataset

The dataset comprises two primary categories:

- **Bleeding Images:** Frames exhibiting signs of bleeding.
- **Healthy Images:** Frames without any bleeding indications.

To mitigate class imbalance, strategies like oversampling and the implementation of weighted loss functions were employed.

![Data Distribution](path/to/your/image2.png)

## Installation

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```
## Usage
Running the GUI
Launch the graphical interface with the following command:

```bash
python gui.py
```
Within the GUI, you can upload an image, choose between the deep learning or classical detection methods, and receive immediate predictions.

## Training Deep Learning Models
To train the models, execute:

```bash
python train.py --model mobilenetv2  # For MobileNetV2
python train.py --model resnet50     # For ResNet50
```
Model Performance
MobileNetV2
Scheduler: Utilized CosineAnnealingLR.
Batch Size: 32.
Accuracy: Achieved an accuracy of 96.07%.
ResNet50
Performance: Demonstrated variability in accuracy based on augmentation techniques and scheduler configurations.

## Classical Method
The classical approach calculates a "redness score" by analyzing weighted RGB channel values.

Optimal Threshold: 62,859.
Evaluation Metrics: Precision, recall, F1-score, and precision-recall curves were used for assessment.

## Future Work
Enhancement of Classical Methods: Aim to improve robustness and accuracy.
GUI Integration: Incorporate classical detection methods into the graphical interface.
Architectural Exploration: Experiment with alternative deep learning architectures to further boost performance.
## Contributors
Meriç Demirörs
Yusuf Bera Demirhan
Janina Fritzke
Waleed Ahmed Shahid
