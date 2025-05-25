# Bleeding Detection on Capsule Endoscopy

![Sample Image](path/to/your/image1.png)

## Installation
Ensure you have Python installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Warning
Results are not meaningful, because the patients in the train and test data did not get separeted which could result in data leakage.

## Introduction
This project aims to classify capsule endoscopy images into bleeding and healthy categories. The dataset consists of:
- **713 bleeding images**
- **6161 non-bleeding images**

## Features
- **Deep Learning Models:** Fine-tuned ResNet50, MobileNetV2, AlexNet, and VGG19 for image classification.
- **Classical Image Processing:** Redness-based analysis to detect bleeding.
- **Graphical User Interface (GUI):** A user-friendly interface for easy interaction.
- **Data Augmentation:** Attempts to address class imbalance.
- **Comprehensive Evaluation:** Using precision, recall, F1-score, and area under the precision-recall curve.

## Dataset
The dataset is divided into two categories:
- **Bleeding Images**
- **Healthy Images**

To handle class imbalance, oversampling techniques and weighted loss functions were applied.

![Data Distribution](path/to/your/image2.png)

## Usage
### Running the GUI
```bash
python gui.py
```
Upload an image, select a model, and get the prediction.

### Running Deep Learning Models
Run the training script in Google Colab using `model_finetuning.ipynb`. Open the notebook, configure the desired model parameters, and execute all cells.

## Models
Several pretrained models were fine-tuned, including:
- ResNet50
- MobileNetV2
- AlexNet
- VGG19

The best-performing model was **MobileNetV2**, achieving an accuracy of **96.07%**.

### Best Model: MobileNetV2
**Configuration:**
- **No augmentation**
- **Scheduler:** CosineAnnealingLR
- **Optimizer:** Adam
- **Criterion:** BCEWithLogitsLoss with penalty for class imbalance
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Epochs:** 20
- **Early Stopping:** 3
- **Threshold:** 0.5

#### Accuracy
- **Total accuracy:** 0.9607
- **Healthy detection:** 591/616 (**95.94%**)
- **Bleeding detection:** 69/71 (**97.18%**)

## Model Performance
### Hyperparameter Tuning
Different approaches were explored to improve performance:
- **Schedulers:** StepLR, CosineAnnealingLR
- **Batch Sizes:** 16, 32
- **Augmentation:** Applied to bleeding class, but did not improve performance. 

### Augmentation Usage
Augmentation did not improve the results, but if you want to try it, just set *APPLY_AUGMENTATION* in the `model_finetuning.ipynb` to True. Of course you can also adjust the *AUGMENT_TIMES* parameter, that decides how many times the bleeding images are augmented. But by leaving it on 8, you balance out the bleeding and healthy images almost perfectly: `713 * 8 = 5.704 ~ 6161`. If you want to oversample the bleeding images you can instead choose 9 augmentations.

### Unresolvable Issue
The validation and test sets contain multiple copies of images due to class augmentation, leading to repeated image evaluations. The validation and test sets still contain `AUGMENT_TIMES` times more images, even though augmentation is disabled for them. This issue remains unresolved.

![Model Performance Comparison](path/to/your/image3.png)

## Classical Method
This method calculates a "redness score" by analyzing weighted RGB channel values.
- **Optimal Threshold:** 62,859
- **Evaluation Metrics:** Precision, recall, F1-score, and precision-recall curves.

![Redness Score Distribution](path/to/your/image4.png)

## Training Environment
The deep learning models were trained using **Google Colab** with the script `model_finetuning.ipynb`.

## Future Work
- **Enhancement of Classical Methods:** Aim to improve robustness and accuracy.
- **GUI Integration:** Incorporate classical detection methods into the graphical interface.
- **Architectural Exploration:** Experiment with alternative deep learning architectures to further boost performance.
- **Saving hyperparemters:** Save a file with all hyperpareters, together with the trained model. Currently only the model.pth, its accurcy and a plot of the train and validation loss is saved.

## Contributors
- Meriç Demirörs
- Yusuf Bera Demirhan
- Janina Fritzke
- Waleed Ahmed Shahid

## License
This project is licensed under the MIT License.

