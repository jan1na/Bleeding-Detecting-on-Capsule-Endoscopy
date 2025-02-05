# Bleeding-Detecting-on-Capsule-Endoscopy
---
## Introduction
This project is to classify capsule endoscopy images into bleeding and healthy. The dataset contains  images with 713 images of bleeding 
and 6161 images of non-bleeding.
---
## Models
We fine-tuned several pretrained models including ResNet50, MobileNetV2, AlexNet and VGG19. The best performing model 
was MobileNetV2 with an accuracy of 0.96.

### (best) MobileNetV2 no augmentation CosineAnnealingLR
#### Accuracy
**Total accuracy**: 0.9606986899563319 \
**Healthy detection**: 591/616 | accuracy: 0.9594155844155844 \
**Bleeding detection**: 69/71 | accuracy: 0.971830985915493
#### Hyperparameters
- Batch size: 32
- Learning rate: 0.001
- Epochs: 20
- Early stopping: 3
- Threshold: 0.5
- Augmentation: no augmentation
- optimizer: Adam
- scheduler: CosineAnnealingLR
- criterion: BCEWithLogitsLoss with penalty for class imbalance
---
## Approaches
### Scheduler
- StepLR
- CosineAnnealingLR
### Batch Size
- 32
- 16
### Augmentation
Tried to overcome the class imbalance by augmenting the bleeding class. However, the performance was not improved.
But if you want to try, you can enable it in using the BleedingDataset with the parameter `apply_augmentation=True`.
That is not all. To use it in the training script model_finetuning.py, you need to uncomment the lines marked with 
`if augmentation used`. First is at the bleeding_weight definition. After that 
you need to uncomment `full_dataset.enable_augmentation()` and `full_dataset.disable_augmentation()` in the training loop
in front of the training and  validation phase. And finally in front of the test phase in the testing space.

#### Unresolvable Issue
The validation and test set also has `AUGMENT_TIMES` times images, but the augmentation is disabled for those images.
So it happens that the same image is used in validation and testing multiple times.
