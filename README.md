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
But if you want to try, you can enable it by setting the hyperparameter APPLY_AUGMENTATION to True.

#### Unresolvable Issue
The validation and test set also has `AUGMENT_TIMES` times images, but the augmentation is disabled for those images.
So it happens that the same image is used in validation and testing multiple times.
