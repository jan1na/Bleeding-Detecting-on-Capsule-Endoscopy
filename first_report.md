# Project Planning

The main flow of the project is decided on, technical requirements are discussed, and an estimated timeline is created.

## Main Flow of the Project
---
### Models to Experiment With and Parameter Counts:
- LeNet-5 (60K), GoogLeNet (5M), ResNet50 (26.5M), AlexNet (60M), VGG19 (138M)  
- Transfer learning, where first few layers are frozen, will also be tested.  

#### Reasons:
1. To see the performance difference between various sized models.
2. To check the transfer learning's effect on the task.

---

### Pre-Process Blocks to Use:
- Color conversion to RGB, HSV, YUV, and YCbCr  

#### Reasons:
1. To check the influence of color information.

---

### Augmentations to Use:
- Zoom, Rotate, Intensity Changing, Noise  
- Different influence scales will be tested  

#### Reasons:
1. To see the performance boost coming from different scales of augmentations.

---

### Metrics to Measure With:
1. Accuracy, Precision, Recall, F1-Score, Area Under ROC Curve, Area Under Precision-Recall Curve  

#### Reasons:
1. To display performances from different points of view.


## Technical Requirements:
---
1. A NVIDIA GPU is available for the first experiments. If anything goes wrong or the process is slow, a secondary option will be tested:  
2. The dataset is small enough to store on Google Drive, making Google Colab usage feasible.


## Estimated Timeline:
---
| **Date**       | **Milestone**                                      |
|-----------------|---------------------------------------------------|
| **January 15**  | Finalizing the main planning of the project        |
| **January 22**  | Finalizing the dataset inspection and project codes |
| **January 29**  | Finalizing the deep learning trainings             |
| **January 29**  | Finalizing the classical method experiments        |
| **February 5**  | Finalizing the GUI implementation                 |
| **February 12** | Finalizing the final presentation                 |

