# Fine-Grained Image Classification on CUB Birds, Combined Dataset, and FoodX

## Abstract
This repository contains the implementation of a comprehensive approach to fine-grained image classification applied to three distinct datasets: CUB birds, a combined dataset of CUB birds and FGVC Aircraft, and the FoodX dataset. The goal is to enhance classification performance without significantly increasing the computational requirements of the baseline model, ConvNeXt-V2-Large, by more than 5%. This implementation showcases how strategic adjustments and the incorporation of Selective Channel Recalibration Attention (SCRA) can improve accuracy while adhering to computational constraints.

## Introduction
Fine-grained image classification poses a unique challenge in the field of image analysis due to the high intra-class variance and low inter-class variance. This project leverages a pre-trained baseline model, fine-tuning it for specific tasks across different datasets to assess its adaptability and accuracy. The datasets evaluated include the CUB birds, a combined dataset of CUB birds and FGVC Aircraft, and the FoodX dataset. This work explores various modifications, from data augmentation to architectural adjustments, to improve classification performance.

## Dataset
- **CUB Birds**: 200 classes.
- **Combined Dataset (CUB Birds + FGVC Aircraft)**: 300 classes.
- **FoodX**: 251 classes.

## Results
Our methodologies yielded significant improvements across all tasks:

- **CUB Birds Dataset**:
  - **Baseline Accuracy**: 89.49%
  - **Improved Accuracy**: 90.40%

- **Combined Dataset (CUB Birds + FGVC Aircraft)**:
  - **Baseline Accuracy**: 89.48%
  - **Improved Accuracy**: 89.99%

- **FoodX Dataset**:
  - **Baseline Accuracy**: 78.62%
  - **Improved Accuracy**: 79.44%

These results demonstrate the effectiveness of our approach, showing marked improvements in classification accuracy across all evaluated datasets without substantial increases in computational demands.

