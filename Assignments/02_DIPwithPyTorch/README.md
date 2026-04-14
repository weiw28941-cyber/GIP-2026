# Assignment 2 - DIP with PyTorch

This repository contains the implementation of Assignment 02: Image-to-Image Translation using PyTorch. The goal is to convert architectural semantic labels into photorealistic RGB images using the Pix2Pix framework (FCN-based).

## 1. Requirements

Before running the project, ensure your environment meets the following specifications:

* **Python**: 3.8+
* **Virtual Environment**: Miniconda (Environment name: `fcn_train`)
* **Core Dependencies**: 
    * `torch` (CUDA version recommended for GPU acceleration)
    * `opencv-python`
    * `numpy`

## 2. Dataset Preparation

This project utilizes the [Berkeley Facades Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz).

1.  **Download & Extract**: Extract the dataset into the `./datasets/facades/` directory.
2.  **Generate Index Files**: To ensure the scripts correctly locate the image paths, run the following PowerShell commands in the project root:
    ```powershell
    # Generate Training List
    Get-ChildItem "./datasets/facades/train/*.jpg" | Select-Object -ExpandProperty FullName > train_list.txt
    # Generate Validation List
    Get-ChildItem "./datasets/facades/val/*.jpg" | Select-Object -ExpandProperty FullName > val_list.txt
    ```

## 3. Training Procedure

Execute the following command in your terminal to start training:

```bash
# Ensure you are in the project root directory
python train.py
## 4. Results and Metrics

### Training Logs Summary
| Phase | Epoch | Training L1 Loss | Validation L1 Loss | Visual Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Initial** | 30 | ~0.37 | ~0.41 | Basic color blocks identified; edges are extremely blurry. |
| **Mid-term** | 130 | ~0.28 | ~0.40 | Structures are accurate; severe over-smoothing present. |
| **Final** | 300 | **~0.24** | **~0.40** | Stable color tones; missing high-frequency textures. |

### Visual Results
As observed in the `./train_results` folder, the model successfully learns the spatial mapping (e.g., placing windows and doors correctly). However, the output images exhibit an "out-of-focus" effect, which is a characteristic limitation of the FCN + L1 Loss combination.

## 5. Critical Analysis

Regarding the phenomenon where **"Loss is low, but visual quality is blurry"**:

1.  **Averaging Effect of L1 Loss**: L1 loss minimizes the mean absolute error. When the model is uncertain about the exact location of a sharp edge, it outputs the "statistical average" of possible pixel values to minimize the penalty, resulting in blurriness.
2.  **Structural Bottleneck**: The FCN architecture lacks **Skip Connections**. Low-level spatial details (like sharp lines and textures) are lost during the downsampling process and cannot be recovered during upsampling, leading to a loss of high-frequency information.
3.  **Generalization Gap**: While the Training Loss decreases after the learning rate decay (Epoch 200+), the Validation Loss plateaus around 0.40. This indicates the model has reached its representational capacity for unseen data.

## 6. Future Improvements
* **U-Net Architecture**: Implement skip connections to preserve low-level feature maps.
* **Generative Adversarial Networks (GANs)**: Introduce a Discriminator and a GAN loss (Adversarial Loss) to force the Generator to produce sharper, more realistic textures.

---
*DIP-2026 Assignment Report*
