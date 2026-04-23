# SAGA: Spatially-Adaptive Gated Activation for Medical Image Restoration

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official `PyTorch` implementation and supplementary codebase for the manuscript: **An Interpretable Deep Learning Method for Medical Image Deblurring and Restoration**.

## Overview
This repository provides the complete, reproducible pipeline for the **Spatially-Adaptive Gated Activation (SAGA)** operator. It includes the synthetic degradation modeling, modular CNN backbones, training routines, and the rigorous Explainable AI (XAI) statistical frameworks used to validate the architecture's preservation of high-frequency anatomical boundaries.

> **Peer-Review Note:** This repository has been structured to ensure complete methodological transparency and reproducibility, directly addressing the architectural and interpretability evaluations requested during the peer-review process.

---

## 1. Repository Structure

```text
SAGA_Supplementary_Code/
│
├── README.md                  # This document
├── requirements.txt           # Python package dependencies
│
├── generate_dataset.py        # Phase 2: Degradation & Patch Extraction Pipeline
├── train.py                   # Phase 4: Training & Hyperparameter Optimization
├── xai_analysis.py            # Phase 5 &7: True LRP, K-Paths & Statistical Significance
├── evaluate.py                # Phase 6 part-I: Quantitative Metric Evaluation
├── clinical_validation.py     # Phase 6 part-II: Evaluation on real patient data
|
├── data/
│   ├── CT_final.zip           # (Place your Chest CT source zip here)
│   └── Osteoporosis_final.zip # (Place your DXA source zip here)
│
└── models/
    ├── __init__.py            
    ├── saga_layer.py          # Contains proposed SAGA, FReLU, and baselines
    ├── unet.py                # U-Net Architecture
    ├── resnet.py              # DeblurResNet Architecture
    ├── edsr.py                # EDSR Architecture
    └── vggnet.py              # Plain VGG Baseline Architecture
```

## 2. Installation & Setup

All experiments were conducted using `Python 3.10` and `PyTorch 2.0+` on `NVIDIA RTX 3090` GPUs. To set up the exact environment, run:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)sijuswamyresearch/SAGA.git
cd SAGA_Supplementory_Code

# Install dependencies
pip install -r requirements.txt
```

## 3. Reproducing the Pipeline

>**Step 1: Data Access & Preparation (Phase 2)**
Due to GitHub file size limits and licensing policies, the raw medical imaging datasets are not included directly in this repository. To reproduce the synthetic degradation pipeline, please follow these steps:

**A. Download the Public Datasets**

1. **Chest CT Dataset:** Download and extract the *Lung Cancer Classification* (Syed et al., 2024) from Mendeley Data: [https://data.mendeley.com/datasets/5kbjrgsncf](https://data.mendeley.com/datasets/5kbjrgsncf))[^1] .

[^1]: An extracted form of this dataset is now also available at [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images]
(https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images).


2. **Osteoporosis DXA Dataset:** Download the *Knee X-ray Osteoporosis Database* (Wani et al., 2021) from Mendeley Data: [https://data.mendeley.com/datasets/fxjm8fb6mw/2](https://data.mendeley.com/datasets/fxjm8fb6mw/2)

**B. Structure the Archives**
To match the experimental setup detailed in the manuscript (which partitions 1000 source images into 800 train / 100 validation / 100 test), you must create two zip archives:

1. Extract the downloaded datasets to your local machine.
2. Select **1000** representative 2D image slices (`.jpg`, `.png`, or `.tif`) from the CT dataset and compress them into a single archive named `CT_final.zip`.
3. Select **1000** images from the Osteoporosis dataset and compress them into a single archive named `Osteoporosis_final.zip`.

>*Note:* The images can be at the root of the `.zip` or inside a single subfolder; the generation script will recursively locate them.

**C. Generate the Training Data**

Move both `CT_final.zip` and `Osteoporosis_final.zip` into the `data/` directory of this repository. Then, run the unified generator to simulate the clinical artifacts (Gaussian, Motion, Defocus) and extract the $256 \times 256$ spatial patches:

```bash
# Generate the Chest CT Dataset
python generate_dataset.py --dataset CT

# Generate the Osteoporosis DXA Dataset
python generate_dataset.py --dataset Osteoporosis
```

>**Step 2: Training & Evaluation (Phase 4 & 5)**

To train the architectures across the evaluation matrix, execute `train.py`. The script will automatically load the generated datasets, train the models, save the best weights to models/, and output evaluation metrics (PSNR, SSIM, EPI, LPIPS) to results/.

>*Note:* Ensure you update the `MODEL_CHOICE` and `DATASET_NAME` variables in the configuration section of `train.py` prior to execution.

>**Step 3: Real-World Clinical Validation (Phase 6)**

To perform perceptual quality assessment on genuine acquisition artifacts (lacking pristine ground truths), place real-world unsimulated scans in the `data/SARS-CoV-2/` directory and execute:

```bash
python clinical_validation.py
```

>**Step 4: Explainable AI & Statistical Significance (Phase 5 & 7)**

To run the rigorous interpretability analysis—including the Montavon et al. (2018) True LRP ($\alpha_1\beta_0$ rule), the Bhati et al. (2025) K-Path extraction, and the paired t-tests (with Cohen's $d_z$)—execute the XAI script:

```bash
# Automatically loads pre-trained weights and generates visual/statistical reports
python xai_analysis.py --dataset_root ./data/CT_dataset
```

Outputs, including LRP heatmaps, dynamic thresholding masks, and the final statistical tables comparing SAGA against baselines, will be saved to the `./lrp_results_Both-K-Path/` directory.


### 4. Citation

If you find this code or our SAGA architecture useful in your research, please consider citing our paper:

```bash
@article{saga2026,
  title={An Interpretable Deep Learning Method for Medical Image Deblurring and Restoration},
  author={Siju K S, Vipin Venugopal, Mithun Kumar Kar, and Jayakrishnan Anandakrishnan },
  journal={Healthcare Analytics},
  year={2026}
}
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.
