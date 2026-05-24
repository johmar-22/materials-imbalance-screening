# Overcoming Class Imbalance in Materials Discovery


**Author:** Johaimen M. Omar & Kemal Akyol

## Overview
This repository contains the complete code, dataset, and results for the cross-validated ablation study on high-mobility p-type semiconductor screening. The pipeline addresses extreme class imbalance (19.3:1) in the BoltzTraP database by applying Borderline-SMOTE strictly within training folds to prevent data leakage.

## Repository Structure
**
materials-imbalance-screening/
├── README.md                         
├── requirements.txt                  
├── main.py                           
│
├── data/
│   └── boltztrap_mp.csv              
│
├── results/
│   ├── Ablation_Study_Metrics.csv
│   ├── Best_Hyperparameters.csv
│   ├── Feature_Importances.csv
│   ├── Threshold_Sensitivity.csv
│   ├── Extra_Pipeline_Results.csv
│   ├── Overfitting_TrainVal_Gap.csv
│   ├── Stat_Pipeline_Ranking.csv
│   └── LLM_Explanations.txt
│
└── figures/
    ├── Fig1_Confusion_Matrices.pdf
    ├── Fig2_ROC_PR_Curves.pdf
    ├── Fig3_Radar_Chart.pdf
    ├── Fig4_Feature_Importances.pdf
    ├── Fig5_PCA_SMOTE_CrimeScene.pdf
    ├── Fig6_Threshold_Sensitivity.pdf
    ├── Fig7_Ensemble_Comparison.pdf
    └── Fig8_Learning_Curve.pdf
**


## Reproducing the Results
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main pipeline: `python main.py`
This script executes the nested 5-fold cross-validation, hyperparameter tuning, overfitting diagnostics, statistical tests (McNemar and Wilcoxon), and generates all publication figures.

## Data
The `data/boltztrap_mp.csv` file contains the 9,036 inorganic crystalline materials from the BoltzTraP dataset, including the n-type transport properties used for cross-band-structure predictive mapping.

## UNDER REVIEW
