# Credit Card Fraud Detection — Logistic Regression + SMOTE

A project demonstrating fraud detection using **Logistic Regression**, handling **class imbalance with SMOTE**, and selecting an optimal cutoff via **Youden’s J statistic**. Evaluation includes **Confusion Matrix** and **ROC Curve**.

---

## 🔍 Overview
- **Goal:** Detect fraudulent credit card transactions in a highly imbalanced dataset  
- **Model:** Logistic Regression (binomial GLM)  
- **Imbalance Handling:** SMOTE oversampling of minority class  
- **Threshold Selection:** Youden’s J statistic  
- **Evaluation:** Confusion Matrix, ROC–AUC, Accuracy  
- **Results:** ~97% accuracy, 0.99 AUC on test data  

---

## 📦 Setup
```r
install.packages(c("smotefamily", "caTools", "pROC", "caret"))

library(smotefamily)
library(caTools)
library(pROC)
library(caret)
