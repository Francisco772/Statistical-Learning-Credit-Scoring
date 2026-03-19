# Statistical-Learning-Credit-Scoring
A statistical study of credit risk using the South German Credit dataset. This project implements Lasso Logistic Regression and Bootstrap resampling to evaluate loan default probability. Key features include uncertainty assessment, Bayes Error analysis, and a focus on model interpretability over black-box prediction for financial decision-making.


# German Credit Risk: Statistical Inference & Uncertainty Assessment

This repository contains an empirical study of credit risk using the **South German Credit dataset**. Moving beyond simple classification, this project focuses on **statistical inference**, **Lasso regularization trade-offs**, and **quantifying the irreducible error** (Bayes Error) inherent in financial demographic data.

## Project Objectives
* **Predictive Modeling**: Classify applicants into "Good" or "Bad" credit risks using Generalized Linear Models.
* **Feature Selection**: Utilize **Lasso ($L_1$) regularization** to identify the most parsimonious set of risk drivers.
* **Uncertainty Quantification**: Employ **Bootstrap resampling** to move from "point estimates" to confidence intervals for model coefficients and performance.
* **Decision Theory**: Analyze the trade-off between Overall Accuracy and Recall (Risk Detection) in a lending context.

## Technical Implementation
The analysis was performed in Python, prioritizing a "Statistical Learning" approach (as defined by Efron & Hastie) over a pure black-box machine learning approach.

* **Models**: 
    * Full Logistic Regression (Baseline)
    * Reduced Specification (P-value filtered)
    * Lasso Regularized Logistic Regression
* **Validation**: Stratified K-Fold Cross-Validation and Bootstrap Resampling (1000+ iterations).
* **Key Libraries**: `scikit-learn`, `statsmodels`, `seaborn`, `pandas`, `matplotlib`.

## Key Insights

### 1. The "Statistical Overlap" & Bayes Error
A core finding of this study is the visualization of the **Bayes Error Rate**. Even with optimized models, there is a significant overlap in the probability distributions of "Good" and "Bad" clients (particularly in the 0.3 to 0.6 range). This confirms that socio-demographic data alone has an intrinsic mathematical limit for prediction.

### Parsimony vs. Predictive Power
By applying Lasso regularization, we successfully reduced the feature space—removing non-significant predictors like "Number of People Liable"—without a substantial drop in AUC. This highlights that a few key variables (Checking Account Status, Duration, and Credit History) carry the majority of the predictive signal.

### Cost-Sensitive Thresholds
In credit scoring, the cost of lending to a "Bad" client is significantly higher than the cost of rejecting a "Good" one. This project includes a threshold analysis demonstrating how shifting the classification cutoff from the default 0.5 can optimize for risk detection (Recall) at the expense of overall accuracy.


---
**Authors:** Francisco Oliveira, António Santos, Vincenzo Sabino, Tibor Szolomaier, Henrique Aleixo
