
#standard libraries
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

# libraries for model building
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import(train_test_split,GridSearchCV,RandomizedSearchCV,StratifiedKFold,cross_val_score,cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score,f1_score,precision_score, classification_report, 
                             confusion_matrix, roc_curve, recall_score,precision_recall_curve,make_scorer,auc,average_precision_score,ConfusionMatrixDisplay)

import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# VIF for feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('seaborn-v0_8')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


def evaluate_model(pipe, x_test, y_test, threshold: float = 0.5, save_figures = False, output_dir= "reports/figures"):
    if save_figures:
        os.makedirs(output_dir,exist_ok=True)
    
    # ---- Predictions ----
    y_probs = pipe.predict_proba(x_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    # ---- Metrics ----
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs))
    print("PR-AUC:", average_precision_score(y_test, y_probs))

    results = {
        "classification_report": classification_report(y_test, y_preds, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "pr_auc": average_precision_score(y_test, y_probs),
        "confusion_matrix": confusion_matrix(y_test, y_preds)
    }

    # ---- Confusion Matrix ----
    cm = results["confusion_matrix"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_figures:
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {results['roc_auc']:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    if save_figures:
        plt.savefig(f"{output_dir}/roc_curve.png", dpi=120, bbox_inches="tight")
    plt.show()

    # ---- Precision-Recall Curve ----
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions, label=f"PR AUC = {results['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    if save_figures:
        plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=120, bbox_inches="tight")
    plt.show()

    return results