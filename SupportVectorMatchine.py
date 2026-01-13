#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# LEVEL 3 - TASK 2: SUPPORT VECTOR MACHINE (SVM) FOR CLASSIFICATION
# Codveda Machine Learning Internship - Sindiswa
# Dataset: Customer Churn (churn-bigml-80.csv + churn-bigml-20.csv)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ──── Visual style ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


folder_path = r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\Churn Prdiction Data"

train_file = folder_path + r"\churn-bigml-80.csv"
test_file  = folder_path + r"\churn-bigml-20.csv"

try:
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)
    print("✓ Datasets loaded successfully!")
    print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
except FileNotFoundError as e:
    print("File not found! Please double-check:")
    print("Folder:", folder_path)
    print("Files needed: churn-bigml-80.csv and churn-bigml-20.csv")
    print("Error:", e)
    raise

# ──── 2. Quick preprocessing ────────────────────────────────────────────────
# Convert 'Yes'/'No' to 1/0
for col in ['International plan', 'Voice mail plan']:
    train_df[col] = (train_df[col] == 'Yes').astype(int)
    test_df[col]  = (test_df[col] == 'Yes').astype(int)

# Features (dropping State to avoid too many dummies)
features = [
    'Account length', 'International plan', 'Voice mail plan',
    'Number vmail messages', 'Total day minutes', 'Total day calls',
    'Total day charge', 'Total eve minutes', 'Total eve calls',
    'Total eve charge', 'Total night minutes', 'Total night calls',
    'Total night charge', 'Total intl minutes', 'Total intl calls',
    'Total intl charge', 'Customer service calls'
]

X_train = train_df[features]
y_train = train_df['Churn'].astype(int)

X_test = test_df[features]
y_test = test_df['Churn'].astype(int)

# Scale (mandatory for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ──── 3. Train & Compare Two Kernels ────────────────────────────────────────
kernels = ['linear', 'rbf']
results = []

print("\nTraining SVM models...\n")

for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, random_state=42, C=1.0)
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'Kernel': kernel.upper(),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }
    results.append(metrics)

    print(f"{kernel.upper():7} → Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | AUC: {metrics['AUC']:.4f}")

# Results table
print("\n" + "═"*55)
print("KERNEL COMPARISON RESULTS")
print("═"*55)
df_results = pd.DataFrame(results)
print(df_results.round(4))
print("═"*55 + "\n")

# ──── 4. Visualize Decision Boundary (PCA 2D projection) ────────────────────
print("Generating decision boundary visualization...")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Train SVM on 2D data (RBF usually best)
svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_2d.fit(X_train_pca, y_train)

# Mesh for boundary
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', 
            edgecolors='k', s=60, alpha=0.8)
plt.title('SVM Decision Boundary (RBF Kernel) - PCA 2D Projection\n(Customer Churn)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(['No Churn (0)', 'Churn (1)'], loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

# ──── 5. Final Confusion Matrix (using RBF) ─────────────────────────────────
y_pred_rbf = SVC(kernel='rbf', probability=True, random_state=42).fit(X_train_scaled, y_train).predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - SVM (RBF Kernel)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:




