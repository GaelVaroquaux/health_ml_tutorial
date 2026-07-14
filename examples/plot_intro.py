"""
Introduction to Health ML
==========================

This example gives a brief overview of a typical machine learning
workflow applied to health data: loading a dataset, exploring it, and
fitting a simple classifier.

.. note::

   Click the **Launch** button above to run this notebook interactively
   in your browser — no installation needed!
"""

# %%
# Loading a health dataset
# ------------------------
#
# We use the ``load_breast_cancer`` dataset from scikit-learn as a stand-in
# for a real clinical dataset.  It contains 30 numeric features (e.g. tumour
# geometry measurements) and a binary label (malignant / benign).

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# %%
# Visualising feature distributions
# ----------------------------------
#
# Let's plot the distribution of the first four features for each class.

feature_names = X.columns[:4]
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=False)

for ax, feat in zip(axes, feature_names):
    for label, colour in zip([0, 1], ["tab:red", "tab:blue"]):
        ax.hist(X.loc[y == label, feat], bins=20, alpha=0.6,
                color=colour, label=data.target_names[label])
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("Count")

axes[0].legend(fontsize=8)
fig.suptitle("Feature distributions by class", y=1.02)
fig.tight_layout()
plt.show()

# %%
# Training a logistic regression classifier
# -----------------------------------------
#
# We split the data into a training and a test set and fit a logistic
# regression model.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=0)
clf.fit(X_train_s, y_train)

print(classification_report(y_test, clf.predict(X_test_s),
                             target_names=data.target_names))

# %%
# Confusion matrix
# ----------------

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_s, y_test,
    display_labels=data.target_names,
    ax=ax,
)
ax.set_title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.show()
