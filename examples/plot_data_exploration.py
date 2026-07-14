"""
Exploring Health Data
=====================

Before building a model it is essential to understand the data.  This
example demonstrates common exploratory data analysis (EDA) steps:
summary statistics, correlation analysis, and dimensionality reduction
with PCA.

.. note::

   Click the **Launch** button above to run this notebook interactively
   in your browser — no installation needed!
"""

# %%
# Load and inspect the dataset
# ----------------------------
#
# We use the Diabetes dataset from scikit-learn.  It contains 10 baseline
# variables for 442 patients and a quantitative measure of disease
# progression one year after baseline.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data.data.copy()
df["target"] = data.target

print(df.describe().round(2))

# %%
# Correlation matrix
# ------------------
#
# A heatmap of pairwise Pearson correlations helps spot redundant features
# and features that are strongly related to the outcome.

fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr()
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
fig.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Feature correlation matrix")
fig.tight_layout()
plt.show()

# %%
# Dimensionality reduction with PCA
# ----------------------------------
#
# Principal Component Analysis (PCA) projects the 10-dimensional feature
# space onto 2 dimensions so we can visualise the patient population and
# colour-code patients by their disease score.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = data.data.values
y = data.target.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                c=y, cmap="viridis", alpha=0.7, s=20)
fig.colorbar(sc, ax=ax, label="Disease progression")
ax.set_xlabel(f"PC 1 ({explained[0]:.1f} % var)")
ax.set_ylabel(f"PC 2 ({explained[1]:.1f} % var)")
ax.set_title("PCA of diabetes dataset")
fig.tight_layout()
plt.show()

# %%
# Feature importance via a random forest
# ----------------------------------------
#
# A quick way to rank features is to train a random forest regressor and
# read out the impurity-based feature importances.

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_scaled, y)

importances = pd.Series(rf.feature_importances_, index=data.feature_names)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(5, 4))
importances.plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Random Forest feature importances")
fig.tight_layout()
plt.show()
