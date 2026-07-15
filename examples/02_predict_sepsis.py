"""
Predicting sepsis in the ICU
==============================

We predict, from just the first 24 hours of a patient's ICU stay, whether
they will later be diagnosed with sepsis, using a small number of
covariates: age, sex, the delay between hospital and ICU admission, and
diastolic blood pressure. We compare a linear and a non-linear model, and
look at partial dependence plots to understand how the non-linear model
uses these covariates to predict.
"""

# %%
# Load the PhysioNet sepsis dataset
# ------------------------------------
#
# This dataset was built from the PhysioNet/Computing in Cardiology
# Challenge 2019, a public dataset of 40,336 ICU patients:

import pandas as pd

df = pd.read_csv("physionet_sepsis.csv")

X = df[["age", "sex", "hours_before_icu", "diastolic_bp_mmhg"]]
y = df["sepsis"]

print("Covariates used:", X.columns.tolist())
print("\nSepsis rate:", y.mean())

# %%
# Model fitting and prediction
# ==================================
#
# Train / test split
# -------------------
#
# Before fitting models, we split train and test data, in order to have
# untouched hold-out data ("test") to evaluate the model.
# We stratify the split: sepsis is rare in this cohort (~4%).

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# %%
# Fit a linear model: logistic regression
# ------------------------------------------
#
# We use a LogisticRegression from scikit-learn, wrapped in a
# ``tabular_pipeline`` from skrub that does some data preparation.

from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

model_linear = tabular_pipeline(LogisticRegression())
model_linear.fit(X_train, y_train)

# %%
# Evaluate the model on held-out data
from sklearn.metrics import roc_auc_score

y_pred_proba_linear = model_linear.predict_proba(X_test)[:, 1]
auc_linear = roc_auc_score(y_test, y_pred_proba_linear)
print(f"Held-out AUC, linear model (logistic regression): {auc_linear:.3f}")

# %%
# Fit a non-linear model: gradient boosting
# --------------------------------------------
#
# This is skrub's default classification pipeline: a ``TableVectorizer``
# followed by a ``HistGradientBoostingClassifier``. With sepsis this rare,
# a ``RandomForestClassifier`` or ``ExtraTreesClassifier`` left at their
# default settings (unlimited tree depth) over-fit badly and end up
# *worse* than the linear model - gradient boosting's default is much
# more conservative out of the box, and is also the fastest of the three
# to fit.

model_nonlinear = tabular_pipeline("classifier")
model_nonlinear.fit(X_train, y_train)

# %%
# Evaluate it on held-out data
y_pred_proba_nonlinear = model_nonlinear.predict_proba(X_test)[:, 1]
auc_nonlinear = roc_auc_score(y_test, y_pred_proba_nonlinear)
print(f"Held-out AUC, non-linear model (gradient boosting): {auc_nonlinear:.3f}")

# %%
# The non-linear model predicts distinctly better than the linear one,
# with only four covariates. Let's now see what drives this prediction.
#
# Model inspection: how do the models predict
# =============================================
#
# Permutation importance: finding the important variables
# ---------------------------------------------------------
#
# We use permutation importance to see which variables drive the
# prediction.

from sklearn.inspection import permutation_importance

perm_linear = permutation_importance(
    model_linear, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0
)

importances_linear = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm_linear.importances_mean,
    "importance_std": perm_linear.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Permutation importance (drop in prediction performance when a feature is shuffled):")
print(importances_linear.to_string(index=False))

# %%
# For the non-linear model

perm_nonlinear = permutation_importance(
    model_nonlinear, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0
)

importances_nonlinear = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm_nonlinear.importances_mean,
}).sort_values("importance_mean", ascending=False)

print("Permutation importance (drop in prediction performance when a feature is shuffled):")
print(importances_nonlinear.to_string(index=False))

# %%
# Diastolic blood pressure and the delay before ICU admission stand out
# for the non-linear model. Let's now understand the difference in the
# predictions, focusing on diastolic blood pressure.

# %%
# Partial dependence on diastolic blood pressure
# ---------------------------------------------------
#
# Here we plot "partial dependencies", that show how the prediction of
# the model for this feature changes, on average, across the population.
#
# A few patients are missing a diastolic blood pressure reading, so we
# first drop them: the model can handle missing values internally, but
# the plotting code below cannot.

X_test_complete = X_test.dropna(subset=["diastolic_bp_mmhg"])
y_test_complete = y_test.loc[X_test_complete.index]

# %%
# Diastolic blood pressure is also a near-continuous measurement, so we
# group it into 8 bins of equal size before averaging observed sepsis
# rates - otherwise, especially once split by sex, some bins would hold
# only a handful of patients, and the "observed" curve would be pure
# noise. We use the mean diastolic blood pressure inside each bin as its
# position on the x-axis.

observed = X_test_complete[["sex", "diastolic_bp_mmhg"]].copy()
observed["sepsis"] = y_test_complete
observed["dbp_bin"] = pd.qcut(observed["diastolic_bp_mmhg"], q=8)

import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

sex_colors = {"M": "tab:blue", "F": "tab:orange"}
sex_markers = {"M": "o", "F": "^"}

# %%
# For the linear model
# ......................

pd_population_linear = partial_dependence(
    model_linear, X_test_complete, features=["diastolic_bp_mmhg"], grid_resolution=30
)
plt.plot(pd_population_linear["grid_values"][0], pd_population_linear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for sex_value in ["M", "F"]:
    is_sex = X_test_complete["sex"] == sex_value
    pd_sex_linear = partial_dependence(
        model_linear, X_test_complete[is_sex], features=["diastolic_bp_mmhg"], grid_resolution=30
    )
    plt.plot(pd_sex_linear["grid_values"][0], pd_sex_linear["average"][0],
             color=sex_colors[sex_value], linewidth=2,
             label=f"Model prediction, averaged for sex = {sex_value}")

    observed_sex = observed[observed["sex"] == sex_value]
    observed_by_bin = observed_sex.groupby("dbp_bin")[["diastolic_bp_mmhg", "sepsis"]].mean()
    plt.plot(observed_by_bin["diastolic_bp_mmhg"], observed_by_bin["sepsis"],
             color=sex_colors[sex_value], linewidth=1, linestyle="--",
             marker=sex_markers[sex_value], markersize=4,
             label=f"Average sepsis rate, sex = {sex_value}")

plt.xlabel("diastolic blood pressure (mmHg)")
plt.ylabel("predicted probability of sepsis")
plt.title("Partial dependence of diastolic BP, by sex - linear model")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# The linear model draws a straight line, decreasing steadily as
# diastolic blood pressure rises. This only captures part of the
# picture: low blood pressure (a sign of poor perfusion / possible
# septic shock) drives risk up sharply, but risk does not keep
# decreasing forever as pressure rises further.

# %%
# For the non-linear model
# .........................

pd_population_nonlinear = partial_dependence(
    model_nonlinear, X_test_complete, features=["diastolic_bp_mmhg"], grid_resolution=30
)

plt.figure()
plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for sex_value in ["M", "F"]:
    is_sex = X_test_complete["sex"] == sex_value
    pd_sex_nonlinear = partial_dependence(
        model_nonlinear, X_test_complete[is_sex], features=["diastolic_bp_mmhg"], grid_resolution=30
    )
    plt.plot(pd_sex_nonlinear["grid_values"][0], pd_sex_nonlinear["average"][0],
             color=sex_colors[sex_value], linewidth=2,
             label=f"Model prediction, averaged for sex = {sex_value}")

    observed_sex = observed[observed["sex"] == sex_value]
    observed_by_bin = observed_sex.groupby("dbp_bin")[["diastolic_bp_mmhg", "sepsis"]].mean()
    plt.plot(observed_by_bin["diastolic_bp_mmhg"], observed_by_bin["sepsis"],
             color=sex_colors[sex_value], linewidth=1, linestyle="--",
             marker=sex_markers[sex_value], markersize=4,
             label=f"Average sepsis rate, sex = {sex_value}")

plt.xlabel("diastolic blood pressure (mmHg)")
plt.ylabel("predicted probability of sepsis")
plt.title("Partial dependence of diastolic BP, by sex - non-linear model")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# The non-linear model instead traces a bumpy, non-monotonic curve that
# tracks the jagged rises and dips of the observed rates much more
# closely than the linear model's straight line. Whether those bumps
# reflect real physiology or just noise in this particular sample is
# exactly the kind of question the next example, on under- and
# over-fitting, will help answer.

# %%
# 2D partial dependence: delay before ICU admission and diastolic BP
# ------------------------------------------------------------------------

pd_2d = partial_dependence(
    model_nonlinear, X_test_complete, features=["hours_before_icu", "diastolic_bp_mmhg"],
    grid_resolution=30,
)
hours_grid, dbp_grid = pd_2d["grid_values"]

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(hours_grid, dbp_grid, pd_2d["average"][0].T, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="predicted probability of sepsis")
ax.set_xlabel("hours between hospital and ICU admission")
ax.set_ylabel("diastolic blood pressure (mmHg)")
ax.set_title("Partial dependence of admission delay and diastolic BP")
fig.tight_layout()
