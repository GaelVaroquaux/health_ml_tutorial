"""
An epidemiological study: Predicting 5-year mortality
======================================================

The NHANES study has been runs dietary interviews and blood tests for
several decade to measure the nutritional status of U.S. adults and
children and relate it to health.

Here we use models to predict 5-year mortality from the NHANES
covariates. We use both a simple model (a linear, that is a classic tool
in epidemiology) and a more flexible machine learning model.

We show how to understand these models within the context of our specific
question: mortality prediction. For this, we inspect features importance,
and look at partial dependence plots to understand how the model uses
covariates to predict.

Predicting a single yes/no outcome at a fixed horizon (here, 5 years)
like this is itself a common way of side-stepping censoring - the
"finite-horizon" approach revisited, and contrasted with full survival
analysis, in the last notebook.

Learning objectives and take home messages
-------------------------------------------

**This notebook introduces a dataset, and the finite-horizon approach to
a time-to-event question - both used as a baseline for the later notebook
on survival analysis.**

It can be skipped unless you have a particular interest in this type of
datasets.
"""

# %%
# Load NHANES 5-year-outcome dataset
# --------------------------------------
#
# This dataset was built from the NHANES study:

import pandas as pd

df = pd.read_csv("nhanes_1999_2018_mortality_5yr_horizon.csv")

X = df.drop(columns=["death_within_5y"])
y = df["death_within_5y"]

print("Covariates used:", X.columns.tolist())
print("\n5-year mortality rate:", y.mean())

# %%
# Model fitting and prediction
# ==================================
#
# Train / test split
# -------------------
#
# Before fitting models, we split train and test data, in order to have
# untouched hold-out data ("test") to evaluate the model.
# We stratify the split, the outcome is rare (~7%).

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# %%
# Fit a linear model: logistic regression
# ------------------------------------------
#
# We use a LogisticRegression from scikit-learn, but wrap it in a
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
# followed by a ``HistGradientBoostingClassifier``.

model_nonlinear = tabular_pipeline("classifier")
model_nonlinear.fit(X_train, y_train)

# %%
# Once again, evaluate it on left-out data
y_pred_proba_nonlinear = model_nonlinear.predict_proba(X_test)[:, 1]
auc_nonlinear = roc_auc_score(y_test, y_pred_proba_nonlinear)
print(f"Held-out AUC, non-linear model (gradient boosting): {auc_nonlinear:.3f}")

# %%
# The non-linear model predicts slightly better than the linear one.
# Let's now see what drives this prediction.
#
# Model inspection: how do the models predict
# =============================================
#
# Permutation importance: finding the important variables
# ---------------------------------------------------------
#
# We use permutation importance to see which variables drive the prediction
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
# The ordering of which variables are important for prediction are quite similar
# Across the two models.
# Let's now understand the difference in the predictions

# %%
# Partial dependence on age
# ---------------------------------------------------
#
# Here we plot "partial dependencies", that show how the prediction of the model
# for a given feature changes, on average, across the population.

# %%
# For the linear model
# ......................
#
# We plot both the prediction of the model (averaged across the population),
# and the average observed mortality for each age bin.

import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
sex_colors = {"M": "tab:blue", "F": "tab:orange"}
sex_markers = {"M": "o", "F": "^"}

# Plot the partial dependence of age for the whole population
pd_population_linear = partial_dependence(model_linear, X_test, features=["age"], grid_resolution=30)
plt.plot(pd_population_linear["grid_values"][0], pd_population_linear["average"][0],
        color="black", linewidth=2, label="Model prediction, averaged over whole population")

# Now plot for each sex, and overlay the average observed mortality for the corresponding sex
for sex_value in ["M", "F"]:
    is_sex = X_test["sex"] == sex_value
    pd_sex_linear = partial_dependence(model_linear, X_test[is_sex], features=["age"], grid_resolution=30)
    plt.plot(pd_sex_linear["grid_values"][0], pd_sex_linear["average"][0],
            color=sex_colors[sex_value], linewidth=2,
            label=f"Model prediction, averaged for sex = {sex_value}")
    age_sex_mean = (
        pd.DataFrame({"age": X_test.loc[is_sex, "age"], "y_test": y_test[is_sex]})
        .groupby("age", as_index=False)["y_test"].mean()
    )
    plt.plot(age_sex_mean["age"], age_sex_mean["y_test"],
            color=sex_colors[sex_value], linewidth=1, linestyle="--", marker=sex_markers[sex_value], markersize=4,
            label=f"Average 5 year mortality, sex = {sex_value}")


plt.xlabel("age")
plt.ylabel("predicted probability of death within 5 years")
plt.title("Partial dependence of age, by sex - linear model")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# What we see is that the model appears a bit as a "smoother" compared to the
# bin-wise average observed mortality. This is a good picture to have in mind for
# machine learning.
#
# This "smoothing" is a tradeoff to have in mind. More smoothing removes noise,
# but can also remove useful trends.
#
# The non-linear model is more flexible, and can capture more complex trends, but
# it can also capture more noise (overfitting).

# %%
# For the non-linear model
# .........................

pd_population_nonlinear = partial_dependence(model_nonlinear, X_test, features=["age"], grid_resolution=30)

plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
        color="black", linewidth=2, label="Model prediction, averaged over whole population")

for sex_value in ["M", "F"]:
    is_sex = X_test["sex"] == sex_value
    pd_sex_nonlinear = partial_dependence(model_nonlinear, X_test[is_sex], features=["age"], grid_resolution=30)
    plt.plot(pd_sex_nonlinear["grid_values"][0], pd_sex_nonlinear["average"][0],
            color=sex_colors[sex_value], linewidth=2,
            label=f"Model prediction, averaged for sex = {sex_value}")
    age_sex_mean = (
        pd.DataFrame({"age": X_test.loc[is_sex, "age"], "y_test": y_test[is_sex]})
        .groupby("age", as_index=False)["y_test"].mean()
    )
    plt.plot(age_sex_mean["age"], age_sex_mean["y_test"],
            color=sex_colors[sex_value], linewidth=1, linestyle="--", marker=sex_markers[sex_value], markersize=4,
            label=f"Average 5 year mortality, sex = {sex_value}")

plt.xlabel("age")
plt.ylabel("predicted probability of death within 5 years")
plt.title("Partial dependence of age, by sex - non-linear model")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# The non-linear model's curve is close to the linear one here: with
# age as the single dominant predictor of mortality, there is little
# non-linear structure left to gain from extra flexibility.
#
# Note that the data here is designed around a 5-year horizon. As such it
# sidesteps a challenge: censoring. The survival analysis notebook
# revisits this same dataset to get a full time-to-event answer instead,
# using every participant's actual follow-up duration, however long or
# short.

