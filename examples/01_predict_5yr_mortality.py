"""
Predicting 5-year mortality
============================

We predict 5-year mortality from the NHANES covariates, inspect features
importance, and look at partial dependence plots to understand how the model
uses covariates to predict.
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
    "importance_std": perm_nonlinear.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Permutation importance (drop in prediction performance when a feature is shuffled):")
print(importances_nonlinear.to_string(index=False))

# %%
# Partial dependence of age, by sex - non-linear model
# ------------------------------------------------------
#
# We compute the partial dependence of age on the predicted probability
# of death within 5 years, on the test set: once using the whole
# population, and once restricted to males or to females. Individual
# test-set predictions are overlaid as a scatter, colored and marked by
# sex, to relate the average curves to the spread of individual
# predictions.

import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

sex_styles = {"M": ("tab:blue", "o"), "F": ("tab:orange", "^")}

fig, ax = plt.subplots(figsize=(7, 5))

pd_population_nonlinear = partial_dependence(model_nonlinear, X_test, features=["age"], grid_resolution=30)
ax.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
        color="black", linewidth=2, label="whole population")

for sex_value, (color, marker) in sex_styles.items():
    is_sex = X_test["sex"] == sex_value
    pd_sex_nonlinear = partial_dependence(model_nonlinear, X_test[is_sex], features=["age"], grid_resolution=30)
    ax.plot(pd_sex_nonlinear["grid_values"][0], pd_sex_nonlinear["average"][0],
            color=color, linewidth=2, label=f"sex = {sex_value}")
    ax.scatter(X_test.loc[is_sex, "age"], y_pred_proba_nonlinear[is_sex],
               color=color, marker=marker, alpha=0.3, s=15)

ax.set_xlabel("age")
ax.set_ylabel("predicted probability of death within 5 years")
ax.set_title("Partial dependence of age, by sex - non-linear model")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The ordering of which variables are important for prediction are quite similar
# Across the two models.
# Let's now understand the difference in the predictions

# %%
# Partial dependence of age, by sex - linear model
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

pd_population_linear = partial_dependence(model_linear, X_test, features=["age"], grid_resolution=30)
ax.plot(pd_population_linear["grid_values"][0], pd_population_linear["average"][0],
        color="black", linewidth=2, label="whole population")

for sex_value, (color, marker) in sex_styles.items():
    is_sex = X_test["sex"] == sex_value
    pd_sex_linear = partial_dependence(model_linear, X_test[is_sex], features=["age"], grid_resolution=30)
    ax.plot(pd_sex_linear["grid_values"][0], pd_sex_linear["average"][0],
            color=color, linewidth=2, label=f"sex = {sex_value}")
    ax.scatter(X_test.loc[is_sex, "age"], y_pred_proba_linear[is_sex],
               color=color, marker=marker, alpha=0.3, s=15)

ax.set_xlabel("age")
ax.set_ylabel("predicted probability of death within 5 years")
ax.set_title("Partial dependence of age, by sex - linear model")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Partial dependence of age, by race/ethnicity
# -----------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
        color="black", linewidth=2, label="whole population")

for race_value in sorted(X_test["race_eth"].unique()):
    is_race = X_test["race_eth"] == race_value
    pd_race = partial_dependence(model_nonlinear, X_test[is_race], features=["age"], grid_resolution=30)
    line, = ax.plot(pd_race["grid_values"][0], pd_race["average"][0],
                     linewidth=2, label=race_value)
    ax.scatter(X_test.loc[is_race, "age"], y_pred_proba_nonlinear[is_race],
               color=line.get_color(), marker="o", alpha=0.3, s=15)

ax.set_xlabel("age")
ax.set_ylabel("predicted probability of death within 5 years")
ax.set_title("Partial dependence of age, by race/ethnicity")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
# Partial dependence of age, by education
# -------------------------------------------
#
# Education codes 7 ("refused") and 9 ("don't know") concern too few
# participants to give a meaningful curve, so they are excluded here.

education_labels = {
    1.0: "less than 9th grade",
    2.0: "9-11th grade",
    3.0: "high school grad / GED",
    4.0: "some college / AA degree",
    5.0: "college graduate or above",
}

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
        color="black", linewidth=2, label="whole population")

for education_value, education_label in education_labels.items():
    is_education = X_test["education"] == education_value
    pd_education = partial_dependence(model_nonlinear, X_test[is_education], features=["age"], grid_resolution=30)
    line, = ax.plot(pd_education["grid_values"][0], pd_education["average"][0],
                     linewidth=2, label=education_label)
    ax.scatter(X_test.loc[is_education, "age"], y_pred_proba_nonlinear[is_education],
               color=line.get_color(), marker="o", alpha=0.3, s=15)

ax.set_xlabel("age")
ax.set_ylabel("predicted probability of death within 5 years")
ax.set_title("Partial dependence of age, by education")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
# 2D partial dependence: age and waist circumference
# ------------------------------------------------------
#
# Rows with a missing waist circumference are dropped: with them
# included, the 5th-95th percentile grid used for the plot's axis
# becomes undefined.

X_test_complete = X_test.dropna(subset=["age", "waist_cm"])
pd_age_waist = partial_dependence(
    model_nonlinear, X_test_complete, features=["age", "waist_cm"], grid_resolution=30
)
age_grid, waist_grid = pd_age_waist["grid_values"]

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(age_grid, waist_grid, pd_age_waist["average"][0].T, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="predicted probability of death within 5 years")
ax.set_xlabel("age")
ax.set_ylabel("waist circumference (cm)")
ax.set_title("Partial dependence of age and waist circumference")
fig.tight_layout()
plt.show()

# %%
# 2D partial dependence: age and BMI
# ---------------------------------------

X_test_complete = X_test.dropna(subset=["age", "bmi"])
pd_age_bmi = partial_dependence(
    model_nonlinear, X_test_complete, features=["age", "bmi"], grid_resolution=30
)
age_grid, bmi_grid = pd_age_bmi["grid_values"]

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(age_grid, bmi_grid, pd_age_bmi["average"][0].T, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="predicted probability of death within 5 years")
ax.set_xlabel("age")
ax.set_ylabel("BMI")
ax.set_title("Partial dependence of age and BMI")
fig.tight_layout()
plt.show()

# %%
