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


# %%
# Partial dependence of age, by race/ethnicity
# -----------------------------------------------

race_markers = {
    "Mexican American": "o",
    "Other Hispanic": "^",
    "Non-Hispanic White": "s",
    "Non-Hispanic Black": "D",
    "Other Race - Including Multi-Racial": "P",
}

plt.figure(figsize=(7, 5))

plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for race_value in sorted(X_test["race_eth"].unique()):
    is_race = X_test["race_eth"] == race_value
    pd_race = partial_dependence(model_nonlinear, X_test[is_race], features=["age"], grid_resolution=30)
    line, = plt.plot(pd_race["grid_values"][0], pd_race["average"][0],
                     linewidth=2, label=f"Model prediction, averaged for race/ethnicity = {race_value}")
    age_race_mean = (
        pd.DataFrame({"age": X_test.loc[is_race, "age"], "y_test": y_test[is_race]})
        .groupby("age", as_index=False)["y_test"].mean()
    )
    plt.plot(
        age_race_mean["age"],
        age_race_mean["y_test"],
        color=line.get_color(),
        linewidth=1,
        linestyle="--",
        marker=race_markers.get(race_value, "o"),
        markersize=4,
        label=f"Average 5 year mortality, race/ethnicity = {race_value}",
    )

plt.xlabel("age")
plt.ylabel("predicted probability of death within 5 years")
plt.title("Partial dependence of age, by race/ethnicity")
plt.legend(fontsize=8)
plt.tight_layout()

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
education_markers = {
    1.0: "o",
    2.0: "^",
    3.0: "s",
    4.0: "D",
    5.0: "P",
}

plt.figure(figsize=(7, 5))

plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for education_value, education_label in education_labels.items():
    is_education = X_test["education"] == education_value
    pd_education = partial_dependence(model_nonlinear, X_test[is_education], features=["age"], grid_resolution=30)
    line, = plt.plot(
        pd_education["grid_values"][0],
        pd_education["average"][0],
        linewidth=2,
        label=f"Model prediction, averaged for education = {education_label}",
    )
    age_education_mean = (
        pd.DataFrame({"age": X_test.loc[is_education, "age"], "y_test": y_test[is_education]})
        .groupby("age", as_index=False)["y_test"].mean()
    )
    plt.plot(
        age_education_mean["age"],
        age_education_mean["y_test"],
        color=line.get_color(),
        linewidth=1,
        linestyle="--",
        marker=education_markers[education_value],
        markersize=4,
        label=f"Average 5 year mortality, education = {education_label}",
    )

plt.xlabel("age")
plt.ylabel("predicted probability of death within 5 years")
plt.title("Partial dependence of age, by education")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# 2D partial dependence: age and waist circumference
# ------------------------------------------------------
#
# We drop rows with a missing waist circumference are dropped, to ease plotting,
# though the model handes missing values

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

