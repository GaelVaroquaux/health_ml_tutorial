"""
Predicting ICU mortality
=========================

We predict mortality in critically ill patients from the SUPPORT2 study
covariates, inspect feature importances, and look at partial dependence
plots to understand how the model uses covariates to predict.
"""

# %%
# Load the SUPPORT2 ICU-mortality dataset
# ------------------------------------------
#
# This dataset was built from the SUPPORT2 study of critically ill
# hospitalized patients:

import pandas as pd

df = pd.read_csv("support2_icu_mortality.csv")

X = df.drop(columns=["death"])
y = df["death"]

print("Covariates used:", X.columns.tolist())
print("\nDeath rate:", y.mean())

# %%
# Model fitting and prediction
# ==================================
#
# Train / test split
# -------------------
#
# Before fitting models, we split train and test data, in order to have
# untouched hold-out data ("test") to evaluate the model.
# We stratify the split: patients in this ICU cohort are very sick, and
# most (~68%) die by the end of follow-up.

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
# Fit a non-linear model: random forest
# --------------------------------------
#
# This time we use a ``RandomForestClassifier`` rather than the gradient
# boosting used in the previous example: it is a bit less powerful, but
# has a single, easy-to-reason-about "flexibility" knob, the maximum
# depth of its trees. We pick that depth with cross-validation on the
# training data alone, so the test set stays untouched.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

for max_depth in [3, 5, 8, 10, 12, 15, None]:
    candidate = tabular_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=max_depth, random_state=0)
    )
    scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"max_depth={str(max_depth):5s} mean CV AUC={scores.mean():.3f}")

# %%
# A depth of 10 gives close to the best cross-validated performance,
# without being at the extreme end of the range. The next example digs
# into what happens if we push that depth much further in either
# direction.

model_nonlinear = tabular_pipeline(
    RandomForestClassifier(n_estimators=300, max_depth=10, random_state=0)
)
model_nonlinear.fit(X_train, y_train)

# %%
# Evaluate it on held-out data
y_pred_proba_nonlinear = model_nonlinear.predict_proba(X_test)[:, 1]
auc_nonlinear = roc_auc_score(y_test, y_pred_proba_nonlinear)
print(f"Held-out AUC, non-linear model (random forest): {auc_nonlinear:.3f}")

# %%
# The non-linear model predicts distinctly better than the linear one -
# a bigger gap than we saw in the NHANES example. Let's now see what
# drives this prediction.
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
# For both models, cancer status, the ICU-intensity score, age, and the
# coma score come out on top. Let's now understand the difference in
# the predictions, focusing on the ICU-intensity score: a measure of how
# much care (monitoring, interventions) a patient received in the ICU.

# %%
# Partial dependence on ICU-intensity score
# ---------------------------------------------------
#
# Here we plot "partial dependencies", that show how the prediction of
# the model for this feature changes, on average, across the
# population.

# %%
# For the linear model
# ......................
#
# We plot both the prediction of the model (averaged across the
# population), and the average observed mortality for each score value.

import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

sex_colors = {"male": "tab:blue", "female": "tab:orange"}
sex_markers = {"male": "o", "female": "^"}

# ICU-intensity score is a near-continuous score (352 distinct values), so
# we bin it into 15 groups before averaging observed mortality - otherwise
# most bins would hold only 1 or 2 patients, and the "observed" curve would
# be pure noise.
score_bins = pd.cut(X_test["icu_intensity_score"], bins=15)
score_bin_centers = score_bins.apply(lambda interval: interval.mid).astype(float)

# Plot the partial dependence for the whole population
pd_population_linear = partial_dependence(
    model_linear, X_test, features=["icu_intensity_score"], grid_resolution=30
)
plt.plot(pd_population_linear["grid_values"][0], pd_population_linear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

# Now plot for each sex, and overlay the average observed mortality for the corresponding sex
for sex_value in ["male", "female"]:
    is_sex = X_test["sex"] == sex_value
    pd_sex_linear = partial_dependence(
        model_linear, X_test[is_sex], features=["icu_intensity_score"], grid_resolution=30
    )
    plt.plot(pd_sex_linear["grid_values"][0], pd_sex_linear["average"][0],
             color=sex_colors[sex_value], linewidth=2,
             label=f"Model prediction, averaged for sex = {sex_value}")
    score_sex_mean = (
        pd.DataFrame({
            "icu_intensity_score": score_bin_centers[is_sex],
            "y_test": y_test[is_sex],
        })
        .groupby("icu_intensity_score", as_index=False)["y_test"].mean()
    )
    plt.plot(score_sex_mean["icu_intensity_score"], score_sex_mean["y_test"],
             color=sex_colors[sex_value], linewidth=1, linestyle="--",
             marker=sex_markers[sex_value], markersize=4,
             label=f"Average mortality, sex = {sex_value}")

plt.xlabel("ICU-intensity score")
plt.ylabel("predicted probability of death")
plt.title("Partial dependence of ICU-intensity score, by sex - linear model")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# The linear model draws a straight line through a relationship that is
# clearly not straight: mortality rises steeply for the sickest patients
# (highest ICU-intensity scores), a rise the linear model can only
# partly follow.

# %%
# For the non-linear model
# .........................

pd_population_nonlinear = partial_dependence(
    model_nonlinear, X_test, features=["icu_intensity_score"], grid_resolution=30
)

plt.figure()
plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for sex_value in ["male", "female"]:
    is_sex = X_test["sex"] == sex_value
    pd_sex_nonlinear = partial_dependence(
        model_nonlinear, X_test[is_sex], features=["icu_intensity_score"], grid_resolution=30
    )
    plt.plot(pd_sex_nonlinear["grid_values"][0], pd_sex_nonlinear["average"][0],
             color=sex_colors[sex_value], linewidth=2,
             label=f"Model prediction, averaged for sex = {sex_value}")
    score_sex_mean = (
        pd.DataFrame({
            "icu_intensity_score": score_bin_centers[is_sex],
            "y_test": y_test[is_sex],
        })
        .groupby("icu_intensity_score", as_index=False)["y_test"].mean()
    )
    plt.plot(score_sex_mean["icu_intensity_score"], score_sex_mean["y_test"],
             color=sex_colors[sex_value], linewidth=1, linestyle="--",
             marker=sex_markers[sex_value], markersize=4,
             label=f"Average mortality, sex = {sex_value}")

plt.xlabel("ICU-intensity score")
plt.ylabel("predicted probability of death")
plt.title("Partial dependence of ICU-intensity score, by sex - non-linear model")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# The non-linear model follows the rise in mortality at high
# ICU-intensity scores much more closely. This is the kind of
# non-linear link that motivated using a flexible, non-parametric
# model in the first place.

# %%
# Partial dependence of ICU-intensity score, by cancer status
# -----------------------------------------------------------------

cancer_colors = {"no": "tab:green", "yes": "tab:red", "metastatic": "tab:purple"}
cancer_markers = {"no": "o", "yes": "^", "metastatic": "s"}

plt.figure(figsize=(7, 5))

plt.plot(pd_population_nonlinear["grid_values"][0], pd_population_nonlinear["average"][0],
         color="black", linewidth=2, label="Model prediction, averaged over whole population")

for cancer_value in ["no", "yes", "metastatic"]:
    is_cancer = X_test["cancer_status"] == cancer_value
    pd_cancer = partial_dependence(
        model_nonlinear, X_test[is_cancer], features=["icu_intensity_score"], grid_resolution=30
    )
    plt.plot(pd_cancer["grid_values"][0], pd_cancer["average"][0],
             color=cancer_colors[cancer_value], linewidth=2,
             label=f"Model prediction, averaged for cancer status = {cancer_value}")
    score_cancer_mean = (
        pd.DataFrame({
            "icu_intensity_score": score_bin_centers[is_cancer],
            "y_test": y_test[is_cancer],
        })
        .groupby("icu_intensity_score", as_index=False)["y_test"].mean()
    )
    plt.plot(score_cancer_mean["icu_intensity_score"], score_cancer_mean["y_test"],
             color=cancer_colors[cancer_value], linewidth=1, linestyle="--",
             marker=cancer_markers[cancer_value], markersize=4,
             label=f"Average mortality, cancer status = {cancer_value}")

plt.xlabel("ICU-intensity score")
plt.ylabel("predicted probability of death")
plt.title("Partial dependence of ICU-intensity score, by cancer status")
plt.legend(fontsize=8)
plt.tight_layout()

# %%
# Cancer status shifts the whole curve up or down, but the rise with
# ICU-intensity score remains visible in every group - the two effects
# are mostly additive rather than interacting.

# %%
# 2D partial dependence: age and ICU-intensity score
# ------------------------------------------------------

pd_age_score = partial_dependence(
    model_nonlinear, X_test, features=["age", "icu_intensity_score"], grid_resolution=30
)
age_grid, score_grid = pd_age_score["grid_values"]

fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(age_grid, score_grid, pd_age_score["average"][0].T, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax, label="predicted probability of death")
ax.set_xlabel("age")
ax.set_ylabel("ICU-intensity score")
ax.set_title("Partial dependence of age and ICU-intensity score")
fig.tight_layout()
