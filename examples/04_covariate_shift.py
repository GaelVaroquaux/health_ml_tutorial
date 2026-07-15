"""
Covariate shift: when the deployment population differs
============================================================

A model is often developed on one population (the "study" population),
then used to make predictions on a different population (the "target"
population). If the two populations have a different distribution of
covariates, but the same underlying relationship between covariates and
outcome, this is called *covariate shift*. It is one of the most common
ways a machine-learning risk score breaks once it leaves the population
it was built on - see Dockès, Varoquaux and Poline (2021), "Preventing
dataset shift from breaking machine learning biomarkers", GigaScience,
for a general discussion of this failure mode.

Here, we simulate a covariate shift on the PhysioNet sepsis data: a
"study" population, mostly admitted to the ICU soon after hospital
admission, and a "target" population, where ICU admission more often
comes late - a well known patient-safety scenario (a patient
deteriorates on the ward for a while before being escalated to the
ICU). We show that a linear model, well calibrated on the study
population, badly under-estimates risk on the target population, while
a non-linear model is more robust.
"""

# %%
# Load the data
# ---------------

import pandas as pd

df = pd.read_csv("physionet_sepsis.csv")
df = df.dropna(subset=["hours_before_icu"])

# %%
# Building a study and a target population
# --------------------------------------------
#
# We give every patient a "lateness" score between 0 (admitted to the
# ICU right away) and 1 (admitted very late), based on their rank on
# ``hours_before_icu``. We then draw:
#
# - a "study" population, weighted towards early ICU admission (low
#   lateness), but with no patient excluded outright
# - a "target" population, drawn from the remaining patients, weighted
#   towards late ICU admission (high lateness)
#
# This is a *soft* shift: both populations cover the whole range of
# admission delays, they are just weighted very differently, so they
# still overlap.

lateness = 1 - df["hours_before_icu"].rank(pct=True)

study_weight = (1 - lateness) ** 2
study = df.sample(n=15000, weights=study_weight, random_state=0)

remaining = df.drop(study.index)
target_weight = lateness.loc[remaining.index] ** 2
target = remaining.sample(n=8000, weights=target_weight, random_state=0)

study = study.copy()
target = target.copy()
study["population"] = "study"
target["population"] = "target"

# %%
# Visualizing the shift
# ------------------------
#
# The delay before ICU admission is shifted markedly between the two
# populations, but they still overlap: some study patients are admitted
# late, and some target patients are admitted early.

import matplotlib.pyplot as plt
import seaborn as sns

both = pd.concat([study, target])

plt.figure()
sns.histplot(
    data=both, x="hours_before_icu", hue="population",
    stat="density", common_norm=False, bins=40, binrange=(-400, 50),
)
plt.xlim(-400, 50)
plt.xlabel("hours between hospital and ICU admission")
plt.title("Distribution shift in the delay before ICU admission")
plt.tight_layout()

# %%
# The other covariates are almost unaffected by the shift - this is a
# shift in one covariate, not a wholesale change of population.

plt.figure()
sns.histplot(
    data=both, x="diastolic_bp_mmhg", hue="population",
    stat="density", common_norm=False, bins=40,
)
plt.xlabel("diastolic blood pressure (mmHg)")
plt.title("Diastolic blood pressure is barely affected by the shift")
plt.tight_layout()

# %%
# Model fitting and prediction
# ===============================
#
# We fit both models on the study population only, keeping a held-out
# study test set: this lets us check that the models generalize within
# the study population, before checking how well they generalize to
# the target population.

from sklearn.model_selection import train_test_split

X_study = study.drop(columns=["sepsis", "population"])
y_study = study["sepsis"]
X_target = target.drop(columns=["sepsis", "population"])
y_target = target["sepsis"]

X_study_train, X_study_test, y_study_train, y_study_test = train_test_split(
    X_study, y_study, test_size=0.3, stratify=y_study, random_state=0
)

# %%
# Fit a linear model: logistic regression

from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

model_linear = tabular_pipeline(LogisticRegression())
model_linear.fit(X_study_train, y_study_train)

# %%
# Fit a non-linear model: gradient boosting

model_nonlinear = tabular_pipeline("classifier")
model_nonlinear.fit(X_study_train, y_study_train)

# %%
# Comparing predicted and observed risk, on both populations
# ---------------------------------------------------------------
#
# Rather than looking at a ranking metric such as AUC, we compare the
# *average* predicted risk to the *average* observed sepsis rate: this
# is exactly what a covariate shift can break, even for a model that
# still ranks patients in the right order.

y_pred_linear_study = model_linear.predict_proba(X_study_test)[:, 1]
y_pred_linear_target = model_linear.predict_proba(X_target)[:, 1]

y_pred_nonlinear_study = model_nonlinear.predict_proba(X_study_test)[:, 1]
y_pred_nonlinear_target = model_nonlinear.predict_proba(X_target)[:, 1]

comparison = pd.DataFrame({
    "population": ["study", "target", "study", "target"],
    "model": ["linear", "linear", "non-linear", "non-linear"],
    "observed_sepsis_rate": [
        y_study_test.mean(), y_target.mean(),
        y_study_test.mean(), y_target.mean(),
    ],
    "predicted_sepsis_rate": [
        y_pred_linear_study.mean(), y_pred_linear_target.mean(),
        y_pred_nonlinear_study.mean(), y_pred_nonlinear_target.mean(),
    ],
})
print(comparison.to_string(index=False))

# %%
# On the study population, both models predict a risk close to the
# observed one. On the target population, the linear model's predicted
# risk drops far below the actual observed rate: deployed on the
# shifted population, it badly under-estimates sepsis risk. The
# non-linear model also under-estimates risk on the target population,
# but far less severely.

# %%
# Partial dependence: why the linear model fails to generalize
# -------------------------------------------------------------------
#
# We plot, for each population, the two models' partial dependence on
# the delay before ICU admission (both fit once, on the study
# population only), next to the observed sepsis rate actually measured
# in that population. Since the delay is a near-continuous measurement,
# we group it into 12 bins of equal size before averaging observed
# rates.

import numpy as np
from sklearn.inspection import partial_dependence

# A fixed grid, shared by both populations, covering the range where
# most of the target population lies. Using the same grid for both
# panels, rather than letting each pick its own range, makes the two
# panels directly comparable.
delay_grid = np.linspace(-200, 25, 50)
delay_bin_edges = np.linspace(-200, 25, 13)

grid_background = pd.concat([X_study_test, X_target])
pd_linear = partial_dependence(
    model_linear, grid_background, features=["hours_before_icu"],
    custom_values={0: delay_grid},
)
pd_nonlinear = partial_dependence(
    model_nonlinear, grid_background, features=["hours_before_icu"],
    custom_values={0: delay_grid},
)

population_names = ["study", "target"]
population_X = [X_study_test, X_target]
population_y = [y_study_test, y_target]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for i in range(2):
    ax = axes[i]
    population_name = population_names[i]
    X_population = population_X[i]
    y_population = population_y[i]

    ax.plot(pd_linear["grid_values"][0], pd_linear["average"][0],
            linewidth=2, label="linear model")
    ax.plot(pd_nonlinear["grid_values"][0], pd_nonlinear["average"][0],
            linewidth=2, label="non-linear model")

    observed = pd.DataFrame({
        "hours_before_icu": X_population["hours_before_icu"],
        "sepsis": y_population.values,
    })
    observed["delay_bin"] = pd.cut(observed["hours_before_icu"], bins=delay_bin_edges)
    observed_by_bin = observed.groupby("delay_bin")[["hours_before_icu", "sepsis"]].mean()
    ax.plot(observed_by_bin["hours_before_icu"], observed_by_bin["sepsis"],
            color="black", linewidth=1, linestyle="--", marker="o", markersize=4,
            label="observed sepsis rate")

    ax.set_xlim(-200, 25)
    ax.set_xlabel("hours between hospital and ICU admission")
    ax.set_title(f"{population_name.capitalize()} population")

axes[0].set_ylabel("probability of sepsis")
axes[1].legend(fontsize=8)
fig.tight_layout()

# %%
# On the study population (left), most patients are admitted with
# little delay, and that is exactly where both curves track the
# observed rate. Away from that dense region, the two curves diverge
# from each other, but few study patients are there to notice. On the
# target population (right), where long delays are now common, that
# same region is where most patients sit, and the difference matters:
# the linear model's straight line drifts far below the observed rate,
# while the non-linear model - which could represent a non-monotonic
# link, and had at least *some* long-delay patients to learn from in
# the study population - stays much closer.
#
# This is the essence of covariate shift: nothing changed in *how*
# admission delay relates to sepsis risk, only *how often* each delay
# is observed. A model that only approximates that relationship well
# where the study data happens to be dense can fail badly once deployed
# where the data is dense elsewhere.
