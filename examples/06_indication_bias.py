"""
Indication bias: challenge in reasonning on interventions
============================================================

In health we often want to do more than mere prediction: we would like to
intervene, change something on the patient or the care, to improve a
health outcome. For instance, instance, we are predicting sepsis given
the current situation, and we'd like to avoid it, and thus change our
care strategy for the patient. We will call this change a "treatment".

In routine care, the treatment is a consequence of many things, such as
the patients baseline health or prior history. Often, the sickest
patients get a different treatment than the more healthy one. This is
called *indication bias* (or confounding by indication): the indication
for treatment is itself a marker of worse prognosis, so a naive
comparison of treated and untreated patients mixes up the effect of the
treatment with the effect of whatever made clinicians choose to treat.
This is unlike in a randomized trial, who gets the treatment is decided
by a coin flip.

As a consequence of indication bias, **using predictive models to reason
on whether or not to assign a treatment, to intervene, is challenging.**


Here, we illustrate these challenge, as well as possible solutions
(though there is no magic bullet).
Real ICU data has no known "true" treatment effect to check our
methods against, so here we use real PhysioNet sepsis covariates, but
simulate the treatment assignment and the outcome ourselves, with a
treatment effect we fix in advance. This lets us compare every
estimate to a known ground truth - something we could never do with
purely real data - while keeping the confounding realistic. This
follows the discussion in Doutreligne and Varoquaux (2025), "How to
select predictive models for decision-making or causal inference",
GigaScience, which shows that a model's *predictive* accuracy does not
tell you how good it will be at estimating a *causal* effect.

|

**Reference** Useful big-picture reading: Abécassis, J., Dumas, É.,
Alberge, J., & Varoquaux, G. (2025). *From prediction to prescription:
Machine learning and causal inference for the heterogeneous treatment
effect.* Annual Review of Biomedical Data Science, 8.
https://doi.org/10.1146/annurev-biodatasci-103123-095750

"""

# %%
# Load the data and build a severity score
# ---------------------------------------------
#
# We use age and five vitals recorded during the first 24h in the ICU
# to build a simple severity score: each vital is standardized (so it
# has mean 0 and standard deviation 1 across patients), then added up,
# with blood pressure and oxygen saturation counting *against* severity
# (lower values are worse for those two).

import pandas as pd

df = pd.read_csv("physionet_sepsis_causal.csv")
df = df.dropna(subset=[
    "heart_rate_bpm", "resp_rate", "temp_celsius",
    "mean_arterial_bp_mmhg", "o2_sat_pct", "age",
])


def standardize(values):
    return (values - values.mean()) / values.std()


fever_or_hypothermia = (df["temp_celsius"] - 37).abs()

severity = (
    standardize(df["resp_rate"])
    + standardize(df["heart_rate_bpm"])
    + standardize(fever_or_hypothermia)
    - standardize(df["mean_arterial_bp_mmhg"])
    - standardize(df["o2_sat_pct"])
    + standardize(df["age"])
)
severity = standardize(severity)
df["severity"] = severity

print(df["severity"].describe())

# %%
# Simulating an indication-biased treatment
# ---------------------------------------------
#
# We simulate a binary treatment - think of it as an aggressive
# intervention such as early vasopressor use - whose probability
# increases with severity: sicker patients are more likely to receive
# it, exactly as in real practice.

import numpy as np

rng = np.random.RandomState(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


propensity_true = sigmoid(-0.5 + 1.5 * df["severity"])
treatment = rng.binomial(1, propensity_true)
df["treatment"] = treatment

print("Fraction treated:", treatment.mean())
print("Mean severity, treated:", df.loc[treatment == 1, "severity"].mean())
print("Mean severity, untreated:", df.loc[treatment == 0, "severity"].mean())

# %%
# Treated patients are, on average, considerably sicker than untreated
# ones - by construction, but this is exactly the pattern indication
# bias produces in real data too.

import matplotlib.pyplot as plt

plt.figure()
plt.hist(df.loc[treatment == 0, "severity"], bins=40, density=True, alpha=0.6, label="untreated")
plt.hist(df.loc[treatment == 1, "severity"], bins=40, density=True, alpha=0.6, label="treated")
plt.xlabel("severity score")
plt.ylabel("density")
plt.title("Treated patients are sicker than untreated ones")
plt.legend()
plt.tight_layout()

# %%
# Simulating an outcome with a known treatment effect
# ---------------------------------------------------------
#
# We simulate a continuous outcome - think of it as a 48h organ-failure
# score, higher is worse - that depends on severity (sicker patients do
# worse) and on the treatment, with a *fixed, known* effect: the
# treatment reduces the outcome by exactly 1 point on average. This
# ``true_effect`` is what every method below is trying to recover.

true_effect = -1.0
noise = rng.normal(0, 1.0, size=len(df))
outcome = 2.0 * df["severity"] + true_effect * treatment + noise
df["outcome"] = outcome

# %%
# The wrong way: a naive difference in means
# ===============================================
#
# The simplest possible analysis compares the average outcome of
# treated and untreated patients directly.

naive_effect = df.loc[treatment == 1, "outcome"].mean() - df.loc[treatment == 0, "outcome"].mean()
print(f"True effect:            {true_effect:+.2f}")
print(f"Naive difference in means: {naive_effect:+.2f}")

# %%
# The naive estimate has the *wrong sign*: it suggests the treatment is
# harmful, when by construction it is beneficial. This is not a subtle
# statistical error - it is the direct consequence of comparing two
# groups that were never comparable to begin with: the treated group is
# sicker, and sicker patients do worse regardless of treatment.

# %%
# Counterfactual reasoning: G-formula
# ========================================
#
# The idea behind the G-formula (also called "outcome regression") is
# to model the outcome from the covariates *and* the treatment, then
# ask the model to predict the outcome for every patient twice: once as
# if they had been treated, and once as if they had not. Averaging the
# difference between these two counterfactual predictions gives an
# estimate of the treatment effect, adjusted for how the covariates
# relate to both treatment and outcome.

covariate_columns = [
    "age", "sex", "heart_rate_bpm", "resp_rate", "temp_celsius",
    "mean_arterial_bp_mmhg", "o2_sat_pct", "wbc_count", "creatinine_mgdl",
]

X = df[covariate_columns].copy()
X["treatment"] = df["treatment"]
y = df["outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def g_formula_effect(fitted_pipeline, X_all):
    X_treated = X_all.copy()
    X_treated["treatment"] = 1
    X_untreated = X_all.copy()
    X_untreated["treatment"] = 0
    predicted_treated = fitted_pipeline.predict(X_treated)
    predicted_untreated = fitted_pipeline.predict(X_untreated)
    return (predicted_treated - predicted_untreated).mean()


# %%
# The challenge of model selection
# -------------------------------------
#
# There is no shortage of models we could use for the outcome
# regression. We try four, from the simplest to the most flexible, and
# for each one we report two very different things: how well it
# predicts the *observed* outcome (a held-out R2, the usual way to pick
# a model), and what treatment effect it implies through the G-formula.

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from skrub import tabular_pipeline

model_names = ["linear regression", "shallow tree", "random forest", "gradient boosting"]
model_estimators = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=3, random_state=0),
    RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0),
    HistGradientBoostingRegressor(random_state=0),
]

results = []
for i in range(len(model_names)):
    pipeline = tabular_pipeline(model_estimators[i])
    pipeline.fit(X_train, y_train)

    predicted_test = pipeline.predict(X_test)
    held_out_r2 = r2_score(y_test, predicted_test)
    g_formula_ate = g_formula_effect(pipeline, X)

    results.append({
        "model": model_names[i],
        "held_out_r2": held_out_r2,
        "g_formula_ate": g_formula_ate,
    })

results = pd.DataFrame(results)
print(results.to_string(index=False))
print(f"\nTrue effect: {true_effect:+.2f}")
print(f"Naive difference in means: {naive_effect:+.2f}")

# %%
# Random forest and linear regression reach a similar held-out R2, yet
# their G-formula estimates are nowhere close to each other: one of
# them recovers a treatment effect not far from the truth, the other
# estimates an effect close to zero. Predictive accuracy on the
# observed outcome does not tell us which model to trust for the
# causal question - exactly the point made by Doutreligne and
# Varoquaux: good prediction and good causal estimation are different
# goals, and optimizing for one does not guarantee the other.
#
# The gradient boosting model happens to combine strong held-out
# prediction with a G-formula estimate close to the true effect - but
# nothing in the prediction score alone told us this in advance.

# %%
# Inverse probability weighting
# ==================================
#
# A different strategy models the *treatment* rather than the outcome:
# for each patient, estimate their probability of being treated given
# their covariates (their "propensity score"), then reweight patients
# by the inverse of that probability. This corrects for the fact that
# sicker patients are over-represented among the treated.

from sklearn.ensemble import HistGradientBoostingClassifier

propensity_pipeline = tabular_pipeline(HistGradientBoostingClassifier(random_state=0))
propensity_pipeline.fit(df[covariate_columns], df["treatment"])
propensity_score = propensity_pipeline.predict_proba(df[covariate_columns])[:, 1]
propensity_score = np.clip(propensity_score, 0.02, 0.98)

# %%
# Before using these propensity scores, we check *overlap*: treated and
# untreated patients should have overlapping ranges of propensity
# scores. If some patients are almost never treated, or almost always
# treated, there is not enough information to compare them, no matter
# how clever the method.

plt.figure()
plt.hist(propensity_score[treatment == 0], bins=40, density=True, alpha=0.6, label="untreated")
plt.hist(propensity_score[treatment == 1], bins=40, density=True, alpha=0.6, label="treated")
plt.xlabel("estimated propensity score")
plt.ylabel("density")
plt.title("Overlap between treated and untreated propensity scores")
plt.legend()
plt.tight_layout()

# %%
# The two distributions overlap substantially, though treated patients
# skew towards higher propensity scores, as expected. This is enough
# overlap for inverse probability weighting to be meaningful.

ipw_effect = np.mean(
    treatment * df["outcome"] / propensity_score
    - (1 - treatment) * df["outcome"] / (1 - propensity_score)
)
print(f"True effect:                   {true_effect:+.2f}")
print(f"Naive difference in means:     {naive_effect:+.2f}")
print(f"Inverse probability weighting: {ipw_effect:+.2f}")

# %%
# Doubly robust estimation
# =============================
#
# Doubly robust estimators combine an outcome model and a propensity
# model: they start from the G-formula estimate, then add a correction
# term, weighted by the inverse propensity score, for how wrong the
# outcome model was on the patients actually observed. The appeal is
# that the estimate stays valid if *either* the outcome model or the
# propensity model is reasonable - not necessarily both.

outcome_pipeline = tabular_pipeline(HistGradientBoostingRegressor(random_state=0))
outcome_pipeline.fit(X, y)

X_treated_all = X.copy()
X_treated_all["treatment"] = 1
X_untreated_all = X.copy()
X_untreated_all["treatment"] = 0
predicted_if_treated = outcome_pipeline.predict(X_treated_all)
predicted_if_untreated = outcome_pipeline.predict(X_untreated_all)

doubly_robust_terms = (
    predicted_if_treated - predicted_if_untreated
    + treatment * (df["outcome"] - predicted_if_treated) / propensity_score
    - (1 - treatment) * (df["outcome"] - predicted_if_untreated) / (1 - propensity_score)
)
doubly_robust_effect = doubly_robust_terms.mean()

# %%
# Putting all the estimates side by side against the true effect shows
# how much a careless analysis can mislead, and how much closer the
# methods designed for causal questions get.

summary = pd.DataFrame({
    "method": [
        "true effect", "naive difference in means", "inverse probability weighting",
        "G-formula (gradient boosting)", "doubly robust",
    ],
    "estimated_effect": [
        true_effect, naive_effect, ipw_effect,
        results.loc[results["model"] == "gradient boosting", "g_formula_ate"].iloc[0],
        doubly_robust_effect,
    ],
})
print(summary.to_string(index=False))

plt.figure()
plt.barh(summary["method"], summary["estimated_effect"])
plt.axvline(true_effect, color="black", linestyle="--", label="true effect")
plt.xlabel("estimated treatment effect")
plt.legend()
plt.tight_layout()

# %%
# Every method that accounts for indication bias - inverse probability
# weighting, the G-formula, and the doubly robust estimator - correctly
# identifies the treatment as beneficial, and lands much closer to the
# true effect than the naive comparison. None of them needed to know
# the true propensity or outcome model: they only needed a rich enough
# set of covariates to capture what drove the clinicians' decision to
# treat. Had we omitted the vitals that make up our severity score,
# none of these methods could have corrected for the bias - adjustment
# can only compensate for confounding that is actually measured.
