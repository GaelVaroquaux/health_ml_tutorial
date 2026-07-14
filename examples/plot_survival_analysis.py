"""
Survival Analysis with SurvivalBoost
======================================

Survival analysis models *time-to-event* outcomes, such as the time until a
patient experiences a clinical event (e.g. death, disease recurrence, hospital
re-admission).  Unlike ordinary regression, many observations are *censored*:
we know the event had not yet occurred at the last follow-up, but we do not
know the exact time it eventually will.

This example demonstrates a complete survival analysis workflow:

1. Simulate a realistic clinical dataset with censoring.
2. Explore the data with a Kaplan–Meier survival curve.
3. Fit a gradient-boosted survival model (``SurvivalBoost`` from the
   `hazardous <https://soda-inria.github.io/hazardous/>`_ library).
4. Compare predicted survival curves for high-risk vs low-risk patients.
5. Evaluate model calibration with the time-dependent Brier score.

.. note::

   Click the **Launch** button above to run this notebook interactively
   in your browser — no installation needed!
"""

# %%
# Simulating a clinical dataset
# ------------------------------
#
# We use the ``load_breast_cancer`` dataset from scikit-learn as a starting
# point.  A logistic regression model provides a *risk score* for each
# patient (probability of malignancy).  We then simulate survival times so
# that higher-risk patients tend to have shorter survival — a realistic
# clinical scenario.
#
# Each patient has:
#
# * ``duration`` — the observed follow-up time (in days).
# * ``event`` — 1 if the event occurred before censoring, 0 otherwise.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Load features
data = load_breast_cancer(as_frame=True)
X, y_clf = data.data, data.target
n = len(X)

# Derive a risk score from a quick logistic regression
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
lr = LogisticRegression(max_iter=500, solver="lbfgs", random_state=0)
lr.fit(Xs, y_clf)
# risk: probability of malignant class (class 0 in the dataset)
risk = lr.predict_proba(Xs)[:, 0]

# Simulate true event times: higher risk → shorter expected survival
true_duration = rng.exponential(scale=1000 * (1 - risk + 0.1), size=n).astype(int) + 1

# Administrative censoring: each patient is followed for a random period
censor_time = rng.integers(100, 1500, size=n)
event_observed = true_duration < censor_time
duration_obs = np.where(event_observed, true_duration, censor_time)

y_surv = pd.DataFrame({
    "event": event_observed.astype(int),
    "duration": duration_obs,
})

print(f"Patients: {n}")
print(f"Event rate: {y_surv['event'].mean():.1%}")
print(f"Median follow-up: {y_surv['duration'].median():.0f} days")

# %%
# Train / test split
# ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_surv, test_size=0.3, random_state=0
)

# %%
# Kaplan–Meier estimator
# -----------------------
#
# The Kaplan–Meier (KM) estimator is the non-parametric baseline in survival
# analysis.  It estimates the probability of surviving beyond each observed
# event time without any covariate information.
#
# We split patients into two groups based on their risk score and overlay
# their KM curves to confirm that the simulated data has the expected
# separation.


def kaplan_meier(duration, event):
    """Return (times, survival_probability) arrays for the KM estimator."""
    order = np.argsort(duration)
    t = duration[order]
    e = event[order]
    n_at_risk = np.arange(len(t), 0, -1)
    # product-limit estimator
    with np.errstate(invalid="ignore"):
        km = np.cumprod(np.where(e == 1, 1 - 1 / n_at_risk, 1.0))
    # prepend time 0 with S(0)=1
    return np.concatenate([[0], t]), np.concatenate([[1.0], km])


# Use the training set so evaluation remains independent
risk_train = lr.predict_proba(scaler.transform(X_train))[:, 0]
median_risk = np.median(risk_train)

high_mask = risk_train >= median_risk
low_mask = ~high_mask

fig, ax = plt.subplots(figsize=(7, 4))

for mask, label, colour in [
    (low_mask, "Low risk (benign)", "tab:blue"),
    (high_mask, "High risk (malignant)", "tab:red"),
]:
    t_km, s_km = kaplan_meier(
        y_train.loc[mask, "duration"].values,
        y_train.loc[mask, "event"].values,
    )
    ax.step(t_km, s_km, where="post", color=colour, label=label)

ax.set_xlabel("Days")
ax.set_ylabel("Survival probability S(t)")
ax.set_title("Kaplan–Meier curves by risk group (training set)")
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
plt.show()

# %%
# Fitting SurvivalBoost
# ----------------------
#
# ``SurvivalBoost`` from the ``hazardous`` library is a gradient-boosted model
# tailored for survival analysis.  It directly estimates the *survival
# function* S(t) = P(T > t) for each patient across a grid of time points.
#
# The model accepts a ``y`` DataFrame with two columns:
#
# * ``"event"`` — 0 (censored) or a positive integer identifying the event
#   type (here 1 = event occurred).
# * ``"duration"`` — observed follow-up time.

from hazardous import SurvivalBoost

model = SurvivalBoost(
    n_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=31,
    show_progressbar=False,
    random_state=0,
)
model.fit(X_train, y_train)

times = model.time_grid_  # the 100 time points used by the model

print(f"Time grid: {times[0]:.0f} – {times[-1]:.0f} days ({len(times)} points)")

# %%
# Predicted survival curves: high-risk vs low-risk patients
# ----------------------------------------------------------
#
# We select the 10 highest- and 10 lowest-risk patients from the test set
# (based on the logistic-regression score) and plot their individual
# survival curves predicted by SurvivalBoost.

surv_pred = model.predict_survival_function(X_test)  # shape (n_test, n_times)

risk_test = lr.predict_proba(scaler.transform(X_test))[:, 0]
high_idx = np.argsort(risk_test)[-10:]   # 10 highest-risk patients
low_idx = np.argsort(risk_test)[:10]     # 10 lowest-risk patients

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, idx, label, colour in [
    (axes[0], low_idx, "Low-risk patients", "tab:blue"),
    (axes[1], high_idx, "High-risk patients", "tab:red"),
]:
    for i in idx:
        ax.plot(times, surv_pred[i], color=colour, alpha=0.4, linewidth=1)
    # bold mean curve
    ax.plot(times, surv_pred[idx].mean(axis=0), color=colour,
            linewidth=2.5, label="Mean")
    ax.set_title(label)
    ax.set_xlabel("Days")
    ax.set_ylabel("S(t)")
    ax.set_ylim(0, 1.05)
    ax.legend()

fig.suptitle("SurvivalBoost: predicted survival functions (test set)", y=1.01)
fig.tight_layout()
plt.show()

# %%
# Evaluating calibration: the Brier score
# ----------------------------------------
#
# The *Brier score* at time *t* measures the average squared difference
# between the predicted survival probability and the actual outcome
# (1 if the patient survived beyond *t*, 0 otherwise), corrected for
# informative censoring using inverse probability of censoring weighting
# (IPCW).
#
# A lower score indicates better calibration.  As a reference, a model
# that always predicts S(t) = 0.5 achieves a Brier score of 0.25.
#
# We also compute the *Integrated Brier Score* (IBS), the time-average of
# the Brier score curve.

from hazardous.metrics import brier_score_survival

bs = brier_score_survival(y_train, y_test, surv_pred, times)

# IBS via trapezoidal integration (compatible with all numpy versions)
ibs = np.trapezoid(bs, times) / (times[-1] - times[0])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, bs, color="steelblue")
ax.axhline(ibs, color="steelblue", linestyle="--",
           label=f"IBS = {ibs:.3f}")
ax.axhline(0.25, color="grey", linestyle=":", label="Reference (S=0.5)")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Brier score")
ax.set_title("Time-dependent Brier score (test set)")
ax.legend()
ax.set_ylim(0, 0.3)
fig.tight_layout()
plt.show()

print(f"Integrated Brier Score (IBS): {ibs:.3f}")
