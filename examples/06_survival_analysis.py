"""
Survival and time-to-event: accounting for censoring
======================================================

In the NHANES mortality data participant we know ``months_in_study``
(their follow-up duration) and ``event`` (whether they died, or were
still alive - "censored" - at the end of follow-up). We can use it in a
*time-to-event* question, studying the time people live, ie a *survival*
question.

And yet, there is a challenge: censoring. Participants enrolled towards
the end of the study simply have not been followed long enough to know if
or when they will die: their true time to death is unknown, only a lower
bound on it.

We first show what goes wrong if this censoring is ignored: predicting
"time to death" using only the participants who died. We then show how
a proper survival model - ``SurvivalBoost`` from the `hazardous
<https://soda-inria.github.io/hazardous/>`_ library - uses the censored
participants too, without needing to know their exact time of death.

Learning objectives and take home messages
-------------------------------------------

**When following individuals to observe an event (such as death), the
observation period may often be too small to observe this event for all
individuals. Dropping the individuals for which the event is not observed
creates a bias. The right solution is to use techniques known as
"survival analysis".**

"""

# %%
# The data: an epidemiological cohort
# -------------------------------------

import pandas as pd

df = pd.read_csv("nhanes_1999_2018_mortality.csv")

# %%
# A quick glance at the data
from skrub import TableReport
TableReport(df)

# %%
# Our covariates and target
X = df.drop(columns=["participant_id", "cycle", "months_in_study", "event"])
duration = df["months_in_study"]
is_dead = df["event"] == "deceased"

print("Number of participants:", len(df))
print("Deaths observed:", is_dead.sum())
print("Still alive at the end of follow-up (censored):", (~is_dead).sum())



# %%
# The wrong way: dropping the censored participants
# =======================================================
#
# A tempting shortcut is to only keep the participants who died, and
# fit an ordinary regression model to predict how many months they
# survived from their covariates. Participants still alive at the end
# of follow-up are dropped, since we do not know their exact time of
# death.

from sklearn.model_selection import train_test_split
from skrub import tabular_pipeline

X_dead = X[is_dead]
duration_dead = duration[is_dead]

X_dead_train, X_dead_test, duration_dead_train, duration_dead_test = train_test_split(
    X_dead, duration_dead, test_size=0.2, random_state=0
)

naive_model = tabular_pipeline("regressor")
naive_model.fit(X_dead_train, duration_dead_train)

# %%
# Evaluated the usual way, on a held-out set of participants who also
# died, this model looks reasonable.

from sklearn.metrics import mean_absolute_error, r2_score

predicted_dead_test = naive_model.predict(X_dead_test)
print(f"MAE on held-out deaths: {mean_absolute_error(duration_dead_test, predicted_dead_test):.1f} months")
print(f"R2 on held-out deaths: {r2_score(duration_dead_test, predicted_dead_test):.3f}")

# %%
# Why it is biased
# --------------------
#
# Now apply this same model to the participants it never saw: the ones
# who were still alive at the end of follow-up. For them, we do not
# know their true time to death, but we do know a lower bound on it:
# they survived at least ``months_in_study``, since that is how long
# they were observed to still be alive.
#
# If the model's predicted time to death is *less* than that lower
# bound, the prediction is not just inaccurate - it is logically
# impossible: the model predicts these participants should already be
# dead, when we know for a fact they were not.

X_censored = X[~is_dead]
already_survived = duration[~is_dead]
predicted_censored = naive_model.predict(X_censored)

impossible = predicted_censored < already_survived.values
print(f"Fraction of censored participants with an impossible prediction: {impossible.mean():.1%}")
print(f"Mean predicted time to death (censored): {predicted_censored.mean():.1f} months")
print(f"Mean months already survived (censored): {already_survived.mean():.1f} months")

# %%
# Almost 6 out of 10 censored participants get an impossible
# prediction. The figure below makes the mechanism clear: predicted
# time to death is squeezed into a narrow range around the middle of
# the follow-up window, while participants have already survived up to
# the full length of the study.

import matplotlib.pyplot as plt

plt.figure()
plt.hist(predicted_censored, bins=50, density=True, alpha=0.6,
         label="predicted time to death")
plt.hist(already_survived, bins=50, density=True, alpha=0.6,
         label="months already survived\n(known lower bound)")
plt.xlabel("months")
plt.ylabel("density")
plt.title("A model trained on deaths only, applied to censored participants")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# This is not a fluke of this particular model: it follows directly
# from dropping the censored participants. Among people who died during
# the study, none of them could have been the healthiest, longest-lived
# participants - those are exactly the ones still alive, and excluded.
# The model only ever learns from a population selected for dying
# within the follow-up window, so it has no way to predict a survival
# time longer than that window, no matter who it is asked about.

# %%
# Survival analysis: accounting for censoring properly
# ==========================================================
#
# A survival model uses *all* participants, including the censored
# ones, by working with two pieces of information for each of them:
# whether the event happened (``event``), and for how long they were
# observed (``duration``) - censored or not.

# %%
# A 'y' target that exposes censoring
# -----------------------------------
#
# We build 'y' vector that is specific to censoring models, exposing both
# the duration and if the event we're interesting (death, here) is
# observed or not after this duration.

y = pd.DataFrame({
    "event": is_dead.astype(int),
    "duration": duration,
})

# %%
# This two-column ``y`` is exactly the format ``SurvivalBoost`` (used
# below) expects: a dataframe with columns named ``"event"`` and
# ``"duration"``, ``"event"`` being 0 for a censored participant and a
# positive integer for an observed event. Passing ``y`` this way, instead
# of ``duration`` alone, is precisely what tells the model which
# participants are censored - it never sees an exact time of death for
# them, only that they were still alive at their recorded ``duration``.
# ``duration`` itself plays a double role depending on ``event``: for a
# death, it is the exact time to death; for a censored participant, it is
# only a lower bound on their (unknown) true time to death - the model
# has to treat these two cases differently, and ``event`` is what lets
# it tell them apart.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# %%
# The Kaplan-Meier estimator
# ------------------------------
#
# Before fitting any model, the Kaplan-Meier estimator gives a
# non-parametric estimate of the population's survival curve - the
# probability of still being alive at each point in time - using both
# the deaths and the censored participants. Rather than implementing
# this ourselves, we use ``scipy.stats``, which has built-in support for
# right-censored data through ``CensoredData``: we give it the exact
# durations for participants who died (``uncensored``) and the observed
# durations for participants still alive at the end of follow-up
# (``right``, for "right-censored"), and ``ecdf`` returns the
# Kaplan-Meier survival curve as its survival function ``.sf``.

from scipy.stats import CensoredData, ecdf


def kaplan_meier(duration_values, event_values):
    observed_duration = duration_values[event_values == 1]
    censored_duration = duration_values[event_values == 0]
    survival_function = ecdf(
        CensoredData(uncensored=observed_duration, right=censored_duration)
    ).sf
    time_points = [0] + list(survival_function.quantiles)
    survival_curve = [1.0] + list(survival_function.probabilities)
    return time_points, survival_curve


plt.figure()
time_points, survival_curve = kaplan_meier(y_train["duration"], y_train["event"])
plt.step(time_points, survival_curve, where="post", color="black", label="Whole population")

sex_colors = {"M": "tab:blue", "F": "tab:orange"}
for sex_value, color in sex_colors.items():
    is_sex = X_train["sex"] == sex_value
    time_points_sex, survival_curve_sex = kaplan_meier(
        y_train.loc[is_sex, "duration"], y_train.loc[is_sex, "event"]
    )
    plt.step(time_points_sex, survival_curve_sex, where="post", color=color,
             label=f"sex = {sex_value}")

plt.xlabel("Months")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier survival curves (training set)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Women have a consistently higher survival probability than men
# throughout the follow-up period - a well known epidemiological
# pattern, and a good sanity check that the estimator behaves sensibly.

# %%
# Fitting SurvivalBoost
# -------------------------
#
# ``SurvivalBoost`` is a gradient-boosted survival model: rather than
# predicting a single number, it predicts a whole survival curve for
# each participant, using every participant - censored or not - during
# training. It is described in Alberge, Maladiere, Grisel, Abécassis
# and Varoquaux (2025), `"Survival Models: Proper Scoring Rule and
# Stochastic Optimization with Competing Risks"
# <https://proceedings.mlr.press/v258/alberge25a.html>`_, Proceedings
# of the 28th International Conference on Artificial Intelligence and
# Statistics (AISTATS), PMLR 258:3619-3627.

from hazardous import SurvivalBoost

survival_model = tabular_pipeline(
    SurvivalBoost(n_iter=50, show_progressbar=False, random_state=0)
)
survival_model.fit(X_train, y_train)

# %%
# Predicted survival curves: younger vs older participants
# ---------------------------------------------------------------
#
# We compare the predicted survival curves of the 15 youngest and 15
# oldest participants in the test set.

survival_boost_model = survival_model.named_steps["survivalboost"]
times = survival_boost_model.time_grid_

X_test_transformed = survival_model[:-1].transform(X_test)
predicted_survival_test = survival_boost_model.predict_survival_function(X_test_transformed)

age_order = X_test["age"].values.argsort()
youngest_index = age_order[:15]
oldest_index = age_order[-15:]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for index in youngest_index:
    axes[0].plot(times, predicted_survival_test[index], color="tab:blue", alpha=0.5)
for index in oldest_index:
    axes[1].plot(times, predicted_survival_test[index], color="tab:red", alpha=0.5)

axes[0].set_title("15 youngest participants")
axes[1].set_title("15 oldest participants")
for ax in axes:
    ax.set_xlabel("Months")
axes[0].set_ylabel("Predicted survival probability")
fig.tight_layout()
plt.show()

# %%
# The youngest participants' predicted survival stays close to 1
# throughout follow-up, while the oldest participants' predicted
# survival declines much faster and more steeply - exactly the pattern
# we would expect, and something the naive regression from the first
# section has no way to express at all.

# %%
# Calibration: does the predicted risk match what actually happens?
# -----------------------------------------------------------------------
#
# The integrated Brier score summarizes, across the whole follow-up
# period, how far predicted survival probabilities are from the actual
# outcomes - lower is better.

from hazardous.metrics import integrated_brier_score_survival

ibs = integrated_brier_score_survival(y_train, y_test, predicted_survival_test, times)
print(f"Integrated Brier score (test set): {ibs:.3f}")

# %%
# No impossible predictions
# -----------------------------
#
# Finally, we repeat the same check as in the first section: for each
# censored participant, does the model at least acknowledge they were
# alive at the time we know they were? We read, for every test
# participant, the predicted survival probability at their own
# ``duration`` - the exact time of death for those who died, or the
# last known "still alive" time for those censored.

import numpy as np

survival_probability_at_own_time = np.zeros(len(X_test))
for i in range(len(X_test)):
    survival_probability_at_own_time[i] = np.interp(
        y_test["duration"].values[i], times, predicted_survival_test[i]
    )

is_censored_test = y_test["event"].values == 0
print(
    "Mean predicted survival probability at participants' own censoring time "
    f"(censored): {survival_probability_at_own_time[is_censored_test].mean():.3f}"
)
print(
    "Mean predicted survival probability at participants' own time of death "
    f"(deaths): {survival_probability_at_own_time[~is_censored_test].mean():.3f}"
)

# %%
# Censored participants get a mean predicted survival probability of
# about 0.9 at the exact time we know they were still alive - close to
# certainty, and correctly so. Participants who died get a
# substantially lower predicted survival probability at their own time
# of death. Unlike the naive regression, the survival model never
# contradicts what we already know about a participant's outcome - it
# simply has no mechanism to, since it was built from the start to
# represent that some participants' true time to event is only known to
# be "later than this".

# %%
# Another approach: finite horizon
# ---------------------------------
#
# Another way to avoid biases with such righ-censored data is to do a
# finite-horizon analysis: rather than computing the time that it takes
# for a given individual to die, choose a time horizon (eg 5 years), keep
# only individuals that have been observed more than this time, and
# classify whether after 5 years they were alive or not. This is what we
# did in the first notebook introducing this dataset. The first drawback
# of this approach is that it does not give an estimated time, but rather
# a yes/no answer after a given horizon. One can then use multiple
# horizons but, by treating each horizon separately, such an approach is
# less powerful statistically to capture the link between the covariates
# and the outcome: giving personnalized time-to-events for a given X. It
# is particularly inefficient at long time horizons, where the censoring
# depletes massively the cohort. A survival analysis model can learn the
# link between covariates X and survival for small time horizons where
# there are many non censored observations, and extrapolate it to large
# time horizons.

