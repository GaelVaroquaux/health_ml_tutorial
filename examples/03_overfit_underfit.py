"""
Under-fitting, over-fitting, and the role of sample size
============================================================

Using the same SUPPORT2 ICU-mortality data as the previous example, we
push a single, easy-to-reason-about "flexibility" knob - the maximum
depth of a decision tree - hard in both directions, to see under-fitting
and over-fitting happen concretely. We then look at why averaging many
trees (a random forest) tames over-fitting, and how more data pushes the
same model from over-fitting to a good fit.
"""

# %%
# Load the data and split train / test
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("support2_icu_mortality.csv")
X = df.drop(columns=["death"])
y = df["death"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# %%
# A single decision tree: one knob for flexibility
# ----------------------------------------------------
#
# A decision tree predicts by asking a sequence of yes/no questions
# about the covariates. Its ``max_depth`` sets how many questions can be
# chained before making a prediction: a shallow tree can only represent
# a simple, rigid rule, while a deep tree can carve out a rule specific
# to almost every individual patient in the training set.
#
# We fit one tree per depth, and score it both on the data it was
# trained on and on the held-out test set.

from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from skrub import tabular_pipeline

max_depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]

train_scores, test_scores = [], []
for max_depth in max_depths:
    model = tabular_pipeline(DecisionTreeClassifier(max_depth=max_depth, random_state=0))
    model.fit(X_train, y_train)
    train_scores.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    test_scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(f"max_depth={str(max_depth):5s}  train AUC={train_scores[-1]:.3f}  test AUC={test_scores[-1]:.3f}")

# %%
# Plotting train and test performance side by side makes the two
# failure modes obvious.

import matplotlib.pyplot as plt

# A plain integer x-axis, with "None" (unlimited depth) placed one step
# after the deepest limited depth.
x_positions = range(len(max_depths))
x_labels = [str(d) if d is not None else "None\n(unlimited)" for d in max_depths]

plt.plot(x_positions, train_scores, marker="o", label="Train AUC")
plt.plot(x_positions, test_scores, marker="o", label="Test AUC")
plt.xticks(list(x_positions), x_labels)
plt.xlabel("max_depth (tree flexibility)")
plt.ylabel("AUC")
plt.title("A single decision tree: under-fitting to over-fitting")
plt.legend()
plt.tight_layout()

# %%
# On the left (small ``max_depth``), the tree is too rigid: it cannot
# even fit the training data well (train AUC is low), and it does no
# better - or even worse - on the test set. This is under-fitting.
#
# On the right (``max_depth=None``), the tree perfectly memorizes the
# training data (train AUC = 1.0), but test performance collapses to
# barely better than chance. This is over-fitting: the tree has learned
# the noise specific to the training patients, not the pattern that
# generalizes.
#
# In between, around ``max_depth=6``, test performance peaks: the tree
# is flexible enough to capture real structure, but not so flexible
# that it chases noise. Only the held-out test set lets us find this
# sweet spot - the training score alone keeps improving all the way to
# the right, and would be a misleading guide on its own.

# %%
# What the extremes actually look like
# ---------------------------------------
#
# To make "under-fit" and "over-fit" concrete rather than abstract, we
# look at the partial dependence of the ICU-intensity score - the
# clearest non-linear feature in this dataset - at three depths: too
# shallow, the sweet spot, and unlimited.

from sklearn.inspection import partial_dependence

depths_to_show = {"under-fit (max_depth=1)": 1, "sweet spot (max_depth=6)": 6, "over-fit (max_depth=None)": None}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, (label, max_depth) in zip(axes, depths_to_show.items()):
    model = tabular_pipeline(DecisionTreeClassifier(max_depth=max_depth, random_state=0))
    model.fit(X_train, y_train)
    pd_result = partial_dependence(
        model, X_train, features=["icu_intensity_score"], grid_resolution=100
    )
    ax.plot(pd_result["grid_values"][0], pd_result["average"][0], color="black")
    ax.set_xlabel("ICU-intensity score")
    ax.set_title(label)

axes[0].set_ylabel("predicted probability of death")
fig.tight_layout()

# %%
# The under-fit tree is completely flat: with only one yes/no question
# to spend, it spent it on a different, more informative feature
# (cancer status) and has nothing left to represent a link with
# ICU-intensity score at all. The over-fit tree is a wild, jagged
# staircase, chasing every idiosyncrasy of the training patients. The
# sweet-spot tree in the middle traces a much more plausible, smoothly
# rising staircase.

# %%
# Averaging many trees tames over-fitting
# -------------------------------------------
#
# A random forest fits many trees, each on a bootstrap resample of the
# data, and averages their predictions. This averaging cancels out a lot
# of the noise any single deep tree picks up, without giving up
# flexibility. We repeat the same ``max_depth`` sweep with a random
# forest of 300 trees, and compare its test curve to the single tree's.

from sklearn.ensemble import RandomForestClassifier

forest_test_scores = []
for max_depth in max_depths:
    model = tabular_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=max_depth, random_state=0)
    )
    model.fit(X_train, y_train)
    forest_test_scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

plt.figure()
plt.plot(x_positions, test_scores, marker="o", label="Single tree, test AUC")
plt.plot(x_positions, forest_test_scores, marker="o", label="Random forest (300 trees), test AUC")
plt.xticks(list(x_positions), x_labels)
plt.xlabel("max_depth (tree flexibility)")
plt.ylabel("Test AUC")
plt.title("Averaging many trees is far more robust to over-fitting")
plt.legend()
plt.tight_layout()

# %%
# The random forest's test performance barely drops at large depths,
# unlike the single tree's collapse. This is why the previous example
# used a random forest rather than a single tree: it lets us pick a
# reasonably large ``max_depth`` without having to get the value exactly
# right.

# %%
# Asymptotics: more data turns over-fitting into a good fit
# --------------------------------------------------------------
#
# Over-fitting is not just a property of the model: it is a property of
# the model *relative to how much data we have*. We fix a single tree
# at ``max_depth=6`` - the sweet spot we found above with the full
# training set - and refit it on growing subsets of the training data,
# to see what happens with less, or more, data.

sample_sizes = [100, 200, 400, 800, 1600, 3200, len(X_train)]

train_scores_by_n, test_scores_by_n = [], []
for n in sample_sizes:
    X_sub = X_train.iloc[:n]
    y_sub = y_train.iloc[:n]
    model = tabular_pipeline(DecisionTreeClassifier(max_depth=6, random_state=0))
    model.fit(X_sub, y_sub)
    train_scores_by_n.append(roc_auc_score(y_sub, model.predict_proba(X_sub)[:, 1]))
    test_scores_by_n.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(f"n={n:5d}  train AUC={train_scores_by_n[-1]:.3f}  test AUC={test_scores_by_n[-1]:.3f}")

plt.figure()
plt.plot(sample_sizes, train_scores_by_n, marker="o", label="Train AUC")
plt.plot(sample_sizes, test_scores_by_n, marker="o", label="Test AUC")
plt.xscale("log")
plt.xlabel("Number of training samples")
plt.ylabel("AUC")
plt.title("A depth-6 tree: over-fitting shrinks as training data grows")
plt.legend()
plt.tight_layout()

# %%
# With only 100 training patients, this same depth-6 tree over-fits
# badly (train AUC far above test AUC): there simply are not enough
# patients per leaf of the tree to estimate a reliable probability. As
# the training set grows towards its full size, train and test AUC
# converge, and test performance keeps improving. The model's
# flexibility did not change - only the amount of data did. This is
# the asymptotic behaviour that makes flexible, non-parametric models
# useful in the first place: give them enough data, and they can
# approach the true underlying relationship.
