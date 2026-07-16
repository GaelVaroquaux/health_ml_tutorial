.. Health ML Tutorial documentation master file.

==================================================================
An introduction to Machine Learning for health and epidemiology
==================================================================

Understand concepts important to Machine Learning in Health with
notebooks that run on real health data, giving the practical elements to
tackle the complexity of real statistical learning questions in health.

Machine-learning: flexible models for health data
===================================================

**Machine-learning tools provide flexible models** that can relate a
health outcome (called "target" or "y" in machine learning) to covariates
(called "features" or "X" in machine learning).

These first notebooks introduce the two health datasets used throughout
the course, and the core machine-learning ideas: fitting a model,
checking it on held-out data, comparing a linear and a non-linear
model, and the trade-off between under-fitting and over-fitting.

.. sidebar:: Runnable code

    Each page has **Launch** buttons at the top left that open the
    corresponding notebook in your browser with JupyterLite.

.. minigallery:: ../examples/01_predict_icu_sepsis.py ../examples/02_overfit_underfit.py ../examples/03_nhanes_predict_5yr_mortality.py

Biases that can break a model's utility
========================================

**Multiple biases that can arise in the data** and prevent the success of
models, flexible or not. These next notebooks discuss these biases, that can silently distort an
analysis.

.. minigallery:: ../examples/04_covariate_shift.py ../examples/05_indication_bias.py ../examples/06_survival_analysis.py



