.. Health ML Tutorial documentation master file.

==================================================================
An introduction to Machine Learning for health and epidemiology
==================================================================

Materials to understand concepts important to Machine Learning in Health:
practical Python examples with interactive Jupyter notebooks that run
entirely in your browser.

This course addresses two topics:

* **How machine-learning tools provide flexible models** that can relate a
  health outcome (called "target" or "y" in machine learning) to
  covariates (called "features" or "X" in machine learning)

* **Multiple biases that can arise in the data** and prevent the success
  of models, flexible or not

The course is based on notebooks that run on real health data, giving the
practical elements to tackle the complexity of real statistical learning
questions in health.

Understanding data and learning on it
==========================================

These first notebooks introduce the two health datasets used throughout
the course, and the core machine-learning ideas: fitting a model,
checking it on held-out data, comparing a linear and a non-linear
model, and the trade-off between under-fitting and over-fitting.

.. sidebar:: Runnable code

    Each page has **Launch** buttons at the top left that open the
    corresponding notebook in your browser with JupyterLite.

.. minigallery:: ../examples/01_predict_5yr_mortality.py ../examples/02_predict_sepsis.py ../examples/03_overfit_underfit.py

Biases that can break a model's utility
========================================

These next notebooks discuss the biases that can silently distort an
analysis: ignoring censoring, a shift between the population a model was
built on and the population it is used on, and confounding by indication
when reasoning about interventions.

.. minigallery:: ../examples/04_covariate_shift.py ../examples/05_survival_analysis.py ../examples/06_indication_bias.py



