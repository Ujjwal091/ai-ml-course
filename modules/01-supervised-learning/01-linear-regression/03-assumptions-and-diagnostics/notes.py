# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Regression — Assumptions & Diagnostics
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ujjwal091/ai-ml-course/blob/main/modules/01-supervised-learning/01-linear-regression/03-assumptions-and-diagnostics/notes.ipynb)
#
# *Part 3 of 3 on Linear Regression, continuing from
# [Multivariate Regression & Evaluation](../02-multivariate-regression/notes.ipynb).*
#
# **Last time:** using every feature at once, outliers, Adjusted R², StatsModels.
# **Today's central question:** a model with good performance isn't always a *trustworthy* model — how do we verify
# one?
#

# %% [markdown]
# ## 1. Recap, and loading the data again
#
# Self-contained like the rest of this series — reloading the same standardized Cars24 dataset used throughout.
#

# %%
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_URL = "https://raw.githubusercontent.com/28101991SUNNY/DSML-Classical-Machine-Learning-1/main/cars24-car-price-clean.csv"
DATA_PATH = "data/cars24-car-price-clean.csv"

if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)

df = pd.read_csv(DATA_PATH)
df.head()

# %% [markdown]
# ## 2. The 5 assumptions of Linear Regression
#
# A high R² doesn't automatically mean a *trustworthy* model. Linear Regression's coefficients, p-values, and
# confidence intervals are only valid when the data satisfies five assumptions underneath it. Violating them
# doesn't always visibly break the model's predictions — but it can silently invalidate everything you'd want to
# say *about* those predictions.
#
# | # | Assumption | Severity | What it means |
# |---|---|---|---|
# | 1 | Linearity | 🔴 Critical | Each feature relates to the target in a straight line |
# | 2 | No multicollinearity | 🔴 Critical | Features aren't linearly derivable from each other |
# | 3 | Normally distributed errors | 🟠 Important | Residuals follow a bell curve centered at zero |
# | 4 | Error independence | 🟠 Important | Residuals show no pattern or correlation with each other |
# | 5 | Homoscedasticity | 🟠 Important | Residual spread stays roughly constant across all predictions |
#
# **Assumption 1 in a bit more detail — Linearity.** If the true relationship is a curve (an S-curve, an
# exponential, a wave), a straight line structurally cannot capture it, no matter how well you fit `m` and `c`.
#
# - *How to check:* scatter plots of each feature against the target; residual plots — a visible pattern in the
#   residuals means linearity is violated.
# - *How to fix:* this is exactly what [Polynomial Regression](../../02-polynomial-regression/notes.ipynb) is for —
#   transforming the features into higher-order terms so a linear model can still fit the (now-linear-in-the-new-
#   features) relationship.
# - *Real examples of non-linear relationships:* energy consumption vs. year (S-curve), sales vs. ad spend
#   (diminishing returns), blood pressure vs. BMI.
#
# The rest of this notebook walks through Assumptions 2 through 5 one at a time, with the actual detection code for
# each.
#

# %% [markdown]
# ## 3. Multicollinearity and the Variance Inflation Factor (VIF)
#
# **Multicollinearity** happens when one feature can be predicted from the others — e.g. if
# $x_2 = 3x_3 + 2x_4$, then $x_2$ carries no information the model doesn't already have from $x_3$ and $x_4$. The
# model can no longer isolate $x_2$'s *own* effect, so its coefficient becomes unstable and unreliable.
#
# **In our dataset:** `year` and `age` are almost perfectly correlated — `age` is just `current_year − year`.
# Including both leaves the model unable to tell which one is actually driving the price.
#
# **Symptoms:** unexpectedly large coefficients (or the wrong sign), high standard errors despite a good overall
# R², a large condition number in a StatsModels summary.
#
# ### 3.1 How VIF works
#
# For each feature $x_j$: treat it as the *target*, fit a regression predicting it from every other feature, and
# look at that regression's $R^2_j$ — a high $R^2_j$ means $x_j$ is well-explained by the rest, i.e. redundant.
#
# $$
# \text{VIF}_j = \frac{1}{1 - R_j^2}
# $$
#
# | VIF | Interpretation | Action |
# |---|---|---|
# | > 10 | Very high multicollinearity | Remove |
# | 5 – 10 | High | Consider removing |
# | < 5 | Acceptable | Keep |
#

# %%
X_tr_vif, X_te_vif, y_tr_vif, y_te_vif = train_test_split(
    df[df.columns.drop('selling_price')], df['selling_price'], test_size=0.2, random_state=2
)

vif = pd.DataFrame()
vif['Features'] = X_tr_vif.columns
vif['VIF'] = [variance_inflation_factor(X_tr_vif.values, i) for i in range(X_tr_vif.shape[1])]
vif = vif.sort_values(by="VIF", ascending=False).reset_index(drop=True)
vif

# %% [markdown]
# `year` (and its twin `age`) come back with an enormous VIF — the warnings printed above ("rank-deficient",
# "poorly conditioned") *are* the multicollinearity: statsmodels is telling us the exact same thing the VIF numbers
# are, from a different angle. That perfect $\text{corr}(\text{year}, \text{age}) = -1$ is the textbook case this
# whole diagnostic exists to catch.
#
# ### 3.2 Iterative elimination
#
# Remove the worst offender, recompute VIF on what's left, repeat — until every remaining feature is under the
# threshold, or removing more would cost too much Adjusted R².
#

# %%
cols = list(X_tr_vif.columns)
feats_removed = []
vif_threshold = 5
adj_r2_threshold = 0.85

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while True:
        X_current = X_tr_vif[cols]
        vifs = [variance_inflation_factor(X_current.values, i) for i in range(X_current.shape[1])]
        vif_now = pd.DataFrame({'Features': cols, 'VIF': vifs}).sort_values('VIF', ascending=False).reset_index(drop=True)

        X_sm_current = sm.add_constant(X_current)
        current_fit = sm.OLS(y_tr_vif.values, X_sm_current).fit()

        if vif_now.iloc[0]['VIF'] < vif_threshold or current_fit.rsquared_adj < adj_r2_threshold:
            break

        feats_removed.append(vif_now.iloc[0]['Features'])
        cols = [c for c in cols if c != vif_now.iloc[0]['Features']]

print("Features removed:", feats_removed)
print("Final Adjusted R²:", current_fit.rsquared_adj)
vif_now

# %% [markdown]
# ✅ We went from 17 features down to a smaller, decorrelated set with every VIF comfortably under 5, and barely
# gave up any Adjusted R² in the process — the redundant features weren't adding real predictive power, just
# noise in the coefficients.
#

# %% [markdown]
# ## 4. Residual diagnostics — normality, independence, homoscedasticity
#
# The remaining three assumptions are all checked by looking at the model's *residuals* (errors), not its
# predictions directly.
#
# ### 4.1 Assumption 3 — errors should be normally distributed
#
# Well-specified residuals should look like a bell curve centered at zero. If they don't, it's often a sign of
# outliers, a skewed target, or a genuinely misspecified (e.g. non-linear) model.
#

# %%
X_sm_full = sm.add_constant(X_tr_vif)
sm_model_full = sm.OLS(y_tr_vif.values, X_sm_full).fit()

y_hat = sm_model_full.predict(X_sm_full)
errors = y_hat - y_tr_vif.values

plt.figure(figsize=(7, 4))
plt.hist(errors, bins=50, color="#2563EB", alpha=0.8)
plt.xlabel("Residual"); plt.title("Histogram of residuals")
plt.show()

# %%
shapiro_sample = errors if len(errors) <= 5000 else np.random.default_rng(0).choice(errors, 5000, replace=False)
shapiro_result = stats.shapiro(shapiro_sample)
print("Shapiro-Wilk statistic:", shapiro_result.statistic)

# %% [markdown]
# | Statistic | Interpretation |
# |---|---|
# | close to 1.0 | strong normality |
# | 0.85 – 0.95 | moderate normality — acceptable on large datasets |
# | below 0.8 | significant departure — investigate outliers |
#
# ### 4.2 Assumption 4 — errors should be independent
#
# A scatter of residuals vs. the target should show no visible pattern — just random scatter around zero. This is
# mostly a time-series concern (errors at time $t$ correlating with errors at $t-1$, called autocorrelation); for
# a cross-sectional dataset like ours it's rarely an issue.
#
# The **Durbin-Watson statistic**, printed in every StatsModels summary, tests exactly this — values near 2.0 mean
# no autocorrelation; values near 0 or 4 signal a problem.
#

# %%
print("Durbin-Watson:", sm.stats.stattools.durbin_watson(sm_model_full.resid))

# %% [markdown]
# ### 4.3 Assumption 5 — homoscedasticity (constant error variance)
#
# The residuals' spread should stay roughly constant across all predicted values. Its opposite,
# **heteroscedasticity**, shows up as a cone shape — errors fanning out (or in) as predictions grow.
#

# %%
plt.figure(figsize=(7, 4.5))
plt.scatter(y_hat, errors, s=8, alpha=0.4, color="#2563EB")
plt.axhline(0, color="#374151", lw=1)
plt.xlabel("Predicted selling price"); plt.ylabel("Residual")
plt.title("Predicted values vs. residuals")
plt.show()

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gq_stat, gq_pvalue, _ = sms.het_goldfeldquandt(y_tr_vif, X_sm_full)
print(f"Goldfeld-Quandt F statistic: {gq_stat:.3f}")
print(f"Goldfeld-Quandt p-value:     {gq_pvalue:.3f}")

# %% [markdown]
# The Goldfeld-Quandt test formally compares residual variance in the lower half of the data vs. the upper half.
# An F-statistic near 1 with a p-value above 0.05 means we fail to reject the null of equal variance —
# homoscedasticity holds.
#
# > **If you do find heteroscedasticity:** transform the target (`log(y)`, `sqrt(y)`, or Box-Cox), remove outliers
# > that are inflating variance in one region, or move to weighted least squares regression.
#

# %% [markdown]
# ## 5. Summary — revision cheat sheet
#
# This closes out the 3-part Linear Regression series. Full recap:
#
# **The model:** $y = m_1x_1 + m_2x_2 + \cdots + c$ — a straight line (one feature) or a hyperplane (two or more
# features) fit through the data. Every ML algorithm reduces to the same three ingredients: **Model** (predicts),
# **Cost Function** (scores the error), **Optimizer** (reduces it).
#
# **Fitting it — the loop:** start with random $m$, $c$ → predict → measure error with MSE → nudge $m$, $c$ downhill
# via gradient descent → repeat until the error stops meaningfully improving.
#
# **Core definitions:**
# - *MSE (the cost function)* — $\frac{1}{n}\sum(y - y_{predicted})^2$; convex, so it has a single global minimum
#   and gradient descent always finds it.
# - *Gradient descent (the optimizer)* — repeatedly step each parameter opposite its gradient:
#   $x = x - \text{learning_rate} \cdot \frac{d(Error)}{dx}$.
# - *R² (coefficient of determination)* — $1 - RSS/TSS$; how much better your model does than just predicting the
#   mean every time. Range $(-\infty, 1]$; near 1 is a great fit, near 0 is no better than the mean, negative is
#   worse than the mean.
# - *Adjusted R²* — $R^2$ penalized for feature count; the fair way to compare models with different numbers of
#   features, since plain $R^2$ never decreases from adding *any* feature, useful or not.
#
# **Preprocessing this series assumed as a given** (see the dedicated
# [Data Preprocessing](../../../data-preprocessing/eda/notes.ipynb) notebooks for the general techniques):
# split into train/test before fitting, one-hot encode categorical features, impute missing values, scale features,
# detect and treat outliers.
#
# **Trusting the model — the 5 assumptions:**
# 1. Linearity — fix with [Polynomial Regression](../../02-polynomial-regression/notes.ipynb) if violated.
# 2. No multicollinearity — detect with VIF ($1/(1-R_j^2)$), fix by removing high-VIF features.
# 3. Normal residuals — check with a histogram + Shapiro-Wilk.
# 4. Independent errors — check with Durbin-Watson (≈2.0 is healthy).
# 5. Homoscedasticity — check visually (residuals vs. predicted) and with the Goldfeld-Quandt test.
#
# **Next up:** [Polynomial Regression](../../02-polynomial-regression/notes.ipynb) — what to do when Assumption 1
# is violated and the data genuinely isn't a straight line, plus the Bias-Variance Tradeoff that comes with it.
# Beyond that: Regularization (Ridge / Lasso — penalizing large weights directly) and Cross-Validation (a more
# robust way to pick model complexity than a single train/test split).
#
