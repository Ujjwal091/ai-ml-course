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
# # Encoding Categorical Features
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ujjwal091/ai-ml-course/blob/main/modules/data-preprocessing/encoding/notes.ipynb)
#
# *A general-purpose preprocessing step: turning text categories into numbers a model can actually use. There are
# three encodings worth knowing well — Label, One-Hot, and Target — and picking the wrong one for the situation is a
# real, common source of bugs, not a stylistic choice.*
#

# %% [markdown]
# ## 1. Why categorical features need encoding
#
# Most ML algorithms are built on arithmetic — dot products, distances, gradients — none of which are defined on
# the string `"europe"`. Before a categorical column can enter a model, it has to become numbers, and *how* it
# becomes numbers determines what the model is able to learn from it.
#

# %%
import os
import numpy as np
import pandas as pd

MPG_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
MPG_PATH = "data/auto-mpg.csv"

if not os.path.exists(MPG_PATH):
    os.makedirs("data", exist_ok=True)
    pd.read_csv(MPG_URL).to_csv(MPG_PATH, index=False)

df = pd.read_csv(MPG_PATH)
df['origin'].value_counts()

# %% [markdown]
# ## 2. The three encodings, at a glance
#
# | | Label / Ordinal | One-Hot | Target / Mean |
# |---|---|---|---|
# | **Idea** | One integer per category | One binary column per category | Replace category with the mean target value for that category |
# | **Columns added** | 1 | (number of categories − 1, after dropping one) | 1 |
# | **Assumes an order?** | Yes — imposes one | No | No |
# | **Needs the target column?** | No | No | Yes — it's the only encoding that looks at $y$ |
# | **Best for** | Genuinely ordered categories | Low/medium-cardinality unordered categories | High-cardinality unordered categories |
# | **Main risk** | Inventing a fake order | Column count explodes at high cardinality | Target leakage if done carelessly |
#
# The rest of this notebook takes each one in turn: how to use it, its real advantages, its real limitations, and
# when it's actually the right call — not just the default.
#

# %% [markdown]
# ## 3. Label (Ordinal) Encoding
#
# **What it does:** assigns each category a single integer — `usa → 0`, `europe → 1`, `japan → 2`.
#
# **How to use it:**
#

# %%
from sklearn.preprocessing import OrdinalEncoder

# manual version, for full control over which integer means what
label_map = {'usa': 0, 'europe': 1, 'japan': 2}
df['origin_label'] = df['origin'].map(label_map)

# equivalent, via sklearn — useful once this becomes one step in a larger Pipeline
ord_enc = OrdinalEncoder(categories=[['usa', 'europe', 'japan']])
df['origin_label_sklearn'] = ord_enc.fit_transform(df[['origin']])

df[['origin', 'origin_label', 'origin_label_sklearn']].drop_duplicates()

# %% [markdown]
# **Where label encoding genuinely works well** — a column that has a real, meaningful order:
#

# %%
size_df = pd.DataFrame({'size': ['small', 'large', 'medium', 'small', 'large']})
size_order = [['small', 'medium', 'large']]  # the order IS the information

size_ord_enc = OrdinalEncoder(categories=size_order)
size_df['size_encoded'] = size_ord_enc.fit_transform(size_df[['size']])
size_df

# %% [markdown]
# `small=0 < medium=1 < large=2` matches reality — a model can now legitimately use "greater than" comparisons and
# arithmetic on this column, and that arithmetic means something.
#
# **Why it's wrong for `origin`:** nothing about "japan" makes it "twice" anything relative to "usa". A model fed
# `origin_label` would silently learn that `japan` (2) has twice the numeric weight of `europe` (1) in any linear
# term — an ordering the data never actually implied, purely because integers happen to have a natural order.
#
# | | Advantage | Limitation |
# |---|---|---|
# | Label Encoding | 1 column, no dimensionality blow-up; fast, memory-light; the natural (and correct) choice for genuinely ordinal data | Invents a false order/magnitude on unordered categories — actively misleads linear models, distance-based models (KNN, k-means), and anything that treats the number as a magnitude |
#
# > **When to use it:** the categories have a true, known order (`low/medium/high`, `bronze/silver/gold`, education
# > level, star ratings). Tree-based models (decision trees, random forests, gradient boosting) are also more
# > forgiving of label encoding even on unordered categories, since a tree just picks split thresholds — it doesn't
# > assume the numeric gaps between categories mean anything. It's still not the *right* semantic choice there, but
# > it's less actively harmful than in a linear model.
#

# %% [markdown]
# ## 4. One-Hot Encoding
#
# **What it does:** instead of one column with an implied order, create one binary column *per category*. Each row
# gets a 1 in exactly one of them:
#
# | origin | origin_usa | origin_europe | origin_japan |
# |---|---|---|---|
# | usa | 1 | 0 | 0 |
# | europe | 0 | 1 | 0 |
# | japan | 0 | 0 | 1 |
#
# No column implies more or less than any other — the model treats all three as independent, equal-footing
# signals, with no fabricated order or magnitude.
#
# **How to use it:**
#

# %%
one_hot = pd.get_dummies(df[['origin']], columns=['origin'], dtype=int)
one_hot.drop_duplicates()

# %% [markdown]
# ### 4.1 The dummy variable trap
#
# With 3 categories, knowing any 2 of the 3 one-hot columns tells you the 3rd for free — if a row isn't `usa` and
# isn't `europe`, it must be `japan`. That means one column carries no additional information: it's a perfect
# linear combination of the other two ($\text{origin\_japan} = 1 - \text{origin\_usa} - \text{origin\_europe}$),
# which is exactly the kind of multicollinearity that destabilizes a regression model's coefficients (see
# [Assumptions & Diagnostics](../../01-supervised-learning/01-linear-regression/03-assumptions-and-diagnostics/notes.ipynb)).
#
# **The fix:** always drop one category's column — it becomes the implicit "baseline" that every other category is
# compared against.
#

# %%
one_hot_safe = pd.get_dummies(df[['origin']], columns=['origin'], drop_first=True, dtype=int)
one_hot_safe.drop_duplicates()

# %% [markdown]
# `origin_europe` is now the baseline: a row of all zeros in the remaining columns means "europe", and every
# coefficient the model later learns for `origin_usa` or `origin_japan` is interpreted *relative to europe*.
#
# | | Advantage | Limitation |
# |---|---|---|
# | One-Hot Encoding | No fabricated order; every category treated as an independent, equal signal; the safe general-purpose default | Column count grows with the number of categories — a feature with 500 distinct values becomes 499 columns, most of them almost entirely zero (sparse); must always drop one column to avoid the dummy variable trap |
#
# > **When to use it:** unordered categories with a manageable cardinality — a rough rule of thumb is under a few
# > dozen distinct values. Beyond that, one-hot starts hurting more than it helps: the feature space balloons, most
# > columns become sparse and uninformative, and models with many parameters (like linear regression) start
# > overfitting to rare categories that only appear a handful of times.
#

# %% [markdown]
# ## 5. Target (Mean) Encoding
#
# **What it does:** replace each category with the *average target value* for rows in that category. Unlike the
# other two, this encoding actually looks at $y$ — which is exactly where its power and its danger both come from.
#
# **The dataset problem it's built for:** `origin` only has 3 categories, so it doesn't need target encoding — the
# whole point of this technique is high-cardinality columns (hundreds or thousands of distinct values) where
# one-hot would explode. To make that concrete, here's a synthetic example: 60 online stores, uneven traffic, and a
# `converted` outcome to predict.
#

# %%
rng = np.random.default_rng(11)
n_stores = 60
true_conversion_rate = rng.beta(2, 8, n_stores)  # each store has its own real (unknown) conversion rate

store_ids = rng.integers(0, n_stores, 4000)
visits = pd.DataFrame({'store_id': [f"store_{i}" for i in store_ids]})
visits['converted'] = rng.binomial(1, true_conversion_rate[store_ids])

visits['store_id'].value_counts().describe()[['min', '50%', 'max']]

# %% [markdown]
# 60 distinct stores — one-hot would add 59 columns for this alone, most rarely-visited stores contributing almost
# nothing per column. Target encoding instead compresses each store down to a single, meaningful number: how well
# it actually converts.
#
# ### 5.1 The naive version — and why it leaks
#

# %%
naive_means = visits.groupby('store_id')['converted'].transform('mean')
visits['naive_target_enc'] = naive_means

visits[['store_id', 'converted', 'naive_target_enc']].head()

# %% [markdown]
# The bug: each row's encoded value **includes that row's own target** in the average used to encode it. For a
# store with only 3 visits, one row's own outcome is a third of its own encoded feature — the model can partially
# "see the answer" baked into the input. Evaluated on the training set, this looks like an excellent feature.
# Evaluated on genuinely new data, that leaked signal isn't there anymore, and performance quietly drops.
#
# ### 5.2 Doing it correctly — out-of-fold encoding
#
# **The fix:** each row must be encoded using target statistics computed from *other* rows — never its own. The
# standard approach is K-Fold target encoding: split the training data into folds, and encode each fold using the
# mean computed from all the *other* folds.
#

# %%
from sklearn.model_selection import KFold, train_test_split

train_visits, test_visits = train_test_split(visits, test_size=0.25, random_state=0)

global_mean = train_visits['converted'].mean()  # fallback for categories a fold never sees
oof_encoded = pd.Series(index=train_visits.index, dtype=float)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for fit_idx, hold_idx in kf.split(train_visits):
    fit_fold = train_visits.iloc[fit_idx]
    hold_fold = train_visits.iloc[hold_idx]

    fold_means = fit_fold.groupby('store_id')['converted'].mean()
    oof_encoded.iloc[hold_idx] = hold_fold['store_id'].map(fold_means).fillna(global_mean).values

train_visits = train_visits.assign(store_target_enc=oof_encoded)

# for the test set (and any future new data), use statistics from the FULL training set — never the test target
full_train_means = train_visits.groupby('store_id')['converted'].mean()
test_visits = test_visits.assign(
    store_target_enc=test_visits['store_id'].map(full_train_means).fillna(global_mean)
)

train_visits[['store_id', 'converted', 'store_target_enc']].head()

# %% [markdown]
# ### 5.3 Smoothing — handling low-count categories
#
# A store with only 2 visits, both of which converted, gets a raw mean of 1.0 — a confident-looking number built
# from almost no evidence. **Smoothing** pulls low-count categories toward the global average, proportionally to
# how little data they have:
#
# $$
# \text{encoded} = \frac{n_{\text{category}} \cdot \bar{y}_{\text{category}} + m \cdot \bar{y}_{\text{global}}}{n_{\text{category}} + m}
# $$
#
# where $m$ is a smoothing strength (higher = trust the category's own mean less until it has more data).
#

# %%
def smoothed_means(frame, group_col, target_col, m=10):
    global_avg = frame[target_col].mean()
    agg = frame.groupby(group_col)[target_col].agg(['mean', 'count'])
    return (agg['count'] * agg['mean'] + m * global_avg) / (agg['count'] + m)


raw_vs_smoothed = pd.DataFrame({
    'raw_mean': train_visits.groupby('store_id')['converted'].mean(),
    'count': train_visits.groupby('store_id')['converted'].count(),
    'smoothed_mean': smoothed_means(train_visits, 'store_id', 'converted', m=10),
})
raw_vs_smoothed.sort_values('count').head()

# %% [markdown]
# Notice the smallest-count stores get pulled hardest toward the global average — exactly the low-evidence cases
# where the raw mean was least trustworthy.
#
# | | Advantage | Limitation |
# |---|---|---|
# | Target Encoding | Compresses any number of categories into a single, informative column — no dimensionality blow-up, even at thousands of categories; captures how predictive each category actually is | Leaks the target if computed naively (must use out-of-fold / train-only statistics); needs smoothing for low-count categories; an extra moving part (fold logic) compared to one-hot; only usable when a target column exists, so not for unsupervised pipelines |
#
# > **When to use it:** high-cardinality unordered categoricals (zip codes, product IDs, user IDs, store IDs) where
# > one-hot would create an impractical number of columns. Skip it for small category counts — one-hot is simpler
# > and has no leakage risk to manage.
#

# %% [markdown]
# ## 6. Summary — which one, and when
#
# | Encoding | Use when | Watch out for |
# |---|---|---|
# | Label / Ordinal | Categories have a true order | Never use on unordered categories — it invents a false magnitude |
# | One-Hot | Unordered categories, low-to-medium cardinality | Column count explodes at high cardinality; always drop one column |
# | Target / Mean | Unordered categories, high cardinality | Leaks target information unless computed out-of-fold; needs smoothing for rare categories |
#
# **The decision in practice:** check for a genuine order first (label encoding); if there isn't one, check the
# category count (one-hot for a manageable number, target encoding once one-hot would explode); if using target
# encoding, always compute it out-of-fold and smooth low-count categories.
#
# **Where this leads next:** encoded and scaled data (see [Feature Scaling](../feature-scaling/notes.ipynb)) is
# what most model-fitting notebooks in this repo assume as their starting point.
#
