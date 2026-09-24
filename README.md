# AI/ML Course Notes

My running notes as I learn machine learning: theory, diagrams, and runnable code, organized by module.

Live site: https://ujjwal091.github.io/ai-ml-course/ (once GitHub Pages is enabled — see below)

## How this is organized

```
modules/
  00-introduction-to-ml/
    notes.py            <- the real source — edit THIS one
    notes.ipynb          <- auto-generated from notes.py, don't hand-edit
  data-preprocessing/
    eda/
      notes.py
      notes.ipynb
    feature-scaling/
      notes.py
      notes.ipynb
    encoding/
      notes.py
      notes.ipynb
  01-supervised-learning/
    01-linear-regression/
      01-basics/
        notes.py
        notes.ipynb
      02-multivariate-regression/
        notes.py
        notes.ipynb
      03-assumptions-and-diagnostics/
        notes.py
        notes.ipynb
    02-polynomial-regression/
      notes.py
      notes.ipynb
  02-unsupervised-learning/
    k-means/
      notes.py
      notes.ipynb
    k-means-plus-plus/
      notes.py
      notes.ipynb
    hierarchical-clustering/
      notes.py
      notes.ipynb
    gaussian-mixture-models/
      notes.py
      notes.ipynb
    dbscan/
      notes.py
      notes.ipynb
    time-series/
      notes.py
      notes.ipynb
    time-series-forecasting/
      notes.py
      notes.ipynb
  03-case-studies/
    iris-clustering/
      notes.py
      notes.ipynb
```

Each topic gets its own folder with **one file you actually write in: `notes.py`** (plain Python, in Jupyter's
"percent" format — `# %%` marks a code cell, `# %% [markdown]` marks a markdown cell). It reads like a normal
script, diffs cleanly in git, and mixes:
- **Theory** — markdown cells: definitions, intuition, math, diagrams/images, links to articles I used.
- **Code** — runnable Python cells, usually both a from-scratch implementation (to build intuition) and the `scikit-learn` version (for practical use).
- **Visualizations** — plots generated inline; cells that exist *only* to draw a picture (not to teach the algorithm) get `# %% tags=["remove-input"]` so the site shows the picture without the plotting code.

`notes.ipynb` sits alongside it — that's a generated file, kept in sync automatically by
[jupytext](https://jupytext.readthedocs.io/), and is what actually gets executed, opened in Colab, and built into
the site (it's the only place outputs/plots are stored, since the plain-text `.py` doesn't carry cell outputs).
You never hand-edit the `.ipynb` — see the workflow below.

## How to run/edit the code

**Option A — Google Colab (no setup, recommended for quick edits):**
Every notebook has an "Open in Colab" badge at the top. Click it, and you get a live, editable, runnable copy in your browser — changes don't affect the repo unless you save a copy back to GitHub.

**Option B — Locally with Jupyter:**
```bash
git clone https://github.com/Ujjwal091/ai-ml-course.git
cd ai-ml-course
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Editing workflow (the actual day-to-day loop)

1. Open `notes.py` for the topic in any text editor and write — headings, prose, code, whatever. No notebook UI needed.
2. Sync it into the notebook and bake outputs:
   ```bash
   jupytext --sync modules/<topic>/notes.ipynb   # pulls your notes.py edits into notes.ipynb
   jupyter execute --inplace modules/<topic>/notes.ipynb   # actually runs it, saves outputs
   ```
3. Preview the site (see below), then commit **both** `notes.py` and `notes.ipynb` — jupytext keeps them paired via
   metadata in the `.ipynb`, so `--sync` always knows what to do.

(If you'd rather work inside Jupyter/JupyterLab directly — running cells, seeing plots as you go — that works too:
open `notes.ipynb` there, and as long as the [jupytext extension](https://jupytext.readthedocs.io/en/latest/install.html)
is installed, saving the notebook automatically updates `notes.py` for you.)

## Building the site locally

```bash
source .venv/bin/activate   # if not already active
jupyter-book build .
open _build/html/index.html
```

Pushing to `main` also triggers a GitHub Actions workflow that builds and deploys the site to GitHub Pages automatically (see `.github/workflows/deploy.yml`).

## Modules

- [x] Introduction to Machine Learning *(stands alone — not part of a module)*
- [x] Data Preprocessing *(general-purpose, reused across modules)*
  - [x] Exploratory Data Analysis (EDA)
  - [x] Feature Scaling
  - [x] Encoding Categorical Features
- [ ] 01 — Supervised Learning
  - [x] Linear Regression *(basics, multivariate regression & evaluation, the 5 assumptions + diagnostics)*
  - [x] Polynomial Regression *(incl. the bias-variance tradeoff)*
  - [ ] Regularization (Ridge / Lasso)
  - [ ] Cross-Validation
  - [ ] Logistic Regression / Classification
- [ ] 02 — Unsupervised Learning
  - [x] K-Means Clustering
  - [x] K-Means++
  - [x] Hierarchical Clustering
  - [x] Gaussian Mixture Models *(incl. covariance types — spherical/diagonal/full)*
  - [x] DBSCAN
  - [ ] PCA
  - [x] Time Series — Cleaning, Trend & Seasonality
  - [x] Time Series — Forecasting *(baselines, exponential smoothing, stationarity; ARIMA/SARIMA and UMAP to follow)*
- [x] Case Studies
  - [x] Iris Clustering with K-Means

## Adding a new topic

1. Copy `modules/_template/` (has a starter `notes.py` + paired `notes.ipynb`) into the right module folder, rename to the topic.
2. Edit `notes.py` — fill in theory, paste reference links/summaries, write code.
3. Run the sync + execute steps above to generate outputs, add the page to `_toc.yml`.
4. Update the checklist above and commit both files.
