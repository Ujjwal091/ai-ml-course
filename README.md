# AI/ML Course Notes

My running notes as I learn machine learning: theory, diagrams, and runnable code, organized by module.

Live site: https://ujjwal091.github.io/ai-ml-course/ (once GitHub Pages is enabled — see below)

## How this is organized

```
modules/
  00-introduction-to-ml/
    notes.ipynb        <- stands alone, not part of any module
  01-supervised-learning/
    01-linear-regression/
      notes.ipynb       <- theory + code + visualizations for this topic
  02-unsupervised-learning/
    k-means/
      notes.ipynb
    k-means-plus-plus/
      notes.ipynb
```

Each topic gets its own folder with a single notebook (`notes.ipynb`) that mixes:
- **Theory** — markdown cells: definitions, intuition, math, diagrams/images, links to class notes or articles I used.
- **Code** — runnable Python cells, usually both a from-scratch implementation (to build intuition) and the `scikit-learn` version (for practical use).
- **Visualizations** — plots generated inline, since a lot of ML concepts (especially unsupervised learning) click faster when you can see them.

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

## Building the site locally

```bash
source .venv/bin/activate   # if not already active
jupyter-book build .
open _build/html/index.html
```

Pushing to `main` also triggers a GitHub Actions workflow that builds and deploys the site to GitHub Pages automatically (see `.github/workflows/deploy.yml`).

## Modules

- [x] Introduction to Machine Learning *(stands alone — not part of a module)*
- [ ] 01 — Supervised Learning
  - [x] Linear Regression
  - [ ] Polynomial Regression, Bias/Variance, Regularization
  - [ ] Cross-Validation
  - [ ] Logistic Regression / Classification
- [ ] 02 — Unsupervised Learning
  - [x] K-Means Clustering
  - [x] K-Means++
  - [ ] Hierarchical Clustering
  - [ ] DBSCAN
  - [ ] PCA

## Adding a new topic

1. Copy `modules/_template/` into the right module folder, rename to the topic.
2. Fill in theory, paste your class-note links/summaries, write code, add plots.
3. Update the checklist above and commit.
