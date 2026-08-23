import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ================= Header =================
md("""# Introduction to Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ujjwal091/ai-ml-course/blob/main/modules/00-introduction-to-ml/notes.ipynb)

*First topic of the course — what ML is, how it differs from regular programming, and the map of problem types
you'll be solving from here on.*
""")

# ================= 1. ML vs classical programming =================
md("""## 1. How is ML different from just... programming?

One line to remember this by:

- **Classical programming:** `input + program = output` — you hand-write the rules.
- **Machine learning:** `input + output = program` — you show examples, it works out the rules.
""")

md("""```{mermaid}
graph LR
    subgraph Classical["Classical programming"]
        A1[Input data] --> A2[Hand-written rules] --> A3[Output]
    end
    subgraph ML["Machine learning"]
        B1[Data + labels] --> B2[Learning algorithm] --> B3[Model] --> B4[Prediction]
    end

    style Classical fill:#FDE68A,stroke:#374151,stroke-width:2px,color:#111827
    style ML fill:#A7F3D0,stroke:#374151,stroke-width:2px,color:#111827
    style A1 fill:#FFFBEB,stroke:#374151,color:#111827
    style A2 fill:#FFFBEB,stroke:#374151,color:#111827
    style A3 fill:#FFFBEB,stroke:#374151,color:#111827
    style B1 fill:#ECFDF5,stroke:#374151,color:#111827
    style B2 fill:#ECFDF5,stroke:#374151,color:#111827
    style B3 fill:#ECFDF5,stroke:#374151,color:#111827
    style B4 fill:#ECFDF5,stroke:#374151,color:#111827
```""")

md("""**Example — spam detection:**

- Classical approach: notice suspicious keywords, write `if / elif / else` rules to catch them.
- Problem: a spammer swaps "lottery" for "lucky draw" and the rule silently breaks. Every new disguise needs its
  own new rule — brittle by construction.
- ML approach: hand the model thousands of emails already labeled spam / not-spam, let it find the pattern itself.

**When to actually reach for ML** instead of writing rules:
- The pattern is too complex to hand-write.
- You need the system to generalize to new cases it hasn't seen, not just the ones you enumerated.
""")

md("""**Quick check**

> Why does a keyword-based spam filter eventually break down, even if it starts out working well?
>
> It only catches what it was explicitly told to catch — every new phrasing needs a new rule, so the list always
> lags one step behind. A model trained on labeled examples learns the *underlying* pattern instead, so it can
> catch variations it was never explicitly shown.
""")

# ================= 2. The ML pipeline =================
md("""## 2. What actually happens when you "build a model"

Writing the learning algorithm is a small fraction of the work. The real pipeline, grouped into three phases:
""")

md("""```{mermaid}
graph LR
    A["<b>Understand</b><br/>define problem,<br/>collect, explore"] --> B["<b>Prepare</b><br/>clean, EDA,<br/>train/test split"] --> C["<b>Model</b><br/>train, evaluate,<br/>deploy"]

    style A fill:#BFDBFE,stroke:#374151,stroke-width:2px,color:#111827
    style B fill:#FDE68A,stroke:#374151,stroke-width:2px,color:#111827
    style C fill:#A7F3D0,stroke:#374151,stroke-width:2px,color:#111827
```""")

md("""**All nine steps, in order:**

1. Define the problem — state the goal in plain, non-ML terms first (fraud detection looks different in banking
   vs. healthcare).
2. Collect data relevant to that goal.
3. Understand the data before touching it.
4. Clean and prepare it — **this is where most of the time actually goes.**
5. Explore it (EDA) — stats and plots (`pandas`, `matplotlib`) to spot relationships and trends.
6. Split it into a training set and a held-out test set.
7. Train the model on the training data.
8. Evaluate it — specifically on the test set, never the training set.
9. Deploy it — as an API or app. A model stuck in a notebook helps nobody.

**Two rules people actually get burned by:**
- **Garbage in, garbage out** — bad data beats even the best algorithm. "Better data beats a fancier algorithm."
- **Never train on test data** — even accidentally. The model would just memorize the answers instead of learning
  to generalize, and every evaluation number afterward would be a lie.

**Data cleaning, concretely:** remove duplicate rows, handle missing values, fix structural inconsistencies (e.g.
"ML Eng." and "Machine Learning Engineer" being the same title typed two ways), catch outlier data-entry errors
(a salary with an extra zero tacked on).

**Kinds of data you'll run into:**
- **Categorical** — car color, fuel type
- **Numerical** — height, weight, price
- **Time series** — stock prices, daily temperatures (values at evenly-spaced time points)
- **Text** — reviews, chat messages, emails
""")

# ================= 3. Types of learning =================
md("## 3. The three ways a model can learn")
md("""```{mermaid}
graph TD
    T["Types of learning"] --> S["Supervised"]
    T --> U["Unsupervised"]
    T --> R["Reinforcement"]

    S --> S1["Labeled data"]
    S --> S2["Learns input → output"]
    S --> S3["e.g. spam / not-spam"]

    U --> U1["No labels at all"]
    U --> U2["Finds structure itself"]
    U --> U3["e.g. group customers by behavior"]

    R --> R1["Agent + environment"]
    R --> R2["Reward / penalty feedback"]
    R --> R3["e.g. a chess bot"]

    style T fill:#F3F4F6,stroke:#374151,stroke-width:2px,color:#111827
    style S fill:#BFDBFE,stroke:#374151,stroke-width:2px,color:#111827
    style U fill:#FDE68A,stroke:#374151,stroke-width:2px,color:#111827
    style R fill:#A7F3D0,stroke:#374151,stroke-width:2px,color:#111827
    style S1 fill:#EFF6FF,stroke:#374151,color:#111827
    style S2 fill:#EFF6FF,stroke:#374151,color:#111827
    style S3 fill:#EFF6FF,stroke:#374151,color:#111827
    style U1 fill:#FFFBEB,stroke:#374151,color:#111827
    style U2 fill:#FFFBEB,stroke:#374151,color:#111827
    style U3 fill:#FFFBEB,stroke:#374151,color:#111827
    style R1 fill:#ECFDF5,stroke:#374151,color:#111827
    style R2 fill:#ECFDF5,stroke:#374151,color:#111827
    style R3 fill:#ECFDF5,stroke:#374151,color:#111827
```""")

md("""- **Supervised** — labeled data, learn the input → output mapping. Covers **regression** and
  **classification**.
- **Unsupervised** — unlabeled data, find structure on your own. Covers **clustering** (see the K-Means notes).
- **Reinforcement** — an *agent* acts in an *environment*, gets a reward or penalty, learns which behaviors to
  repeat. Like training a dog: fetch the stick → treat; chew the slipper → scolding. A chess bot works the same
  way — agent = bot, environment = board, some evaluator hands back a `+`/`-` signal per move. Enough repetition of
  that loop is how you get things like AlphaGo, or a bot that beats humans at DOTA/CS:GO.
""")

md("""**Quick check**

> A litter of ducklings, a week old, keep falling over trying to walk — and after enough attempts they're walking
> and swimming confidently. Which type of learning is this?
>
> Reinforcement learning — no labeled "correct walking pattern" is handed to them, and they're not just finding
> passive structure sitting still; they're improving through repeated trial, error, and feedback from falling over.
""")

# ================= 4. Task types =================
md("""## 4. The six problem shapes you'll actually be solving

Once you know *how* a model learns, the next question is *what kind of answer* you want out of it:

| Task | Predicts | Example |
|---|---|---|
| **Regression** | A continuous number | House selling price |
| **Classification** | One of several categories | Cat vs. dog; spam vs. not-spam |
| **Clustering** | Groups, no labels given | Segmenting customers by behavior |
| **Recommendation** | Items you'd likely want | YouTube's "Up Next" panel |
| **Time series forecasting** | A future value from past values | Tomorrow's stock price |
| **Reinforcement learning** | The next best action | A game bot's next move |

**Regression, one beat longer** — target `y` ranges $-\\infty$ to $+\\infty$. Given features like area, rooms,
location → predict a price. Whatever factors *you'd* personally weigh to guess a house's price become the model's
*features*.

**Classification, one beat longer** — target `y` is a discrete category, not a number. Binary (cat=0/dog=1) or
multiclass (cat/dog/bird/fish). Under the hood it's about finding a **decision boundary** separating the classes.
Shortcut to remember the split: regression predicts *how much*, classification predicts *which one*.

**Recommendation, one beat longer** — watch history has `v10`, `v12`, `v16` → suggest `v15` (similar to `v12`) and
`v36` (similar to `v10`). Built on collaborative filtering, item-similarity, or content-based similarity.

**Time series, one beat longer** — data at evenly-spaced time points (`y` = value, `t` = time index) → predict `y`
at a future `t` from patterns in the past values.
""")

md("""**Try these yourself before checking the answers** — which task type fits each scenario?

1. Thousands of identical items in inventory — predict how many will sell in the next three months.
2. Industrial cameras inspect currency prints to flag genuine vs. counterfeit banknotes.
3. YouTube shows "Videos you may like" based on your watch history.

**Answers**

> 1. **Regression** — predicting a quantity, a continuous number.
> 2. **Classification** — genuine vs. counterfeit are discrete categories.
> 3. **Recommendation** — suggesting items based on past behavior and similar items/users.
""")

# ================= 5. Summary =================
md("""## 5. Summary — revision cheat sheet

- **The core flip:** classical = `input + program = output`; ML = `input + output = program`. Reach for ML when
  the pattern's too complex to hand-write, or you need generalization beyond what you showed it.
- **The pipeline:** define → collect → understand → clean (most of the time) → EDA → split → train → evaluate →
  deploy. Garbage in, garbage out. Never train on test data.
- **Three ways to learn:** supervised (labeled, → regression/classification), unsupervised (unlabeled, →
  clustering), reinforcement (reward/penalty feedback loop).
- **Six task shapes:** regression, classification, clustering, recommendation, time series forecasting,
  reinforcement learning.
- **Data types:** categorical, numerical, time series, text.

**Next up:** Linear Regression — the first concrete algorithm, using a real Cars24 pricing dataset.
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}
}

with open("modules/00-introduction-to-ml/notes.ipynb", "w") as f:
    nbf.write(nb, f)

print("done, cells:", len(cells))
