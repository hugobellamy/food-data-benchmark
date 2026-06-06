# Food Nutrition Benchmarks

Benchmarking LLMs on estimating calories and macros (protein, fat, carbs) from food descriptions — without internet access. This is for diet tools like [this](https://github.com/hugobellamy/meal-plan-skill/).

## Process

1. **Build datasets** (`01_build_datasets.ipynb`) — sample foods from a nutrition database, optionally use an LLM to generate natural human-like descriptions (e.g. "a chicken breast" instead of "150g of Chicken breast, grilled"). Ground truth macros are computed from the DB values scaled by weight.
2. **Run benchmarks** (`02_benchmark.ipynb`) — send each food prompt to multiple LLMs via OpenRouter, ask them to estimate calories/protein/fat/carbs. Results saved per model.
3. **Analyse** (`03_analysis.ipynb`) — compare models using R², MAE, and MAPE. Results split by "soft" (natural language) vs "raw" (exact gram weights) prompts.

## Data source

[McCance and Widdowson's Composition of Foods Integrated Dataset 2021](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid) — UK government nutrition database with ~2900 foods, values per 100g.

## Results

R² scores (higher is better). "Soft" = natural language prompts ("an apple"), "Raw" = exact weights ("180g of Apples, eating, raw").

### All

| Model                  |   Calories |   Protein |    Fat |   Carbs |   Mean R² |
|:-----------------------|-----------:|----------:|-------:|--------:|----------:|
| gemini-3-flash-preview |     0.8806 |    0.893  | 0.8736 |  0.8164 |    0.8659 |
| claude-sonnet-4.6      |     0.8273 |    0.7747 | 0.8253 |  0.7823 |    0.8024 |
| claude-haiku-4.5       |     0.6371 |    0.6205 | 0.7459 |  0.6861 |    0.6724 |
| qwen3-235b-a22b-2507   |     0.6664 |    0.5876 | 0.62   |  0.6314 |    0.6264 |

### Soft

| Model                  |   Calories |   Protein |     Fat |   Carbs |   Mean R² |
|:-----------------------|-----------:|----------:|--------:|--------:|----------:|
| gemini-3-flash-preview |     0.6718 |    0.8383 |  0.4534 |  0.7469 |    0.6776 |
| claude-sonnet-4.6      |     0.6991 |    0.6927 |  0.6376 |  0.6366 |    0.6665 |
| claude-haiku-4.5       |     0.6277 |    0.5535 |  0.5477 |  0.6474 |    0.5941 |
| qwen3-235b-a22b-2507   |     0.2208 |    0.4453 | -0.2554 |  0.6452 |    0.264  |

### Raw

| Model                  |   Calories |   Protein |    Fat |   Carbs |   Mean R² |
|:-----------------------|-----------:|----------:|-------:|--------:|----------:|
| gemini-3-flash-preview |     0.9046 |    0.9281 | 0.9122 |  0.8103 |    0.8888 |
| claude-sonnet-4.6      |     0.8293 |    0.8227 | 0.834  |  0.7821 |    0.817  |
| qwen3-235b-a22b-2507   |     0.7051 |    0.6685 | 0.6905 |  0.6015 |    0.6664 |
| claude-haiku-4.5       |     0.5914 |    0.6453 | 0.7493 |  0.6663 |    0.6631 |

## Database-grounded technique

The tables above measure models *estimating* macros directly. The
[meal-plan-skill](https://github.com/hugobellamy/meal-plan-skill/) instead uses a
**database-grounded** technique: the model decomposes a meal into components and
weights, fuzzy-searches the McCance database for each, picks the best-matching
row, and the macros are **summed from the real per-100g rows** — the model never
produces a nutrition number itself. Run it with `04_db_grounded.py`:

```bash
uv run 04_db_grounded.py --mode oracle                 # no API key; the ceiling
uv run 04_db_grounded.py --mode llm --model google/gemini-2.0-flash-001 --n 200
```

Results land in `data/results/db-grounded-*.json` and are scored by
`03_analysis.ipynb` alongside the estimate-only models.

> ⚠️ **These numbers are optimistic — the test set is generated from the same
> database the technique looks up in.** Each benchmark item's ground-truth macros
> are computed by summing McCance rows for its labelled components (see
> `01_build_datasets.ipynb`), so the "correct answer" for every item literally
> lives in the lookup table the technique searches. In real use the food eaten
> often has no exact McCance row (brand products, restaurant dishes, personal
> recipes), and portion estimation — the largest error source — still falls to
> the model. Treat the database-grounded scores as an **upper bound**, not a
> like-for-like field accuracy. The `oracle` mode (perfect decomposition, then
> search+sum) quantifies that ceiling; the `llm` mode is the fairer comparison
> against the estimate-only models, since it sees only the natural-language prompt.

### Results

| Technique | Calories | Protein | Fat | Carbs | Mean R² |
|:----------|---------:|--------:|----:|------:|--------:|
| `oracle` — ceiling, no LLM (search+sum) | 0.9999 | 0.9993 | 1.0000 | 1.0000 | **0.9998** |
| `llm` — decompose→search→pick→sum | _run with a key_ | | | | |

The `oracle` row is **~1.0 by construction** — given the correct food names and
weights, searching McCance returns the very rows the labels were summed from. It
is not a measure of real accuracy; it is a measurement of the leakage itself, and
the reason these scores can't be compared head-to-head with the estimate-only
models above. The `llm` row (which sees only the natural-language prompt) is the
fair comparison and will be lower — fill it in by running `--mode llm`.

