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
**database-grounded** technique that mirrors an agent counting calories with a
search tool, in two LLM calls:

1. **search terms** — the model turns the meal into database search terms;
2. **(tool)** — each term is fuzzy-searched in McCance for candidate rows;
3. **pick + weigh** — shown every term's candidates (with kcal/100g), the model
   picks the matching entry *and* the weight for each, choosing the form actually
   eaten (canned vs dried, boiled vs raw) and weighing on that entry's basis;
4. **sum** — the macros are summed from the chosen per-100g rows. The model never
   produces a nutrition number itself.

`04_db_grounded.py` runs it in three modes:

```bash
uv run 04_db_grounded.py --mode oracle                     # ceiling, no API key
uv run 04_db_grounded.py --mode estimate --model M --n 50  # without tool (direct guess)
uv run 04_db_grounded.py --mode llm      --model M --n 50  # with tool (2-call workflow)
# local OpenAI-compatible server (e.g. LM Studio):
uv run 04_db_grounded.py --mode llm --base-url http://HOST:1234/v1 --model google/gemma-4-31b --n 50
```

Results land in `data/results/{estimate,db-grounded}-*.json` and are scored by
`03_analysis.ipynb` alongside the estimate-only models.

### Results — with tool vs without, same 50 items (mean R²)

| Model | Without tool | With tool | Δ | N |
|:------|-----------:|---------:|------:|--:|
| gemma-4-31b (local) | 0.937 | **0.985** | +0.048 | 42 |
| gemini-3.5-flash | 0.948 | **0.984** | +0.036 | 50 |
| claude-sonnet-4.6 | 0.825 | **0.987** | +0.162 | 50 |
| claude-haiku-4.5 | 0.776 | **0.923** | +0.146 | 50 |
| qwen3-235b-a22b-2507 | 0.736 | **0.985** | +0.248 | 49 |
| `oracle` (perfect decomposition, no LLM) | — | **0.9998** | — | 50 |

**The tool helps every model, most where the model is weakest** (qwen +0.25,
sonnet +0.16, haiku +0.15). Without it the models spread from 0.74 to 0.95; with
it they converge to ~0.92–0.99 — the database equalizes nutrition knowledge, so a
cheap model with the tool matches an expensive one. (N<50 where a model failed to
return parseable JSON on a few items — gemma-4 on a small local context window,
qwen on one; those items are dropped from *both* columns so each row stays
like-for-like.)

> ⚠️ **Absolute with-tool scores are optimistic — the test set is generated from
> the same database the technique looks up in.** Each item's ground-truth macros
> are summed McCance rows for its labelled components (`01_build_datasets.ipynb`),
> so the "correct answer" lives in the lookup table the technique searches. The
> `oracle` row (0.9998) measures that leakage directly: with perfect names and
> weights, search+sum just returns the label rows. In real use the eaten food
> often has no exact McCance row (brands, restaurant dishes, recipes). So read the
> absolute with-tool numbers as an **upper bound** — but the **Δ within each model**
> is a controlled experiment (same items, same model, tool vs no tool) and is
> consistently, substantially positive.
