#!/usr/bin/env python3
"""Benchmark the DATABASE-GROUNDED technique (the meal-plan-skill approach).

Instead of asking the model to estimate macros from a description, we:
  stage 1 — model decomposes the prompt into components [{food, grams}]
  stage 2 — for each component, fuzzy-search the McCance database and let the
            model pick the best matching row
  sum     — add up the chosen rows scaled by grams; that is the prediction

Output is a data/results/<tag>.json in the SAME record schema as 02_benchmark,
so 03_analysis.ipynb scores it next to the estimate-only models automatically.

Modes
-----
  oracle  No API key needed. Uses the dataset's own component food_names+weights
          as a perfect stage-1/stage-2, then search+sum. This is the CEILING of
          the technique and exposes the leakage: the benchmark's ground truth IS
          summed McCance rows, so searching McCance recovers it almost exactly.
  llm     The real technique. Needs OPENROUTER_API_KEY. Only the natural-language
          `prompt` is shown to the model — never the components or weights.

Usage
-----
  uv run 04_db_grounded.py --mode oracle --tag db-grounded-oracle
  uv run 04_db_grounded.py --mode llm --model google/gemini-2.0-flash-001 \
                           --n 200 --tag db-grounded-gemini-2.0-flash
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

DATA = Path("data")
SRC = DATA / "source" / "McCance_Widdowsons_Composition_of_Foods_Integrated_Dataset_2021..xlsx"
LOOKUP_CSV = DATA / "foods_lookup.csv"
COLS = ["calories", "protein_g", "fat_g", "carbs_g"]

# Same column mapping the dataset's ground truth was built from (01_build_datasets),
# so oracle search resolves to the very rows that produced the labels.
KEEP = {
    "Food Code": "food_code", "Food Name": "food_name",
    "Protein (g)": "protein_g", "Fat (g)": "fat_g",
    "Carbohydrate (g)": "carbs_g", "Energy (kcal) (kcal)": "calories",
}


def build_lookup() -> pd.DataFrame:
    if LOOKUP_CSV.exists():
        return pd.read_csv(LOOKUP_CSV)
    raw = pd.read_excel(SRC, sheet_name="1.3 Proximates", header=0, skiprows=[1, 2])
    df = raw[list(KEEP)].rename(columns=KEEP).copy()
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.dropna(subset=["food_name"]).reset_index(drop=True)
    df["food_name"] = df["food_name"].str.strip()
    df.to_csv(LOOKUP_CSV, index=False)
    return df


class FoodDB:
    """Lightweight fuzzy search + per-100g lookup over the McCance table."""

    def __init__(self, df: pd.DataFrame):
        self.records = df.to_dict("records")
        self._names = [(r["food_name"].lower(), r) for r in self.records]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = (query or "").lower().strip()
        toks = [t for t in q.replace(",", " ").split() if t]
        scored = []
        for name, rec in self._names:
            hits = sum(1 for t in toks if t in name)
            ratio = difflib.SequenceMatcher(None, q, name).ratio()
            score = hits + 0.5 * ratio - 0.0005 * len(name)
            if hits or ratio > 0.4:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    @staticmethod
    def scaled(rec: dict, grams: float) -> dict:
        f = grams / 100.0
        return {c: rec[c] * f for c in COLS}


def sum_components(db: FoodDB, comps: list[tuple[str, float, dict | None]]) -> dict:
    """comps: list of (search_query, grams, chosen_rec_or_None). If chosen is
    None, search and take top-1."""
    pred = {c: 0.0 for c in COLS}
    for query, grams, chosen in comps:
        rec = chosen
        if rec is None:
            hits = db.search(query, limit=1)
            if not hits:
                continue
            rec = hits[0]
        s = db.scaled(rec, grams)
        for c in COLS:
            pred[c] += s[c]
    return {c: round(pred[c], 1) for c in COLS}


# ---------------------------------------------------------------- oracle mode
def predict_oracle(db: FoodDB, item: dict) -> dict:
    comps = [(c["food_name"], c["weight_g"], None) for c in item["components"]]
    return sum_components(db, comps)


# ------------------------------------------------------------------- llm mode
STAGE1_SYSTEM = (
    "You are a food expert. Break the described meal into its component foods and "
    "estimate each component's weight in grams as actually eaten. Return ONLY a "
    'JSON array of objects: [{"food": "<short food name>", "grams": <number>}]. '
    "Use one object per ingredient; a single-item meal returns an array of length 1."
)


def _stage1(client, model: str, prompt: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": STAGE1_SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    return [{"food": str(d["food"]), "grams": float(d["grams"])} for d in data]


def _stage2_pick(client, model: str, food: str, candidates: list[dict]) -> dict:
    """Ask the model which candidate row best matches `food`. Fall back to top-1."""
    if not candidates:
        return None
    listing = "\n".join(f"{i}: {c['food_name']}" for i, c in enumerate(candidates))
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": (
            f'Which database entry best matches the food "{food}"? Reply with ONLY '
            f'the integer index (0-based). If none fit, reply 0.\n{listing}')}],
    )
    text = (resp.choices[0].message.content or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    idx = int(digits) if digits else 0
    return candidates[min(idx, len(candidates) - 1)]


def predict_llm(client, model: str, db: FoodDB, item: dict) -> dict:
    try:
        comps_in = _stage1(client, model, item["prompt"])
    except Exception as e:  # noqa: BLE001 — record failures as empty prediction
        print(f"stage1 fail '{item['prompt'][:40]}': {e}")
        return {c: None for c in COLS}
    comps = []
    for comp in comps_in:
        cands = db.search(comp["food"], limit=10)
        try:
            chosen = _stage2_pick(client, model, comp["food"], cands)
        except Exception:  # noqa: BLE001
            chosen = cands[0] if cands else None
        comps.append((comp["food"], comp["grams"], chosen))
    return sum_components(db, comps)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "llm"], required=True)
    ap.add_argument("--model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--tag", help="results filename stem (data/results/<tag>.json)")
    ap.add_argument("--n", type=int, default=0, help="limit to first N items (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = pd.read_json(DATA / "benchmark.json")
    if args.n:
        ds = ds.head(args.n).copy()
    db = FoodDB(build_lookup())
    items = ds.to_dict("records")
    preds: list[dict] = [None] * len(items)

    if args.mode == "oracle":
        for i, item in enumerate(tqdm(items, desc="oracle")):
            preds[i] = predict_oracle(db, item)
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        with tqdm(total=len(items), desc=args.model.split("/")[-1]) as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(predict_llm, client, args.model, db, it): i
                        for i, it in enumerate(items)}
                for fut in as_completed(futs):
                    preds[futs[fut]] = fut.result()
                    bar.update(1)

    preds_df = pd.DataFrame(preds).rename(columns={c: f"pred_{c}" for c in COLS})
    out = pd.concat([ds.reset_index(drop=True), preds_df], axis=1)

    tag = args.tag or (f"db-grounded-{args.mode}" if args.mode == "oracle"
                       else f"db-grounded-{args.model.split('/')[-1]}")
    (DATA / "results").mkdir(exist_ok=True)
    out_path = DATA / "results" / f"{tag}.json"
    out.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved {len(out)} predictions -> {out_path}")

    # Preliminary R2 so we can eyeball it without opening 03_analysis.
    try:
        from sklearn.metrics import r2_score
        print("Preliminary R2 (this run):")
        r2s = []
        for c in COLS:
            m = out[f"pred_{c}"].notna()
            r2 = r2_score(out.loc[m, c], out.loc[m, f"pred_{c}"])
            r2s.append(r2)
            print(f"  {c:<10} {r2:.4f}")
        print(f"  {'mean':<10} {sum(r2s) / len(r2s):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped preliminary scoring: {e})")


if __name__ == "__main__":
    main()
