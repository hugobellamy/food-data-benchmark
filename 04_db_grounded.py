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
# Generous cap so reasoning models (which spend tokens thinking before answering)
# still reach the JSON answer. _chat_json doubles this on a truncation retry.
MAX_TOKENS = 4096

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


# -------------------------------------------------------------- estimate mode
# The "without tool" baseline: the model guesses macros directly (same prompt as
# 02_benchmark.ipynb), no database. Lets us compare with-tool vs without-tool for
# the same model on the same items.
ESTIMATE_SYSTEM = (
    "You are a nutrition expert. Given a food description, estimate the total "
    "nutritional content.\n\nReturn ONLY a JSON object with these keys:\n"
    '- "calories": total kcal (number)\n- "protein_g": grams of protein (number)\n'
    '- "fat_g": grams of fat (number)\n- "carbs_g": grams of carbohydrates (number)\n\n'
    "Be as accurate as possible. Do not explain, just return the JSON."
)


def predict_estimate(client, model: str, db: FoodDB, item: dict) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": ESTIMATE_SYSTEM},
                      {"role": "user", "content": item["prompt"]}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        d = json.loads(text)
        return {c: float(d[c]) for c in COLS}
    except Exception as e:  # noqa: BLE001
        print(f"estimate fail '{item['prompt'][:40]}': {e}")
        return {c: None for c in COLS}


# ------------------------------------------------------------------- llm mode
# Faithful automation of the agent's calorie-counting loop, in exactly TWO LLM
# calls regardless of ingredient count:
#   call 1 — meal description -> list of database search terms
#   (tool)  -> fuzzy-search each term, collect candidate rows
#   call 2 — given every term's candidates, pick the entry index AND the weight
#            for each (weight conditioned on the chosen entry's basis)
#   sum    -> chosen rows scaled by grams
SEARCH_TERMS_SYSTEM = (
    "You are counting the calories of a meal using a food-composition database. "
    "Break the meal into its component foods and give the search term you would "
    "look each one up by. Return ONLY a JSON array of short search strings, one "
    'per distinct ingredient, e.g. ["chicken breast", "basmati rice"]. A '
    "single-item meal returns one term."
)
PICK_SYSTEM = (
    "You are counting the calories of a meal using a food-composition database. "
    "For each food you searched, you are shown the candidate database entries "
    "(with kcal per 100 g). For each food, in order, choose the entry that best "
    "matches what was actually eaten and estimate the weight eaten in grams.\n"
    "- Match the FORM eaten: canned vs dried, boiled vs raw, with/without skin.\n"
    "- The weight must match that entry's basis (e.g. a 'boiled' entry takes the "
    "cooked weight; a 'dried/raw' entry takes the dry weight).\n"
    'Return ONLY a JSON array with one object per food, in order: '
    '[{"index": <int>, "grams": <number>}].'
)


def _extract_json(text: str):
    """Parse JSON from a model reply that may be fenced or wrapped in prose.

    Tries a direct parse, then strips ``` fences, then scans for the first
    balanced [...] / {...} block (ignoring brackets inside strings). Raises
    ValueError if nothing parses — including silently-truncated replies."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty reply")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        inner = text.split("\n", 1)[1].rsplit("```", 1)[0].strip() if "\n" in text else text
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            text = inner
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("no parseable JSON (possibly truncated)")


def _chat_json(client, model: str, system: str, user: str, retries: int = 2):
    """Chat call returning parsed JSON, retrying with more tokens on truncation."""
    last = None
    for attempt in range(retries + 1):
        budget = MAX_TOKENS * (2 if attempt else 1)
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=budget,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
        try:
            return _extract_json(content)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            fin = resp.choices[0].finish_reason
            print(f"  parse retry {attempt} (finish={fin}): {repr(content)[:80]}")
    raise last


def _search_terms(client, model: str, prompt: str) -> list[str]:
    data = _chat_json(client, model, SEARCH_TERMS_SYSTEM, prompt)
    return [str(t) for t in data if str(t).strip()]


def _pick_entries(client, model: str, prompt: str,
                  term_cands: list[tuple[str, list[dict]]]) -> list[dict]:
    """One call: for every (term, candidates), return {index, grams}."""
    blocks = []
    for i, (term, cands) in enumerate(term_cands):
        lines = [f'Food {i} — "{term}":']
        for j, c in enumerate(cands):
            lines.append(f"  {j}: {c['food_name']} ({c['calories']:.0f} kcal/100g)")
        blocks.append("\n".join(lines))
    user = (f"Meal: {prompt}\n\n" + "\n\n".join(blocks)
            + f"\n\nReturn a JSON array of {len(term_cands)} objects "
              '[{"index", "grams"}], one per food in order.')
    data = _chat_json(client, model, PICK_SYSTEM, user)
    return data if isinstance(data, list) else [data]


def predict_llm(client, model: str, db: FoodDB, item: dict,
                cand_k: int = 8, **_ignored) -> dict:
    """Two-call agent workflow: search-terms -> tool search -> pick index+weight -> sum."""
    try:
        terms = _search_terms(client, model, item["prompt"])
    except Exception as e:  # noqa: BLE001
        print(f"search-terms fail '{item['prompt'][:40]}': {e}")
        return {c: None for c in COLS}
    term_cands = [(t, db.search(t, limit=cand_k)) for t in terms]
    term_cands = [(t, c) for t, c in term_cands if c]  # drop terms with no hits
    if not term_cands:
        return {c: 0.0 for c in COLS}
    try:
        picks = _pick_entries(client, model, item["prompt"], term_cands)
    except Exception as e:  # noqa: BLE001
        print(f"pick fail '{item['prompt'][:40]}': {e}")
        return {c: None for c in COLS}

    comps = []
    for (term, cands), pick in zip(term_cands, picks):
        try:
            idx = int(pick["index"])
            grams = float(pick["grams"])
        except (TypeError, ValueError, KeyError):
            idx, grams = 0, 0.0
        rec = cands[min(max(idx, 0), len(cands) - 1)]
        comps.append((term, grams, rec))
    return sum_components(db, comps)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "llm", "estimate"], required=True,
                    help="oracle=ceiling; llm=with tool; estimate=without tool")
    ap.add_argument("--model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                    help="OpenAI-compatible endpoint (e.g. a local LM Studio server)")
    ap.add_argument("--tag", help="results filename stem (data/results/<tag>.json)")
    ap.add_argument("--n", type=int, default=0, help="limit to first N items (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600, help="per-request timeout (s)")
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
        # Local OpenAI-compatible servers (LM Studio etc.) need no real key.
        api_key = os.environ.get("OPENROUTER_API_KEY") or "not-needed"
        client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
        if args.mode == "estimate":
            def predict(it):
                return predict_estimate(client, args.model, db, it)
        else:
            def predict(it):
                return predict_llm(client, args.model, db, it)
        with tqdm(total=len(items), desc=args.model.split("/")[-1]) as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(predict, it): i for i, it in enumerate(items)}
                for fut in as_completed(futs):
                    preds[futs[fut]] = fut.result()
                    bar.update(1)

    preds_df = pd.DataFrame(preds).rename(columns={c: f"pred_{c}" for c in COLS})
    out = pd.concat([ds.reset_index(drop=True), preds_df], axis=1)

    slug = args.model.split("/")[-1]
    default_tag = {"oracle": "db-grounded-oracle", "estimate": f"estimate-{slug}",
                   "llm": f"db-grounded-{slug}"}[args.mode]
    tag = args.tag or default_tag
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
