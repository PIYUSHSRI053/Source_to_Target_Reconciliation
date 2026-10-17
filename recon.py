# %%
import json, os, re, sqlite3, sys, urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA = Path("data"); OUT = Path("output")
SOURCE_CSV = DATA / "olist_order_payments_dataset.csv"
TARGET_CSV = DATA / "target_payments.csv"
# Public Kaggle dataset "Brazilian E-Commerce by Olist" (GitHub mirror, no login needed)
URL = ("https://raw.githubusercontent.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist/"
       "HEAD/files/olist_order_payments_dataset.csv")
SOURCE_COLS = ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]
TARGET_COLS = ["order_id", "payment_seq", "payment_type", "installments", "amount"]
DEFECTS = {"missing_in_target": 25, "extra_in_target": 10, "duplicate_key": 8,
           "amount_mismatch": 15, "null_key": 3, "invalid_amount": 5}


class SchemaError(ValueError):
    pass


# %% [markdown]
# ## Step 1 - Get the source, build a target with known defects

# %%
def download_source(path=SOURCE_CSV):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print("downloading Olist payments dataset ...")
        urllib.request.urlretrieve(URL, path)
    return pd.read_csv(path, dtype=str)


def transform(src):
    """The 'ETL job': rename columns, upper-case type, round amount."""
    return pd.DataFrame({"order_id": src["order_id"],
                         "payment_seq": src["payment_sequential"].astype(int),
                         "payment_type": src["payment_type"].str.upper(),
                         "installments": src["payment_installments"].astype(int),
                         "amount": src["payment_value"].astype(float).round(2)})


def inject_defects(tgt, counts=DEFECTS, seed=42):
    """Break the target on purpose. Each defect uses different rows, so results are exact."""
    rng = np.random.default_rng(seed)
    df = tgt.reset_index(drop=True).astype({"amount": object})
    sizes = [counts[k] for k in ["missing_in_target", "duplicate_key", "amount_mismatch",
                                 "invalid_amount", "null_key"]]
    rows = rng.choice(len(df), sum(sizes), replace=False)
    missing, dup, changed, invalid, nullk = np.split(rows, np.cumsum(sizes)[:-1])

    df.loc[changed, "amount"] = [round(float(a) * 1.1 + 1, 2) for a in df.loc[changed, "amount"]]
    df.loc[invalid, "amount"] = (["N/A", "abc", "", "12,5O", "null"] * len(invalid))[:len(invalid)]
    nulls = df.loc[nullk].assign(order_id=None)
    extra = tgt.sample(counts["extra_in_target"], random_state=seed).assign(
        order_id=[rng.bytes(16).hex() for _ in range(counts["extra_in_target"])])
    out = pd.concat([df.drop(index=missing), df.loc[dup], nulls, extra], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# %% [markdown]
# ## Step 2 - Load into SQLite (raw layer)
# Schema is validated; non-numeric amounts become NULL. `replace` makes reruns idempotent.

# %%
def load(conn, source_csv, target_csv):
    src, tgt = pd.read_csv(source_csv, dtype=str), pd.read_csv(target_csv, dtype=str)
    for name, df, cols in [("source", src, SOURCE_COLS), ("target", tgt, TARGET_COLS)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SchemaError(f"{name} file is missing columns: {missing}")
    src["payment_value"] = pd.to_numeric(src["payment_value"], errors="coerce")
    src["payment_sequential"] = pd.to_numeric(src["payment_sequential"]).astype("Int64")
    tgt["amount"] = pd.to_numeric(tgt["amount"], errors="coerce")
    tgt["payment_seq"] = pd.to_numeric(tgt["payment_seq"], errors="coerce").astype("Int64")
    src[SOURCE_COLS].to_sql("raw_source", conn, if_exists="replace", index=False)
    tgt[TARGET_COLS].to_sql("raw_target", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_t ON raw_target(order_id, payment_seq)")
    return len(src), len(tgt)


# %% [markdown]
# ## Step 3 - Reconciliation checks (SQL)
# Key = `(order_id, payment_seq)`. `agg_` checks compare one number; `exc_` checks list bad rows.
# To add a rule, add one more entry here.

# %%
JOIN = "t.order_id = s.order_id AND t.payment_seq = s.payment_sequential"
CHECKS = {
    "agg_row_count": "SELECT (SELECT COUNT(*) FROM raw_source), (SELECT COUNT(*) FROM raw_target)",
    "agg_amount_total": """SELECT (SELECT ROUND(SUM(payment_value),2) FROM raw_source),
                                  (SELECT ROUND(COALESCE(SUM(amount),0),2) FROM raw_target)""",
    "exc_missing_in_target": f"""SELECT s.order_id, s.payment_sequential AS payment_seq,
        s.payment_value AS source_amount, NULL AS target_amount
        FROM raw_source s LEFT JOIN raw_target t ON {JOIN} WHERE t.order_id IS NULL""",
    "exc_extra_in_target": f"""SELECT t.order_id, t.payment_seq, NULL, t.amount
        FROM raw_target t LEFT JOIN raw_source s ON {JOIN}
        WHERE s.order_id IS NULL AND t.order_id IS NOT NULL""",
    "exc_duplicate_key": """SELECT order_id, payment_seq, NULL, MAX(amount) FROM raw_target
        WHERE order_id IS NOT NULL GROUP BY order_id, payment_seq HAVING COUNT(*) > 1""",
    "exc_null_key": """SELECT order_id, payment_seq, NULL, amount FROM raw_target
        WHERE order_id IS NULL OR payment_seq IS NULL""",
    "exc_amount_mismatch": f"""SELECT DISTINCT s.order_id, s.payment_sequential, s.payment_value, t.amount
        FROM raw_source s JOIN raw_target t ON {JOIN}
        WHERE t.amount IS NOT NULL AND ABS(t.amount - s.payment_value) > 0.01""",
    "exc_invalid_amount": "SELECT order_id, payment_seq, NULL, NULL FROM raw_target WHERE amount IS NULL",
}
EXC_COLS = ["order_id", "payment_seq", "source_amount", "target_amount"]


def reconcile(conn, run_id):
    results, frames = [], []
    for name, sql in CHECKS.items():
        check = name[4:]
        if name.startswith("agg_"):
            s, t = (float(v or 0) for v in conn.execute(sql).fetchone())
            ok = abs(s - t) <= 0.01
            results.append(dict(check=check, status="PASS" if ok else "FAIL",
                                source_value=s, target_value=t, issue_count=0 if ok else 1))
        else:
            rows = pd.DataFrame(conn.execute(sql).fetchall(), columns=EXC_COLS)
            frames.append(rows.assign(check_name=check))
            results.append(dict(check=check, status="PASS" if rows.empty else "FAIL",
                                source_value=None, target_value=None, issue_count=len(rows)))
    exceptions = pd.concat(frames, ignore_index=True).assign(run_id=run_id)
    pd.DataFrame(results).rename(columns={"check": "check_name"}).assign(run_id=run_id).to_sql("recon_results", conn, if_exists="append", index=False)
    exceptions.to_sql("recon_exceptions", conn, if_exists="append", index=False)
    return results, exceptions


# %% [markdown]
# ## Step 4 - AI incident summary
# Only the small summary JSON is sent (never raw data). Every number in the answer must exist
# in the report, otherwise the answer is rejected. Any failure falls back to a template.

# %%
class LLMClient:
    """Interface: any model just needs complete(prompt) -> str."""
    name = "base"
    def complete(self, prompt): raise NotImplementedError


class GeminiClient(LLMClient):
    name = "gemini"
    def __init__(self, key, model="gemini-2.5-flash"):
        self.key, self.model = key, model
    def complete(self, prompt):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                          headers={"x-goog-api-key": self.key},
                          json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def get_client():
    key = os.getenv("GEMINI_API_KEY")
    try:                                   # Colab: Secrets panel (key icon) -> GEMINI_API_KEY
        from google.colab import userdata
        key = key or userdata.get("GEMINI_API_KEY")
    except Exception:
        pass
    return GeminiClient(key) if key else None


PROMPT = """You are a data-quality analyst. Using ONLY the JSON below, write a 4-6 sentence plain-text
incident summary: overall result, each defect and its count, row and amount difference, likely
cause, one next action. Copy numbers exactly; do not calculate new numbers. No markdown.
JSON:
{}"""


def template_summary(s):
    c = {x["check"]: x for x in s["checks"]}
    bad = [f"{x['issue_count']} {x['check'].replace('_', ' ')}" for x in s["checks"]
           if x["status"] == "FAIL" and x["source_value"] is None]
    return (f"Run {s['run_id']} {s['overall']}: {s['checks_failed']} of {len(s['checks'])} checks failed. "
            f"Rows: {c['row_count']['source_value']:,.0f} source vs {c['row_count']['target_value']:,.0f} target. "
            f"Amount: R$ {c['amount_total']['source_value']:,.2f} vs R$ {c['amount_total']['target_value']:,.2f}. "
            + (f"Defects: {', '.join(bad)}. Likely partial load, re-insert and bad transforms; "
               "reload affected keys and add a unique key on (order_id, payment_seq)." if bad else "No action needed."))


def numbers(text):
    return {round(float(n.replace(",", "")), 2) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def validate(text, s):
    if not text or not 60 <= len(text) <= 1500:
        return "answer empty or wrong length"
    allowed = numbers(json.dumps(s, default=str)) | {
        round(abs(x["source_value"] - x["target_value"]), 2) for x in s["checks"] if x["source_value"] is not None}
    invented = numbers(text) - allowed
    return f"invented numbers {sorted(invented)[:5]}" if invented else None


def ai_summary(s, client):
    if client is None:
        return template_summary(s), "template (AI off / no API key)"
    try:
        text = client.complete(PROMPT.format(json.dumps(s, default=str)))
        problem = validate(text, s)
    except Exception as e:
        problem, text = f"AI call failed: {e}"[:200], None
    return (template_summary(s), f"template ({problem})") if problem else (text, client.name)


# %% [markdown]
# ## Step 5 - Run everything

# %%
def run(source_csv=SOURCE_CSV, target_csv=TARGET_CSV, out=OUT, client="auto", verbose=True):
    out = Path(out); out.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with sqlite3.connect(out / "recon.db") as conn:
        n_src, n_tgt = load(conn, source_csv, target_csv)
        results, exceptions = reconcile(conn, run_id)
        failed = sum(r["status"] == "FAIL" for r in results)
        summary = dict(run_id=run_id, overall="FAIL" if failed else "PASS", checks_failed=failed,
                       checks=results, examples=exceptions.head(10).to_dict("records"))
        text, written_by = ai_summary(summary, get_client() if client == "auto" else client)
        pd.DataFrame([dict(run_id=run_id, written_by=written_by, text=text)]).to_sql(
            "ai_summaries", conn, if_exists="append", index=False)

    exceptions.to_csv(out / "exception_report.csv", index=False)
    (out / "incident_summary.md").write_text(f"# Run {run_id}: {summary['overall']}\n\n"
                                             f"_Written by: {written_by}_\n\n{text}\n")
    if verbose:
        print(f"source={n_src:,} rows  target={n_tgt:,} rows  run_id={run_id}\n")
        for r in results:
            detail = (f"{r['source_value']:,.2f} vs {r['target_value']:,.2f}" if r["source_value"] is not None
                      else f"{r['issue_count']} rows")
            print(f"  {r['status']}  {r['check']:<18} {detail}")
        print(f"\nexception report: {out/'exception_report.csv'} ({len(exceptions)} rows)")
        print(f"AI summary ({written_by}):\n{text}")
    return summary | {"ai_text": text, "written_by": written_by}


def main():
    src = download_source()
    inject_defects(transform(src)).to_csv(TARGET_CSV, index=False)
    print(f"target built with defects: {DEFECTS}\n")
    return run(client=None if "--no-ai" in sys.argv else "auto")


# %%
if __name__ == "__main__":
    main()
