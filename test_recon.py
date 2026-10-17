"""Tests - run with:  python -m pytest -v   (uses a 200-row fake dataset, no internet needed)"""
import sqlite3

import pandas as pd
import pytest

import recon

N = 200


@pytest.fixture
def src():
    return pd.DataFrame({"order_id": [f"{i:032x}" for i in range(N)],
                         "payment_sequential": "1", "payment_type": "credit_card",
                         "payment_installments": "1",
                         "payment_value": [f"{10 + i * 1.25:.2f}" for i in range(N)]})


def run(tmp_path, src, tgt, client=None):
    src.to_csv(tmp_path / "s.csv", index=False)
    tgt.to_csv(tmp_path / "t.csv", index=False)
    out = recon.run(tmp_path / "s.csv", tmp_path / "t.csv", tmp_path / "out", client, verbose=False)
    return out, {c["check"]: c for c in out["checks"]}


class FakeAI(recon.LLMClient):
    name = "fake"
    def __init__(self, answer=None, error=None): self.answer, self.error = answer, error
    def complete(self, prompt):
        if self.error: raise self.error
        return self.answer


def test_clean_data_passes(tmp_path, src):
    out, _ = run(tmp_path, src, recon.transform(src))
    assert out["overall"] == "PASS"


def test_every_injected_defect_is_found(tmp_path, src):
    counts = {"missing_in_target": 7, "extra_in_target": 4, "duplicate_key": 3,
              "amount_mismatch": 6, "null_key": 2, "invalid_amount": 3}
    out, c = run(tmp_path, src, recon.inject_defects(recon.transform(src), counts, seed=1))
    for name, n in counts.items():
        assert c[name]["issue_count"] == n, name
    assert c["row_count"]["target_value"] == N - 7 + 3 + 2 + 4


def test_empty_target(tmp_path, src):                       # edge case
    _, c = run(tmp_path, src, pd.DataFrame(columns=recon.TARGET_COLS))
    assert c["missing_in_target"]["issue_count"] == N


def test_missing_column_raises(tmp_path, src):              # edge case: schema
    with pytest.raises(recon.SchemaError, match="amount"):
        run(tmp_path, src, recon.transform(src).drop(columns="amount"))


def test_rerun_is_idempotent(tmp_path, src):
    run(tmp_path, src, recon.transform(src))
    run(tmp_path, src, recon.transform(src))
    with sqlite3.connect(tmp_path / "out" / "recon.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_target").fetchone()[0] == N


def test_ai_failure_uses_fallback(tmp_path, src):           # edge case: service down
    out, _ = run(tmp_path, src, recon.transform(src).iloc[5:], FakeAI(error=TimeoutError("timeout")))
    assert out["written_by"].startswith("template") and "timeout" in out["written_by"]
    assert "5 missing in target" in out["ai_text"]


def test_ai_invented_number_is_rejected(tmp_path, src):     # edge case: hallucination
    lie = "The run failed because 7 rows are missing in the target. Reload the batch and add a key."
    out, _ = run(tmp_path, src, recon.transform(src).iloc[5:], FakeAI(answer=lie))
    assert "invented numbers" in out["written_by"]


def test_ai_honest_answer_is_accepted(tmp_path, src):
    good = (f"The run failed: 5 rows are missing in the target, so source has {N} rows and target "
            f"has {N - 5}. Likely a partial load; reload the missing keys.")
    out, _ = run(tmp_path, src, recon.transform(src).iloc[5:], FakeAI(answer=good))
    assert out["written_by"] == "fake" and out["ai_text"] == good
