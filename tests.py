"""Sanity tests for the pipeline outputs. Run after pipeline.py:  python tests.py"""
import json, pandas as pd, numpy as np
R = json.load(open("outputs/results.json"))
s = pd.read_csv("outputs/clean_sales.csv", parse_dates=["InvoiceDate"])
log = R["cleaning_log"]

def test_log_reconciles():
    left = log[0]["rows"] - sum(r["removed"] for r in log[1:])
    assert left == R["clean_rows"] == len(s), (left, R["clean_rows"], len(s))

def test_clean_data_has_no_known_defects():
    assert not s.duplicated().any()
    assert (s.Quantity > 0).all() and (s.UnitPrice > 0).all()
    assert not s.InvoiceNo.astype(str).str.startswith("C").any()
    assert s.StockCode.astype(str).str.match(r"^\d{5}[A-Za-z]{0,2}$").all()

def test_revenue_matches():
    assert abs((s.Quantity * s.UnitPrice).sum() - R["clean_revenue"]) < 1

def test_no_label_leakage():
    m = R["model"]
    last_train = pd.Timestamp(m["train_cutoffs"][-1]) + pd.Timedelta(days=m["horizon_days"])
    assert last_train <= pd.Timestamp(m["test_cutoff"])

def test_model_outputs_sane():
    for k, v in R["model"]["results"].items():
        assert 0.5 < v["roc_auc"] < 1 and v["auc_ci"][0] <= v["roc_auc"] <= v["auc_ci"][1]
    assert all(0 <= a["churn_prob"] <= 1 for a in R["at_risk"])
    p = pd.read_csv("outputs/at_risk_customers.csv"); assert p.churn_prob.between(0, 1).all()

def test_segments_partition_customers():
    assert sum(x["customers"] for x in R["rfm"]) == R["n_customers"]
    assert abs(sum(x["rev_share"] for x in R["rfm"]) - 1) < 1e-6

def test_month_series_lengths():
    n = len(R["months"])
    assert all(len(v) == n for v in R["monthly"].values()) and all(len(v) == n for v in R["state_revenue"].values())

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("PASS", f.__name__)
    print(len(fns), "tests passed")
