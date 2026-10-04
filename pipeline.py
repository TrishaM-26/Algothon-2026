"""
Online Retail - Algothon'26 ALG-DATA-01 (The Mystery Dataset)

Run:  python pipeline.py  <path-to-Online_Retail.xlsx>
Out:  outputs/clean_sales.csv, outputs/results.json, outputs/at_risk_customers.csv

Stages
  1. Clean (every rule is counted in the cleaning log)
  2. Explore (monthly decomposition, concentration, cohorts, RFM, anomalies, countries)
  3. Test hypotheses (Mann-Whitney, bootstrap)
  4. Churn model with time-based validation (baseline vs logistic vs gradient boosting)
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.inspection import permutation_importance

SRC = sys.argv[1] if len(sys.argv) > 1 else "Online_Retail.xlsx"
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
R = {}  # everything the dashboard needs

# ---------------------------------------------------------------- 1. CLEAN
raw = pd.read_excel(SRC)
raw["InvoiceNo"] = raw.InvoiceNo.astype(str)
raw["StockCode"] = raw.StockCode.astype(str)
raw["rev"] = raw.Quantity * raw.UnitPrice
log = [dict(step="Raw rows", rows=len(raw), removed=0, value=float(raw.rev.sum()))]

def step(name, before, after, value):
    log.append(dict(step=name, rows=len(after), removed=len(before) - len(after), value=float(value)))

df = raw.drop_duplicates()
step("Exact duplicate rows", raw, df, raw.loc[~raw.index.isin(df.index), "rev"].sum())

df["cancel"] = df.InvoiceNo.str.startswith("C")
is_prod = df.StockCode.str.match(r"^\d{5}[A-Za-z]{0,2}$")
non_prod = df[~is_prod]
fees = non_prod.groupby("StockCode").rev.sum().sort_values(ascending=False)
R["non_product_codes"] = [dict(code=k, value=float(v), rows=int((non_prod.StockCode == k).sum()))
                          for k, v in fees.head(6).items()]
d2 = df[is_prod]
step("Non-product codes (postage, fees, manual, vouchers)", df, d2, non_prod.rev.sum())

adj = (d2.Quantity < 0) & (~d2.cancel)
d3 = d2[~adj]
step("Stock adjustments (negative qty, not a cancellation)", d2, d3, d2[adj].rev.sum())

zp = d3.UnitPrice <= 0
d4 = d3[~zp]
step("Zero-price rows (free/promo/write-offs)", d3, d4, d3[zp].rev.sum())

sales = d4[~d4.cancel & (d4.Quantity > 0)].copy()
canc = d4[d4.cancel].copy()
step("Cancellation rows set aside (paired to orders or kept as returns below)", d4, sales, canc.rev.sum())

# match cancellations to the order they reverse (same customer, product, exact qty, earlier date)
pool = {}
for i, r in sales[sales.CustomerID.notna()].sort_values("InvoiceDate").iterrows():
    pool.setdefault((r.CustomerID, r.StockCode, r.Quantity), []).append((r.InvoiceDate, i))
matched_sales, matched_canc = [], []
for i, r in canc[canc.CustomerID.notna()].sort_values("InvoiceDate").iterrows():
    lst = pool.get((r.CustomerID, r.StockCode, -r.Quantity), [])
    cand = [x for x in lst if x[0] <= r.InvoiceDate]
    if cand:
        x = cand[-1]; lst.remove(x)
        matched_sales.append(x[1]); matched_canc.append(i)
rev_val = sales.loc[matched_sales, "rev"].sum()
rv = (sales.loc[matched_sales].groupby("InvoiceNo")
      .agg(customer=("CustomerID", "first"), date=("InvoiceDate", "first"), value=("rev", "sum"),
           lines=("StockCode", "size"), item=("Description", "first")).sort_values("value", ascending=False))
R["reversed_top"] = [dict(invoice=i, customer=int(r.customer), date=str(r.date.date()), value=float(r.value),
                          item=str(r["item"]).strip()) for i, r in rv.head(6).iterrows()]
naive = sales[sales.CustomerID.notna()].groupby("CustomerID").rev.sum().sort_values(ascending=False)
R["naive"] = dict(top1pct_share=float(naive.head(int(len(naive) * .01)).sum() / naive.sum()),
                  rank={str(int(c)): int(list(naive.index).index(c) + 1) for c in rv.head(2).customer},
                  value={str(int(c)): float(naive[c]) for c in rv.head(2).customer})
R["reversed_value"] = float(rev_val)
R["reversed_lines"] = len(matched_sales)
before = sales
sales = sales.drop(index=matched_sales)
returns = canc.drop(index=matched_canc)   # partial / unmatched returns, kept as returns
step("Orders that were placed then fully cancelled (matched pairs)", before, sales, rev_val)
R["unmatched_returns_value"] = float(returns.rev.sum())

sales["month"] = sales.InvoiceDate.dt.to_period("M")
guest = sales.CustomerID.isna()
R["guest_share_rows"] = float(guest.mean())
R["guest_share_rev"] = float(sales[guest].rev.sum() / sales.rev.sum())
log.append(dict(step="Guest orders (no CustomerID): kept for totals, excluded from customer analysis",
                rows=int(guest.sum()), removed=0, value=float(sales[guest].rev.sum())))
R["cleaning_log"] = log
R["clean_rows"] = len(sales); R["clean_revenue"] = float(sales.rev.sum())
sales.drop(columns=["cancel"]).to_csv(f"{OUT}/clean_sales.csv", index=False)

cust = sales[sales.CustomerID.notna()].copy()
cust["CustomerID"] = cust.CustomerID.astype(int)
returns = returns[returns.CustomerID.notna()].copy(); returns["CustomerID"] = returns.CustomerID.astype(int)
END = cust.InvoiceDate.max()
R["date_range"] = [str(cust.InvoiceDate.min().date()), str(END.date())]
R["n_customers"] = int(cust.CustomerID.nunique()); R["n_invoices"] = int(cust.InvoiceNo.nunique())
R["n_products"] = int(cust.StockCode.nunique())

# ---------------------------------------------------------------- 2. EXPLORE
months = [str(m) for m in sorted(cust.month.unique())]
R["months"] = months
first = cust.groupby("CustomerID").InvoiceDate.min().dt.to_period("M")
cust["first_m"] = cust.CustomerID.map(first)

# 2a. customer state per month: new / continuing (bought in prior 2 months) / reactivated (earlier, but not in prior 2 months)
act = cust.groupby(["CustomerID", "month"]).rev.sum().unstack(fill_value=0)
act = act.reindex(columns=pd.period_range(cust.month.min(), cust.month.max(), freq="M"), fill_value=0)
state_cnt, state_rev = {k: [] for k in ("new", "continuing", "reactivated")}, {k: [] for k in ("new", "continuing", "reactivated")}
cols = list(act.columns)
for j, m in enumerate(cols):
    a = act[m] > 0
    prior_any = (act[cols[:j]] > 0).any(axis=1) if j > 0 else pd.Series(False, index=act.index)
    prior2 = (act[cols[max(0, j - 2):j]] > 0).any(axis=1) if j > 0 else pd.Series(False, index=act.index)
    new = a & ~prior_any; cont = a & prior2; react = a & prior_any & ~prior2
    for k, mask in (("new", new), ("continuing", cont), ("reactivated", react)):
        state_cnt[k].append(int(mask.sum())); state_rev[k].append(float(act.loc[mask, m].sum()))
idx_sep = months.index("2011-09"); idx_nov = months.index("2011-11")
sm_idx = [months.index(m) for m in ("2011-06", "2011-07", "2011-08")]
R["state_customers"] = state_cnt; R["state_revenue"] = state_rev

mm = cust.groupby("month").agg(rev=("rev", "sum"), customers=("CustomerID", "nunique"), invoices=("InvoiceNo", "nunique"))
mm["aov"] = mm.rev / mm.invoices; mm["rev_per_cust"] = mm.rev / mm.customers
R["monthly"] = {c: [float(x) for x in mm[c]] for c in mm.columns}
# total revenue incl. guests for the headline trend
tot = sales.groupby("month").rev.sum(); R["monthly_total_rev"] = [float(x) for x in tot]

# 2b. concentration
cr = cust.groupby("CustomerID").rev.sum().sort_values(ascending=False)
n = len(cr); cum = cr.cumsum() / cr.sum()
R["top_share"] = {str(p): float(cr.head(int(n * p / 100)).sum() / cr.sum()) for p in (1, 5, 10, 20)}
xs = np.sort(cr.values); R["gini"] = float(1 - 2 * (np.cumsum(xs) / xs.sum()).sum() / len(xs) + 1 / len(xs))
idx = np.linspace(0, n - 1, 41).astype(int)
R["lorenz"] = [[0.0, 0.0]] + [[float((i + 1) / n), float(np.cumsum(xs)[::1][i] / xs.sum())] for i in idx]  # ascending cum share
R["lorenz"] = [[0.0, 0.0]] + [[float((i + 1) / n), float(np.cumsum(xs)[i] / xs.sum())] for i in idx]
inv_per = cust.groupby("CustomerID").InvoiceNo.nunique()
R["one_time_share"] = float((inv_per == 1).mean())
R["uk_share"] = float(cust[cust.Country == "United Kingdom"].rev.sum() / cust.rev.sum())

# 2c. cohorts (first-purchase month x months since)
cust["age"] = (cust.month - cust.first_m).apply(lambda x: x.n)
coh = cust.groupby(["first_m", "age"]).CustomerID.nunique().unstack(fill_value=0)
size = coh[0]
ret = coh.div(size, axis=0)
R["cohort"] = dict(cohorts=[str(c) for c in ret.index], size=[int(x) for x in size],
                   ret=[[None if (a + i) >= len(cols) else float(ret.iloc[i, a]) if a in ret.columns else None
                         for a in range(13)] for i in range(len(ret))])

# 2d. RFM
snap = END.normalize() + pd.Timedelta(days=1)
g = cust.groupby("CustomerID")
rfm = pd.DataFrame({"recency": (snap - g.InvoiceDate.max()).dt.days, "frequency": g.InvoiceNo.nunique(), "monetary": g.rev.sum()})
rfm["R"] = pd.qcut(rfm.recency.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
def seg(r):
    if r.R >= 4 and r.F >= 4: return "Champions"
    if r.R >= 3 and r.F >= 3: return "Loyal"
    if r.R >= 4 and r.F <= 2: return "New / recent"
    if r.R <= 2 and r.F >= 4: return "At risk (was loyal)"
    if r.R <= 2: return "Hibernating"
    return "Needs attention"
rfm["segment"] = rfm.apply(seg, axis=1)
sg = rfm.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
                                med_recency=("recency", "median"), med_freq=("frequency", "median"))
sg["rev_share"] = sg.revenue / sg.revenue.sum(); sg["cust_share"] = sg.customers / sg.customers.sum()
R["rfm"] = [dict(segment=k, **{c: float(v[c]) for c in sg.columns}) for k, v in sg.sort_values("revenue", ascending=False).iterrows()]

# 2e. anomalies: (1) reversed orders (above), (2) invoice-value outliers via robust z on log value,
#     (3) line prices far from the product's own median price
inv = cust.groupby("InvoiceNo").agg(CustomerID=("CustomerID", "first"), date=("InvoiceDate", "first"),
                                    value=("rev", "sum"), lines=("StockCode", "nunique"), qty=("Quantity", "sum"))
lv = np.log(inv.value); mad = np.median(np.abs(lv - lv.median()))
inv["robust_z"] = 0.6745 * (lv - lv.median()) / mad
R["n_high_value_invoices"] = int((inv.robust_z > 3.5).sum())
R["high_value_invoice_share"] = float(inv[inv.robust_z > 3.5].value.sum() / inv.value.sum())
desc = cust.groupby("InvoiceNo").Description.first()
R["largest_invoices"] = [dict(invoice=i, customer=int(r.CustomerID), date=str(r.date.date()), value=float(r.value),
                              lines=int(r.lines), item=str(desc[i]).strip(), z=float(r.robust_z))
                         for i, r in inv.sort_values("value", ascending=False).head(6).iterrows()]
med_price = cust.groupby("StockCode").UnitPrice.transform("median")
ratio = cust.UnitPrice / med_price
odd = cust[(ratio > 10) | (ratio < 0.1)]
R["price_outlier_lines"] = int(len(odd)); R["price_outlier_value"] = float(odd.rev.sum())
R["price_outliers"] = [dict(item=str(r.Description).strip(), price=float(r.UnitPrice), median=float(med_price[i]), qty=int(r.Quantity))
                       for i, r in odd.sort_values("rev", ascending=False).head(5).iterrows()]

# 2f. countries
cn = cust.groupby("Country").agg(revenue=("rev", "sum"), customers=("CustomerID", "nunique"), invoices=("InvoiceNo", "nunique"))
cn["aov"] = cn.revenue / cn.invoices; cn["rev_per_cust"] = cn.revenue / cn.customers
cn = cn.sort_values("revenue", ascending=False)
R["countries"] = [dict(country=k, **{c: float(v[c]) for c in cn.columns}) for k, v in cn.head(8).iterrows()]

# 2g. returns concentration
rc = returns.groupby("CustomerID").rev.sum().sort_values()
R["returns_total"] = float(rc.sum()); R["returns_customers"] = int(len(rc))
R["returns_top1pct_share"] = float(rc.head(max(1, int(len(rc) * .01))).sum() / rc.sum())
R["returns_top10_share"] = float(rc.head(10).sum() / rc.sum())
R["returns_median_ratio"] = float((-rc / cr.reindex(rc.index)).dropna().median())
R["returns_top"] = [dict(customer=int(k), value=float(v)) for k, v in rc.head(5).items()]

# ---------------------------------------------------------------- 3. HYPOTHESES
H = []
SUM = (cust.InvoiceDate >= "2011-06-01") & (cust.InvoiceDate < "2011-09-01")
Q4 = (cust.InvoiceDate >= "2011-09-01") & (cust.InvoiceDate < "2011-12-01")
cs, cq = cust[SUM], cust[Q4]
# H1: lift comes from order COUNT, not order SIZE
a = cs.groupby("InvoiceNo").rev.sum(); b = cq.groupby("InvoiceNo").rev.sum()
u = stats.mannwhitneyu(a, b, alternative="two-sided")
d = [np.median(RNG.choice(b.values, len(b))) - np.median(RNG.choice(a.values, len(a))) for _ in range(2000)]
inv_ratio = (len(b) / 3) / (len(a) / 3); rev_ratio = cq.rev.sum() / cs.rev.sum()
H.append(dict(id="H1", verdict="Supported",
    claim="The Sep-Nov surge comes from more orders, not bigger orders.",
    evidence=dict(rev_growth=float(rev_ratio - 1), order_count_growth=float(inv_ratio - 1),
                  share_of_growth_from_order_count=float(np.log(inv_ratio) / np.log(rev_ratio)),
                  median_order_summer=float(a.median()), median_order_q4=float(b.median()),
                  mean_order_summer=float(a.mean()), mean_order_q4=float(b.mean()),
                  mannwhitney_p=float(u.pvalue), median_diff_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  n_orders=[int(len(a)), int(len(b))])))
# H2: the lift is a Christmas-decoration effect (keyword test) -> expected to FAIL
ps = cs.groupby("StockCode").rev.sum() / 3; pq = cq.groupby("StockCode").rev.sum() / 3
up = pq.sub(ps, fill_value=0).sort_values(ascending=False)
names = cust.groupby("StockCode").Description.agg(lambda x: x.mode().iat[0]).str.strip()
XM = r"CHRISTMAS|XMAS|SANTA|ADVENT|REINDEER|SNOW|TINSEL|STOCKING|BAUBLE|NOEL|HOLLY|GINGERBREAD|PAPER CHAIN"
xcodes = set(names[names.str.upper().str.contains(XM)].index)
pos = up[up > 0]
H.append(dict(id="H2", verdict="Not supported",
    claim="The Sep-Nov surge is a Christmas-decoration effect.",
    evidence=dict(uplift_per_month=float(up.sum()),
                  share_from_christmas_keywords=float(up[up.index.isin(xcodes)].sum() / up.sum()),
                  christmas_share_of_revenue_summer=float(cs[cs.StockCode.isin(xcodes)].rev.sum() / cs.rev.sum()),
                  christmas_share_of_revenue_q4=float(cq[cq.StockCode.isin(xcodes)].rev.sum() / cq.rev.sum()),
                  products_growing=int((up > 0).sum()), products_shrinking=int((up < 0).sum()),
                  top50_share_of_growth=float(pos.head(50).sum() / pos.sum()),
                  top200_share_of_growth=float(pos.head(200).sum() / pos.sum()),
                  top_movers=[dict(item=str(names[k]), uplift=float(v)) for k, v in up.head(8).items()],
                  keyword_note="Keyword match is a lower bound on seasonal items; see README limitations.")))
# H3: lift is existing customers spending more vs customers who were absent in summer
both = set(cs.CustomerID) & set(cq.CustomerID)
H.append(dict(id="H3", verdict="Supported",
    claim="Customers who bought all year spend more in Q4, and customers who skipped summer add a large second source of growth.",
    evidence=dict(summer_customers=int(cs.CustomerID.nunique()), q4_customers=int(cq.CustomerID.nunique()),
                  returning_in_q4=int(len(both)), returning_share=float(len(both) / cs.CustomerID.nunique()),
                  same_customer_growth=float(cq[cq.CustomerID.isin(both)].rev.sum() / cs[cs.CustomerID.isin(both)].rev.sum() - 1),
                  q4_rev_share_from_customers_absent_in_summer=float(cq[~cq.CustomerID.isin(both)].rev.sum() / cq.rev.sum()))))
R["hypotheses"] = H

# ---------------------------------------------------------------- 4. CHURN MODEL
HORIZON = pd.Timedelta(days=100)
FEATS = ["recency", "frequency", "monetary", "avg_order", "tenure", "n_products", "inv_last_90", "rev_last_90",
         "mean_gap", "uk", "n_returns", "return_ratio", "aov_trend"]

def snapshot(T):
    T = pd.Timestamp(T)
    h = cust[cust.InvoiceDate < T]; rt = returns[returns.InvoiceDate < T]
    g = h.groupby("CustomerID")
    f = pd.DataFrame({
        "recency": (T - g.InvoiceDate.max()).dt.days,
        "frequency": g.InvoiceNo.nunique(), "monetary": g.rev.sum(),
        "tenure": (T - g.InvoiceDate.min()).dt.days, "n_products": g.StockCode.nunique(),
        "uk": g.Country.first().eq("United Kingdom").astype(int)})
    f["avg_order"] = f.monetary / f.frequency
    l90 = h[h.InvoiceDate >= T - pd.Timedelta(days=90)].groupby("CustomerID")
    f["inv_last_90"] = l90.InvoiceNo.nunique().reindex(f.index).fillna(0)
    f["rev_last_90"] = l90.rev.sum().reindex(f.index).fillna(0)
    dates = h.drop_duplicates("InvoiceNo").sort_values("InvoiceDate").groupby("CustomerID").InvoiceDate
    gap = dates.apply(lambda s: s.diff().dt.days.mean() if len(s) > 1 else np.nan)
    f["mean_gap"] = gap.reindex(f.index).fillna(f.tenure + 1)  # single purchase -> gap at least tenure
    f["n_returns"] = rt.groupby("CustomerID").InvoiceNo.nunique().reindex(f.index).fillna(0)
    f["return_ratio"] = (-rt.groupby("CustomerID").rev.sum().reindex(f.index).fillna(0) / f.monetary).clip(0, 5)
    half = T - pd.Timedelta(days=90)
    inv_v = h.groupby(["CustomerID", "InvoiceNo"]).agg(d=("InvoiceDate", "first"), v=("rev", "sum")).reset_index()
    recent = inv_v[inv_v.d >= half].groupby("CustomerID").v.mean(); older = inv_v[inv_v.d < half].groupby("CustomerID").v.mean()
    f["aov_trend"] = (recent / older).reindex(f.index).fillna(1.0).clip(0, 5)
    return f

def labelled(T):
    T = pd.Timestamp(T); f = snapshot(T)
    fut = cust[(cust.InvoiceDate >= T) & (cust.InvoiceDate < T + HORIZON)].CustomerID.unique()
    f["churn"] = (~f.index.isin(fut)).astype(int)
    f["T"] = T
    return f

TRAIN_T = ["2011-02-15", "2011-03-15", "2011-04-15", "2011-05-15"]   # label windows end by 2011-08-23
TEST_T = "2011-09-01"                                                 # label window 09-01..12-09 (hidden from training)
tr = pd.concat([labelled(t) for t in TRAIN_T]); te = labelled(TEST_T)
assert pd.Timestamp(TRAIN_T[-1]) + HORIZON <= pd.Timestamp(TEST_T), "label leakage"
Xtr, ytr, Xte, yte = tr[FEATS], tr.churn, te[FEATS], te.churn

models = {
    "Baseline: recency only": None,
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                                         l2_regularization=1.0, random_state=42),
}
res = {}; curves = {}; preds = {}
for name, m in models.items():
    if m is None: p = Xte.recency.values.astype(float)
    else: m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    preds[name] = p
    fpr, tpr, _ = roc_curve(yte, p)
    k = max(1, int(len(p) * .1)); order = np.argsort(-p)[:k]
    res[name] = dict(roc_auc=float(roc_auc_score(yte, p)), pr_auc=float(average_precision_score(yte, p)),
                     top_decile_precision=float(yte.values[order].mean()),
                     top_decile_lift=float(yte.values[order].mean() / yte.mean()))
    sel = np.linspace(0, len(fpr) - 1, 60).astype(int)
    curves[name] = [[float(fpr[i]), float(tpr[i])] for i in sel]
# bootstrap the held-out customers: AUC interval per model and paired gain over the recency baseline
yv = yte.values; B = 500; boots = {k: [] for k in preds}
for _ in range(B):
    ix = RNG.integers(0, len(yv), len(yv))
    if yv[ix].min() == yv[ix].max(): continue
    for k, pv in preds.items(): boots[k].append(roc_auc_score(yv[ix], pv[ix]))
for k in res:
    res[k]["auc_ci"] = [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
    if k != "Baseline: recency only":
        d = np.array(boots[k]) - np.array(boots["Baseline: recency only"])
        res[k]["gain_vs_baseline"] = float(d.mean()); res[k]["gain_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
d = np.array(boots["Logistic regression"]) - np.array(boots["Gradient boosting"])
R["lr_vs_gbm_gap_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
R["model"] = dict(horizon_days=100, train_cutoffs=TRAIN_T, test_cutoff=TEST_T,
                  train_rows=int(len(tr)), test_rows=int(len(te)),
                  train_churn_rate=float(ytr.mean()), test_churn_rate=float(yte.mean()),
                  results=res, roc=curves, features=FEATS)
# permutation importance on the held-out set (best model)
best = max((k for k in res if models[k] is not None), key=lambda k: res[k]["roc_auc"])
R["model"]["best"] = best
pi = permutation_importance(models[best], Xte, yte, scoring="roc_auc", n_repeats=15, random_state=42)
imp = sorted(zip(FEATS, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])
R["model"]["importance"] = [dict(feature=f, drop=float(m), sd=float(s)) for f, m, s in imp]
# calibration by decile on held-out
p_best = models[best].predict_proba(Xte)[:, 1]
cal = pd.DataFrame({"p": p_best, "y": yte.values}); cal["bin"] = pd.qcut(cal.p.rank(method="first"), 5, labels=False)
R["model"]["calibration"] = [dict(pred=float(g.p.mean()), actual=float(g.y.mean()), n=int(len(g))) for _, g in cal.groupby("bin")]

# score customers as of end of data (genuinely unseen future) with the model refit on all labelled snapshots
from sklearn.base import clone
full = pd.concat([tr, te]); final = clone(models[best]).fit(full[FEATS], full.churn)
now = snapshot(snap); now["churn_prob"] = final.predict_proba(now[FEATS])[:, 1]
now = now.join(rfm.segment)
now["value_at_risk"] = now.churn_prob * now.monetary
# "still active" = repeat buyer who ordered in the last 150 days. Customers silent longer are already lapsed
# (win-back, not retention) and sit outside the recency range the model was trained on.
act_now = now[(now.recency <= 150) & (now.frequency >= 2)]
now.sort_values("value_at_risk", ascending=False).reset_index().to_csv(f"{OUT}/at_risk_customers.csv", index=False)
topr = act_now.sort_values("value_at_risk", ascending=False).head(12).reset_index()
R["at_risk"] = [dict(customer=int(r.CustomerID), churn_prob=float(r.churn_prob), monetary=float(r.monetary),
                     recency=int(r.recency), frequency=int(r.frequency), segment=str(r.segment)) for r in topr.itertuples()]
R["at_risk_total_value"] = float(act_now.value_at_risk.sum()); R["scored_customers"] = int(len(now))
R["at_risk_pool"] = int(len(act_now)); R["train_recency_p95"] = float(tr.recency.quantile(.95))

with open(f"{OUT}/results.json", "w") as fh: json.dump(R, fh, indent=1, default=str)
print(json.dumps({k: R[k] for k in ("cleaning_log",)}, indent=1))
print("model", json.dumps(R["model"]["results"], indent=1)); print("churn rates", R["model"]["train_churn_rate"], R["model"]["test_churn_rate"])
print("importance", R["model"]["importance"][:5])
print("H", json.dumps(R["hypotheses"], indent=1)); print("naive", R["naive"], R["reversed_top"][:3]); print("anom", R["largest_invoices"][:3], R["price_outlier_lines"], R["price_outliers"][:3], R["n_high_value_invoices"], R["high_value_invoice_share"])
print("top", R["top_share"], R["gini"])
