# Lift & Drift: Unpacking Seasonal Growth and Predicting Customer Churn
Algothon'26 · **ALG-DATA-01 · The Mystery Dataset** (Data Science)

A UK gift-ware retailer's transaction log (1 Dec 2010 to 9 Dec 2011, 541,909 rows). The project cleans it, finds what drives the autumn revenue surge, tests three hypotheses, and predicts which customers will stop ordering.

## The finding
Sep-Nov revenue from identified customers is **+66%** on Jun-Aug. About **90% of that growth is order count** (+58%), not order size (+5% mean, +2% median). Christmas-keyword items explain only **23%** of the product-level lift, and about 2,100 products grew. The extra revenue comes mainly from customers who were already ordering, with a smaller share from customers returning after a gap or new ones.

A second result comes from cleaning: 2,818 lines (£426k) were orders placed and then fully cancelled. Left in, they make customers 16446 and 12346 look like top-10 spenders.

## Run it
```bash
pip install pandas numpy scipy scikit-learn openpyxl
python pipeline.py Online_Retail.xlsx   # ~1 min -> outputs/
python tests.py                         # 7 checks
python build_dashboard.py               # -> dashboard.html (self-contained)
```
Outputs: `outputs/clean_sales.csv`, `outputs/results.json`, `outputs/at_risk_customers.csv`.

## How the brief is covered
| Must have | Where |
|---|---|
| Data cleaning | `pipeline.py` section 1. Six counted rules, shown on the *Data cleaning* tab |
| Variable understanding | Customer types, RFM, cohorts, country tables on *Overview* and *Customers* |
| Relationship analysis | Revenue = orders x order size decomposition; customer-type split; product-level lift |
| Anomaly detection | Matched order/cancel reversals, robust-z on order value, price vs product median |
| Visualizations | `dashboard.html`, interactive, light and dark, works on a phone |
| Hypothesis | H1 supported, **H2 not supported**, H3 supported. Mann-Whitney, bootstrap CI |
| Analytical / predictive result | 100-day churn model, time-based validation, bootstrap CIs |

## Architecture
```
Online_Retail.xlsx -> Clean (6 counted rules) -> clean_sales.csv
   -> Explore (customer types, cohorts, RFM, anomalies)
   -> Hypothesis tests (Mann-Whitney, bootstrap)
   -> Churn model (snapshot features -> train on 4 earlier snapshots, test on a later one)
   -> results.json -> dashboard.html      and      at_risk_customers.csv
```

## Key decisions
- **Matched reversals are removed from sales.** A cancellation pairs with an earlier order of the same customer, product and exact quantity. Unmatched cancellations stay as returns.
- **Guest orders** (no CustomerID, 15% of revenue) stay in revenue totals and leave customer-level work.
- **Time-based validation.** Four training snapshots (15 Feb, 15 Mar, 15 Apr, 15 May 2011) and a test snapshot at 1 Sep 2011. Each label looks 100 days ahead. The last training window ends before the test snapshot, enforced by an assertion.
- **Models:** recency-only baseline, logistic regression, gradient boosting (scikit-learn).

## Results (held-out snapshot, 3,310 customers, 41% churn)
| Model | ROC AUC (95% CI) | Top-10% lift |
|---|---|---|
| Recency only | 0.685 (0.67-0.70) | 1.56x |
| Logistic regression | 0.743 (0.73-0.76) | 1.59x |
| Gradient boosting | 0.735 (0.72-0.75) | 1.63x |

Both models beat the baseline (paired gain intervals exclude zero). Logistic and boosting are tied.

## Testing and edge cases
`tests.py`: cleaning log reconciles to the raw row count; no duplicates, negative quantities, cancellations or non-product codes remain; revenue totals match; no label leakage; probabilities in [0, 1]; segments partition customers; series lengths agree.
Handled: missing customer IDs, duplicates, fee/postage codes, same-day reversals, a partial final month, single-order customers, customers with no returns.

## Known limitations and future work
- One year of data, so seasonality cannot be learned. The test period contains the Q4 peak and its churn rate differs slightly from training.
- Exact-quantity matching misses partial cancellations.
- The Christmas keyword list is a lower bound on seasonal stock.
- Model choice saw the test snapshot. Three candidates and a tie at the top limit the effect, but it is not a clean final exam. Next step: choose on a validation snapshot, then touch the test snapshot once.
- Predicted probabilities are miscalibrated at the extremes (too low for safe customers, too high for risky ones). Use the score to rank.
- The at-risk list covers repeat buyers active in the last 150 days. Longer-silent customers are outside the training range.
- Country figures rest on very few accounts (Netherlands 9, EIRE 3).
- Findings are descriptive, not causal.

## Disclosure
- Data: supplied `Online_Retail.xlsx` (matches the public UCI Online Retail dataset). No external APIs.
- Libraries: pandas, NumPy, SciPy, scikit-learn.
- AI-assisted: the code, analysis and dashboard were drafted with Claude (Anthropic). All figures come from `pipeline.py` output and can be regenerated.
