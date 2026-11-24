import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ── Config ─────────────────────────────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RATER1_PATH = os.path.join(SCRIPT_DIR, "Bart.csv")
RATER2_PATH = os.path.join(SCRIPT_DIR, "Rebecca.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "IRR_PABAK_results.csv")
SEP         = ";"

LABELS = [
    "death 1",      "death 2",
    "loneliness 1", "loneliness 2",
    "identity 1",   "identity 2",
    "freedom 1",    "freedom 2",
    "meaning 1",    "meaning 2",
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading rater files...")
rater1 = pd.read_csv(RATER1_PATH, sep=SEP)
rater2 = pd.read_csv(RATER2_PATH, sep=SEP)

# Normalise column names: strip whitespace, lowercase
rater1.columns = rater1.columns.str.strip().str.lower()
rater2.columns = rater2.columns.str.strip().str.lower()

print(f"  Rater 1 ({os.path.basename(RATER1_PATH)}): {len(rater1)} rows")
print(f"  Rater 2 ({os.path.basename(RATER2_PATH)}): {len(rater2)} rows")


# ── PABAK helper ───────────────────────────────────────────────────────────────

def pabak(r1: pd.Series, r2: pd.Series) -> float:
    """
    Prevalence-Adjusted and Bias-Adjusted Kappa.

    PABAK = 2 * Po - 1
    where Po = proportion of items on which raters agree.
    """
    po = np.mean(r1.values == r2.values)
    return 2 * po - 1


def positive_agreement(r1: pd.Series, r2: pd.Series) -> float:
    """
    Positive Agreement (PA): proportion of positive cases both raters agree on.
    PA = 2A / (2A + B + C)
    where A = both 1, B = rater1=1 & rater2=0, C = rater1=0 & rater2=1.
    Returns NaN when no positive ratings exist.
    """
    a = np.sum((r1 == 1) & (r2 == 1))
    b = np.sum((r1 == 1) & (r2 == 0))
    c = np.sum((r1 == 0) & (r2 == 1))
    denom = 2 * a + b + c
    return (2 * a) / denom if denom > 0 else np.nan


def negative_agreement(r1: pd.Series, r2: pd.Series) -> float:
    """
    Negative Agreement (NA): proportion of negative cases both raters agree on.
    NA = 2D / (2D + B + C)
    where D = both 0, B = rater1=1 & rater2=0, C = rater1=0 & rater2=1.
    Returns NaN when no negative ratings exist.
    """
    d = np.sum((r1 == 0) & (r2 == 0))
    b = np.sum((r1 == 1) & (r2 == 0))
    c = np.sum((r1 == 0) & (r2 == 1))
    denom = 2 * d + b + c
    return (2 * d) / denom if denom > 0 else np.nan


# ── Compute metrics per label ──────────────────────────────────────────────────

results = []

print(f"\n{'='*74}")
print(f"  {'Label':<16} {'Po':>7} {'Kappa':>8} {'PABAK':>8} {'PA':>8} {'NA':>8}")
print(f"  {'-'*66}")

for label in LABELS:
    r1 = rater1[label]
    r2 = rater2[label]

    po      = float(np.mean(r1.values == r2.values))
    pbk     = pabak(r1, r2)
    pa      = positive_agreement(r1, r2)
    na      = negative_agreement(r1, r2)

    # Cohen's Kappa — may produce NaN/warning when only one class present
    try:
        kappa = cohen_kappa_score(r1, r2)
    except Exception:
        kappa = np.nan

    results.append({
        "label":               label,
        "percent_agreement":   round(po,    4),
        "cohen_kappa":         round(kappa, 4) if not np.isnan(kappa) else np.nan,
        "PABAK":               round(pbk,   4),
        "positive_agreement":  round(pa,    4) if not np.isnan(pa)    else np.nan,
        "negative_agreement":  round(na,    4) if not np.isnan(na)    else np.nan,
    })

    kappa_str = f"{kappa:>8.4f}" if not np.isnan(kappa) else f"{'NaN':>8}"
    pa_str    = f"{pa:>8.4f}"   if not np.isnan(pa)    else f"{'NaN':>8}"
    na_str    = f"{na:>8.4f}"   if not np.isnan(na)    else f"{'NaN':>8}"
    print(f"  {label:<16} {po:>7.4f} {kappa_str} {pbk:>8.4f} {pa_str} {na_str}")

print(f"{'='*74}\n")

# ── Summary row (means across labels) ─────────────────────────────────────────

results_df = pd.DataFrame(results)

numeric_cols = ["percent_agreement", "cohen_kappa", "PABAK",
                "positive_agreement", "negative_agreement"]
means = results_df[numeric_cols].mean()

print("Mean across all labels:")
print(f"  Percent agreement : {means['percent_agreement']:.4f}")
print(f"  Cohen's Kappa     : {means['cohen_kappa']:.4f}")
print(f"  PABAK             : {means['PABAK']:.4f}")
print(f"  Positive agreement: {means['positive_agreement']:.4f}")
print(f"  Negative agreement: {means['negative_agreement']:.4f}")

# ── Save results ───────────────────────────────────────────────────────────────

results_df.to_csv(OUTPUT_PATH, index=False, sep=SEP)
print(f"\nResults saved to: {OUTPUT_PATH}")