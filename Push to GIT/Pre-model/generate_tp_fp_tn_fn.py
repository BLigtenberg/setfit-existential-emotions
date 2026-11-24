import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from setfit import SetFitModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(os.path.dirname(SCRIPT_DIR), "setfit-mpnet-final")
TEST_CSV    = os.path.join(SCRIPT_DIR, "test_cleaned.csv")

LABEL_COLUMNS = [
    "Death Anxiety", "Death Acceptance", "Loneliness", "Solitude",
    "Identity Confusion", "Identity Synthesis", "Freedom Paralysis",
    "Freedom Responsibility", "Meaninglessness", "Engagement"
]

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(TEST_CSV, sep=";")
texts       = test_df["Text"].tolist()
true_labels = test_df[LABEL_COLUMNS].values  # shape (n_samples, n_labels)
print(f"  {len(test_df)} samples loaded.")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at:\n  {MODEL_PATH}")

# Copy to space-free temp path
_tmp_dir   = tempfile.mkdtemp()
_tmp_model = os.path.join(_tmp_dir, "setfit-mpnet-final")
try:
    shutil.copytree(MODEL_PATH, _tmp_model)
    model = SetFitModel.from_pretrained(_tmp_model)
finally:
    shutil.rmtree(_tmp_dir, ignore_errors=True)
print("  Model loaded.")

# ── Run predictions ───────────────────────────────────────────────────────────
print("Running predictions...")
preds = model.predict(texts)
if hasattr(preds, "tolist"):
    preds = preds.tolist()
pred_labels = np.array(preds)   # shape (n_samples, n_labels)
print("  Done.")

# ── Classify each (sample, label) pair ───────────────────────────────────────
tp_rows, fp_rows, tn_rows, fn_rows = [], [], [], []

for i, (text, true_row, pred_row) in enumerate(zip(texts, true_labels, pred_labels)):
    for j, label in enumerate(LABEL_COLUMNS):
        t = int(true_row[j])
        p = int(pred_row[j])
        row = {"Text": text, "Label": label, "True": t, "Predicted": p}
        if   t == 1 and p == 1:
            tp_rows.append(row)
        elif t == 0 and p == 1:
            fp_rows.append(row)
        elif t == 0 and p == 0:
            tn_rows.append(row)
        elif t == 1 and p == 0:
            fn_rows.append(row)

# ── Save ──────────────────────────────────────────────────────────────────────
def save(rows, name):
    path = os.path.join(SCRIPT_DIR, name)
    pd.DataFrame(rows).to_csv(path, index=False, sep=";")
    print(f"  {name:<35} {len(rows):>4} rows")

print("\nSaving output files:")
save(tp_rows, "true_positives.csv")
save(fp_rows, "false_positives.csv")
save(tn_rows, "true_negatives.csv")
save(fn_rows, "false_negatives.csv")

print(f"\nSummary:")
print(f"  TP: {len(tp_rows)}  |  FP: {len(fp_rows)}  |  TN: {len(tn_rows)}  |  FN: {len(fn_rows)}")
print("Done!")
