import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import f1_score, precision_score, recall_score

# -- Paths --------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # "Final model/"
PRE_DIR     = os.path.join(PARENT_DIR, "Pre-model scripts")

INPUT_PATH  = os.path.join(PRE_DIR, "test_cleaned.csv")
MODEL_PATH  = os.path.join(PARENT_DIR, "setfit-mpnet-final")  # sits in "Final model/"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "test_predictions_mpnet.csv")

# -- Label columns (same order as training) -----------------------------------
LABEL_COLUMNS = [
    "Death Anxiety", "Death Acceptance", "Loneliness", "Solitude",
    "Identity Confusion", "Identity Synthesis", "Freedom Paralysis",
    "Freedom Responsibility", "Meaninglessness", "Engagement"
]

# -- Step 1: Load data --------------------------------------------------------
print("--- Step 1: Loading test data ---")
test_df = pd.read_csv(INPUT_PATH, sep=";")
print(f"  Loaded {len(test_df)} rows from '{INPUT_PATH}'")

texts       = test_df["Text"].tolist()
true_labels = test_df[LABEL_COLUMNS].values.tolist()

# -- Step 2: Load model -------------------------------------------------------
# huggingface_hub validates the path string as a repo-ID and rejects Windows
# paths that contain spaces. Workaround: copy the model folder to a temp
# directory whose path is guaranteed to be space-free, load from there, then
# delete the temp copy.
print("\n--- Step 2: Loading model ---")
print(f"  Model path: {MODEL_PATH}")
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Model directory not found:\n  {MODEL_PATH}\n"
        "Please make sure 'setfit-mpnet-final' exists in the 'Final model' folder."
    )
_tmp_dir = tempfile.mkdtemp()          # e.g. C:\Users\bartj\AppData\Local\Temp\tmpXXXXXX
_tmp_model = os.path.join(_tmp_dir, "setfit-mpnet-final")
try:
    shutil.copytree(MODEL_PATH, _tmp_model)
    model = SetFitModel.from_pretrained(_tmp_model)
finally:
    shutil.rmtree(_tmp_dir, ignore_errors=True)
print("  Model loaded successfully.")

# -- Step 3: Run inference ----------------------------------------------------
print("\n--- Step 3: Running inference ---")
preds = model.predict(texts)
if hasattr(preds, "tolist"):
    preds = preds.tolist()
print(f"  Predictions generated for {len(preds)} samples.")

# -- Step 4: Build output DataFrame -------------------------------------------
print("\n--- Step 4: Building output CSV ---")
out_df = test_df[["Text"]].copy()

# Ground-truth columns  (suffix: _true)
for col in LABEL_COLUMNS:
    out_df[f"{col}_true"] = test_df[col].values

# Prediction columns  (suffix: _pred)
preds_arr = np.array(preds)
for i, col in enumerate(LABEL_COLUMNS):
    out_df[f"{col}_pred"] = preds_arr[:, i].astype(int)

# -- Step 5: Per-label metrics ------------------------------------------------
print("\n--- Step 5: Per-label evaluation ---")
true_arr = np.array(true_labels)

print(f"\n{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 65)
for i, label in enumerate(LABEL_COLUMNS):
    p  = precision_score(true_arr[:, i], preds_arr[:, i], zero_division=0)
    r  = recall_score   (true_arr[:, i], preds_arr[:, i], zero_division=0)
    f1 = f1_score       (true_arr[:, i], preds_arr[:, i], zero_division=0)
    sup = int(true_arr[:, i].sum())
    print(f"{label:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10}")
print("-" * 65)
macro_f1    = f1_score(true_labels, preds, average="macro",    zero_division=0)
micro_f1    = f1_score(true_labels, preds, average="micro",    zero_division=0)
weighted_f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)
print(f"{'Macro F1':<25} {macro_f1:>10.4f}")
print(f"{'Micro F1':<25} {micro_f1:>10.4f}")
print(f"{'Weighted F1':<25} {weighted_f1:>10.4f}")

# -- Step 6: Save output ------------------------------------------------------
print(f"\n--- Step 6: Saving output ---")
out_df.to_csv(OUTPUT_PATH, index=False, sep=";")
print(f"  Predictions saved to '{OUTPUT_PATH}'")
print("\nDone.")