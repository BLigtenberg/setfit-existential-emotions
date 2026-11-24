import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import functools
import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoTokenizer

print = functools.partial(print, flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SEP             = ";"
MAX_TOKENS      = 512
TOKENIZER_NAME  = "sentence-transformers/all-mpnet-base-v2"
N_SPLITS        = 5

LABEL_COLUMNS = [
    "Death Anxiety", "Death Acceptance", "Loneliness", "Solitude",
    "Identity Confusion", "Identity Synthesis", "Freedom Paralysis",
    "Freedom Responsibility", "Meaninglessness", "Engagement"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def path(filename):
    return os.path.join(SCRIPT_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_per_label_scores(y_true, y_pred, label_columns, title="Per-Label Scores"):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for i, label in enumerate(label_columns):
        col_true = np.array(y_true)[:, i]
        col_pred = np.array(y_pred)[:, i]
        p   = precision_score(col_true, col_pred, zero_division=0)
        r   = recall_score(col_true, col_pred, zero_division=0)
        f1  = f1_score(col_true, col_pred, zero_division=0)
        sup = int(col_true.sum())
        print(f"{label:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10}")
    print("-" * 65)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"{'Macro avg':<25} {''!s:>10} {''!s:>10} {macro_f1:>10.4f}")
    print(f"{'Micro avg':<25} {''!s:>10} {''!s:>10} {micro_f1:>10.4f}")
    print(f"{'='*60}\n")


def print_per_label_confusion_matrices(y_true, y_pred, label_columns):
    print(f"\n{'='*72}\n  Per-Label Confusion Matrices (binary: 1 vs 0)\n{'='*72}")
    print(f"{'Label':<25} {'TN':>8} {'FP':>8} {'FN':>8} {'TP':>8}")
    print("-" * 72)
    for i, label in enumerate(label_columns):
        col_true = np.array(y_true)[:, i]
        col_pred = np.array(y_pred)[:, i]
        cm = confusion_matrix(col_true, col_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
        print(f"{label:<25} {tn:>8} {fp:>8} {fn:>8} {tp:>8}")
    print(f"{'='*72}\n")


def make_model():
    """SetFit with all-mpnet-base-v2 and a balanced logistic head."""
    m = SetFitModel.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2",
        multi_target_strategy="one-vs-rest",
    )
    m.model_head = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    return m


def run_stratified_kfold_cv(full_df, label_columns, n_splits=5):
    """Multi-label stratified K-Fold CV using iterative stratification."""
    print(f"\n{'='*60}")
    print(f"  Multi-Label Stratified {n_splits}-Fold Cross-Validation")
    print(f"{'='*60}")

    texts     = full_df["Text"].tolist()
    label_mat = full_df[label_columns].values
    mskf      = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits    = list(mskf.split(texts, label_mat))

    fold_macro_f1s, fold_micro_f1s, fold_weighted_f1s, fold_label_f1s = [], [], [], []
    best_macro_f1, best_fold, best_model = -1, -1, None

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n  --- Fold {fold}/{n_splits} ---")

        fold_train_texts  = [texts[i] for i in train_idx]
        fold_val_texts    = [texts[i] for i in val_idx]
        fold_train_labels = label_mat[train_idx].tolist()
        fold_val_labels   = label_mat[val_idx].tolist()

        fold_train_ds = Dataset.from_dict({"text": fold_train_texts, "label": fold_train_labels})
        fold_val_ds   = Dataset.from_dict({"text": fold_val_texts,   "label": fold_val_labels})

        fold_model   = make_model()
        fold_trainer = Trainer(
            model=fold_model,
            args=TrainingArguments(batch_size=8, num_epochs=5, num_iterations=20),
            train_dataset=fold_train_ds,
            eval_dataset=fold_val_ds,
        )
        fold_trainer.train()

        preds = fold_model.predict(fold_val_texts)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        macro_f1    = f1_score(fold_val_labels, preds, average="macro",    zero_division=0)
        micro_f1    = f1_score(fold_val_labels, preds, average="micro",    zero_division=0)
        weighted_f1 = f1_score(fold_val_labels, preds, average="weighted", zero_division=0)
        fold_macro_f1s.append(macro_f1)
        fold_micro_f1s.append(micro_f1)
        fold_weighted_f1s.append(weighted_f1)
        fold_label_f1s.append([
            f1_score(np.array(fold_val_labels)[:, i], np.array(preds)[:, i], zero_division=0)
            for i in range(len(label_columns))
        ])

        print(f"  Fold {fold} — Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        print_per_label_scores(fold_val_labels, preds, label_columns, title=f"Fold {fold} Per-Label Scores")
        print_per_label_confusion_matrices(fold_val_labels, preds, label_columns)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_fold     = fold
            best_model    = fold_model
            print(f"  ★ New best fold: Fold {fold} (Macro F1 = {macro_f1:.4f})")

    fold_label_f1s = np.array(fold_label_f1s)

    # Summary tables
    print(f"\n{'='*50}")
    print(f"  TABLE 1 — Cross-Validation F1 Summary ({len(fold_macro_f1s)} Folds)")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'Mean':>8} {'SD':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Macro F1':<20} {np.mean(fold_macro_f1s):>8.4f} {np.std(fold_macro_f1s):>8.4f}")
    print(f"  {'Micro F1':<20} {np.mean(fold_micro_f1s):>8.4f} {np.std(fold_micro_f1s):>8.4f}")
    print(f"  {'Weighted F1':<20} {np.mean(fold_weighted_f1s):>8.4f} {np.std(fold_weighted_f1s):>8.4f}")
    print(f"{'='*50}\n")

    print(f"{'='*62}")
    print(f"  TABLE 2 — Average F1 per Label Across {len(fold_macro_f1s)} Folds")
    print(f"{'='*62}")
    print(f"  {'Label':<25} {'Mean F1':>10} {'SD':>10}")
    print(f"  {'-'*47}")
    for i, label in enumerate(label_columns):
        col = fold_label_f1s[:, i]
        print(f"  {label:<25} {np.mean(col):>10.4f} {np.std(col):>10.4f}")
    print(f"  {'-'*47}")
    print(f"  {'Macro avg':<25} {np.mean(fold_label_f1s):>10.4f} {np.std(np.mean(fold_label_f1s, axis=1)):>10.4f}")
    print(f"{'='*62}\n")

    # Save CV results
    fold_rows = [
        {"fold": i + 1, "macro_f1": fold_macro_f1s[i], "micro_f1": fold_micro_f1s[i],
         "weighted_f1": fold_weighted_f1s[i],
         **{label: fold_label_f1s[i, j] for j, label in enumerate(label_columns)}}
        for i in range(len(fold_macro_f1s))
    ]
    fold_rows.append({"fold": "Mean", "macro_f1": np.mean(fold_macro_f1s),
                      "micro_f1": np.mean(fold_micro_f1s), "weighted_f1": np.mean(fold_weighted_f1s),
                      **{label: np.mean(fold_label_f1s[:, j]) for j, label in enumerate(label_columns)}})
    fold_rows.append({"fold": "SD", "macro_f1": np.std(fold_macro_f1s),
                      "micro_f1": np.std(fold_micro_f1s), "weighted_f1": np.std(fold_weighted_f1s),
                      **{label: np.std(fold_label_f1s[:, j]) for j, label in enumerate(label_columns)}})
    pd.DataFrame(fold_rows).to_csv(path("cv_results_mpnet.csv"), index=False, sep=SEP)
    print("  CV results saved to 'cv_results_mpnet.csv'\n")

    print(f"  Best fold overall: Fold {best_fold} (Macro F1 = {best_macro_f1:.4f})")
    return fold_macro_f1s, fold_micro_f1s, best_model, best_fold, best_macro_f1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Remove all-zero (no-label) rows
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 1 — Removing all-zero (no-label) rows")
print("="*65)

SPLITS_ZERO = [
    ("training_rebalanced.csv",   "training_filtered.csv"),
    ("validation_rebalanced.csv", "validation_filtered.csv"),
    ("test_rebalanced.csv",       "test_filtered.csv"),
]

for input_file, output_file in SPLITS_ZERO:
    df = pd.read_csv(path(input_file), sep=SEP)
    n_before = len(df)
    mask = df[LABEL_COLUMNS].sum(axis=1) > 0
    df_filtered = df[mask].reset_index(drop=True)
    n_after   = len(df_filtered)
    n_removed = n_before - n_after
    print(f"\n  {input_file}")
    print(f"    Before : {n_before}  |  Removed: {n_removed} ({100*n_removed/n_before:.1f}%)  |  After: {n_after}")
    df_filtered.to_csv(path(output_file), sep=SEP, index=False)
    print(f"    Saved -> {output_file}")

print(f"\n{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Remove >512-token samples; merge train + val
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STEP 2 — Removing >512-token samples & merging train+val")
print("="*65)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def count_tokens(text):
    return len(tokenizer.encode(text, truncation=False))

train_df = pd.read_csv(path("training_filtered.csv"),   sep=SEP)
val_df   = pd.read_csv(path("validation_filtered.csv"), sep=SEP)
test_df  = pd.read_csv(path("test_filtered.csv"),       sep=SEP)

combined_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"\nCombined train+val before token filter: {len(combined_df)} samples")
print("Counting tokens (this may take a moment)...")

combined_df["_tokens"] = combined_df["Text"].apply(count_tokens)
test_df["_tokens"]     = test_df["Text"].apply(count_tokens)

training_cleaned = combined_df[combined_df["_tokens"] <= MAX_TOKENS].drop(columns=["_tokens"]).reset_index(drop=True)
test_cleaned     = test_df[test_df["_tokens"] <= MAX_TOKENS].drop(columns=["_tokens"]).reset_index(drop=True)

print(f"  Removed from training : {len(combined_df) - len(training_cleaned)} samples (>{MAX_TOKENS} tokens)")
print(f"  Removed from test     : {len(test_df)     - len(test_cleaned)} samples (>{MAX_TOKENS} tokens)")

training_cleaned.to_csv(path("training_cleaned.csv"), index=False, sep=SEP)
test_cleaned.to_csv(path("test_cleaned.csv"),         index=False, sep=SEP)
print(f"\nSaved: training_cleaned.csv ({len(training_cleaned)} samples)")
print(f"Saved: test_cleaned.csv     ({len(test_cleaned)} samples)")
print(f"\n{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build HuggingFace Datasets
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STEP 3 — Preparing Datasets")
print("="*65)

full_dataset = Dataset.from_dict({
    "text":  training_cleaned["Text"].tolist(),
    "label": training_cleaned[LABEL_COLUMNS].values.tolist()
})
test_dataset = Dataset.from_dict({
    "text":  test_cleaned["Text"].tolist(),
    "label": test_cleaned[LABEL_COLUMNS].values.tolist()
})
print(f"\nTraining set : {len(full_dataset)} examples")
print(f"Test set     : {len(test_dataset)} examples")
print(f"\n{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — 5-Fold Stratified Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STEP 4 — Stratified 5-Fold Cross-Validation")
print("="*65)

_, _, best_fold_model, best_fold_num, best_fold_f1 = run_stratified_kfold_cv(
    training_cleaned, LABEL_COLUMNS, n_splits=N_SPLITS
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train final model on ALL of training_cleaned
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STEP 5 — Training Final Model on Full Training Set")
print("="*65)

final_model   = make_model()
final_trainer = Trainer(
    model=final_model,
    args=TrainingArguments(batch_size=16, num_epochs=5, num_iterations=20),
    train_dataset=full_dataset,
)
final_trainer.train()
print("\nFinal model training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate on test set (once)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 6 — Evaluating Final Model on Test Set")
print("="*65)

test_texts  = test_cleaned["Text"].tolist()
test_labels = test_cleaned[LABEL_COLUMNS].values.tolist()
preds       = final_model.predict(test_texts)
if hasattr(preds, "tolist"):
    preds = preds.tolist()

print_per_label_scores(test_labels, preds, LABEL_COLUMNS, title="Test Set: Per-Label F-Scores")
print_per_label_confusion_matrices(test_labels, preds, LABEL_COLUMNS)

test_rows = [
    {"label": label,
     "precision": precision_score(np.array(test_labels)[:, i], np.array(preds)[:, i], zero_division=0),
     "recall":    recall_score(np.array(test_labels)[:, i], np.array(preds)[:, i], zero_division=0),
     "f1":        f1_score(np.array(test_labels)[:, i], np.array(preds)[:, i], zero_division=0),
     "support":   int(np.array(test_labels)[:, i].sum())}
    for i, label in enumerate(LABEL_COLUMNS)
]
test_rows += [
    {"label": "Macro avg",    "f1": f1_score(test_labels, preds, average="macro",    zero_division=0)},
    {"label": "Micro avg",    "f1": f1_score(test_labels, preds, average="micro",    zero_division=0)},
    {"label": "Weighted avg", "f1": f1_score(test_labels, preds, average="weighted", zero_division=0)},
]
pd.DataFrame(test_rows).to_csv(path("test_results_mpnet.csv"), index=False, sep=SEP)
print("  Test results saved to 'test_results_mpnet.csv'")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 7 — Saving Models")
print("="*65)

final_model.save_pretrained(path("setfit-mpnet-final"))
print("Final model saved to 'setfit-mpnet-final'")

best_fold_save_path = path(f"setfit-mpnet-best-fold{best_fold_num}")
best_fold_model.save_pretrained(best_fold_save_path)
print(f"Best CV fold model (Fold {best_fold_num}, Macro F1 = {best_fold_f1:.4f}) saved to 'setfit-mpnet-best-fold{best_fold_num}'")

print("\n" + "="*65)
print("  Pipeline complete.")
print("="*65)