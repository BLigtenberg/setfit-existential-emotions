import random
import pandas as pd
from pathlib import Path

# -- Configuration -------------------------------------------------------------

BASE      = Path(__file__).parent          # Post-model scripts/
EPUB_BASE = BASE.parent / "dostoevsky epub"

BOOK_FILES = {
    "Crime and Punishment":       EPUB_BASE / "Crime and punishment"        / "crime_predictions_epub.csv",
    "Notes from the Underground": EPUB_BASE / "Notes from the Underground"  / "underground_predictions_epub.csv",
    "The Brothers Karamazov":     EPUB_BASE / "The Brothers Karamazov"      / "karamazov_predictions_epub.csv",
    "The Idiot":                  EPUB_BASE / "The Idiot"                   / "idiot_predictions_epub.csv",
    "The Possessed":              EPUB_BASE / "The Possessed"               / "possessed_predictions_epub.csv",
}

EMOTION_COLS = [
    "Death Anxiety",
    "Death Acceptance",
    "Loneliness",
    "Solitude",
    "Identity Confusion",
    "Identity Synthesis",
    "Freedom Paralysis",
    "Freedom Responsibility",
    "Meaninglessness",
    "Engagement",
]

ROWS_PER_BOOK    = 4    # 5 books x 4 rows = 20
ROWS_PER_EMOTION = 2    # target: each emotion appears at least this many times
RANDOM_SEED      = 42
OUTPUT_FILE      = BASE / "balanced_20_sample.csv"

rng = random.Random(RANDOM_SEED)

# -- Load & tag ----------------------------------------------------------------

book_dfs: dict = {}
for book_name, csv_path in BOOK_FILES.items():
    df = pd.read_csv(csv_path, sep=";")
    df["book"] = book_name
    # Keep only rows that have at least one positive emotion prediction
    df_pos = df[df[EMOTION_COLS].sum(axis=1) > 0].copy()
    book_dfs[book_name] = df_pos
    print(f"  Loaded {len(df):>6} rows from '{book_name}' "
          f"({len(df_pos)} with >=1 positive emotion)")

# -- Per-book greedy selection -------------------------------------------------
# Pick 4 rows per book that maximise emotion coverage within that book.

selected_rows = []

for book_name, df in book_dfs.items():
    pool = df.copy()
    chosen = []
    covered_emotions = set()

    for _ in range(ROWS_PER_BOOK):
        # Score each candidate by how many NEW emotions it adds
        best_score = -1
        best_idx   = None

        candidates = pool.index.tolist()
        rng.shuffle(candidates)          # random tie-breaking

        for idx in candidates:
            row_emotions = {e for e in EMOTION_COLS if pool.loc[idx, e] == 1}
            score = len(row_emotions - covered_emotions)
            if score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx is None:
            break

        chosen_row = pool.loc[best_idx]
        chosen.append(chosen_row)
        covered_emotions.update({e for e in EMOTION_COLS if chosen_row[e] == 1})
        pool = pool.drop(index=best_idx)

    # If we didn't fill all slots (unlikely), top up randomly
    while len(chosen) < ROWS_PER_BOOK and len(pool) > 0:
        idx = rng.choice(pool.index.tolist())
        chosen.append(pool.loc[idx])
        pool = pool.drop(index=idx)

    selected_rows.extend(chosen)
    covered_str = ", ".join(e for e in EMOTION_COLS if e in covered_emotions)
    print(f"  '{book_name}': selected {len(chosen)} rows -- emotions: {covered_str}")

# -- Build working dataframe ---------------------------------------------------

result = pd.DataFrame(selected_rows).reset_index(drop=True)
print(f"\nInitial selection: {len(result)} rows")

# -- Verify & fix emotion coverage --------------------------------------------
# Target: every emotion appears at least ROWS_PER_EMOTION times.

print("\nEmotion coverage after per-book pass:")
for emotion in EMOTION_COLS:
    count = int(result[emotion].sum())
    status = "OK" if count >= ROWS_PER_EMOTION else "UNDER"
    print(f"  {emotion:<26}: {count} {status}")

# For emotions that are under-represented, try to swap a row from the
# current selection with a row from the global pool that has the missing emotion.

global_pool = pd.concat(book_dfs.values(), ignore_index=False)

def emotion_counts(df):
    return {e: int(df[e].sum()) for e in EMOTION_COLS}

def under_represented(df):
    return [e for e, c in emotion_counts(df).items() if c < ROWS_PER_EMOTION]

hopeless: set = set()  # emotions with zero global predictions — permanently skip
max_iterations = 200
for iteration in range(max_iterations):
    under = [e for e in under_represented(result) if e not in hopeless]
    if not under:
        break

    emotion_to_fix = rng.choice(under)

    # Rows in result that already have this emotion (can't be swapped out)
    rows_with_emotion = result[result[emotion_to_fix] == 1].index.tolist()

    # Candidate rows in global_pool with the missing emotion (not already selected)
    candidates = global_pool[
        (global_pool[emotion_to_fix] == 1) &
        (~global_pool.index.isin(result.index))
    ]

    if candidates.empty:
        print(f"  WARNING: No candidates exist for '{emotion_to_fix}' anywhere — marking hopeless.")
        hopeless.add(emotion_to_fix)
        continue

    # Find rows in result that we could swap out:
    # - Must not be one of the rows already covering the emotion we want to fix
    swappable = result[~result.index.isin(rows_with_emotion)]

    if swappable.empty:
        continue

    # Prefer swapping within the same book to keep book balance
    books_with_candidates = candidates["book"].unique().tolist()
    swappable_preferred = swappable[swappable["book"].isin(books_with_candidates)]
    swap_source = swappable_preferred if not swappable_preferred.empty else swappable

    swap_out_idx = rng.choice(swap_source.index.tolist())
    swap_out_book = result.loc[swap_out_idx, "book"]

    same_book_candidates = candidates[candidates["book"] == swap_out_book]
    final_candidates = same_book_candidates if not same_book_candidates.empty else candidates

    swap_in_row = final_candidates.sample(1, random_state=rng.randint(0, 99999)).iloc[0]

    # Perform the swap
    result = result.drop(index=swap_out_idx).reset_index(drop=True)
    result = pd.concat([result, swap_in_row.to_frame().T], ignore_index=True)

print(f"\nFixing loop ended after {iteration + 1} iteration(s).")

# -- Final report --------------------------------------------------------------

print(f"\nFinal sample size : {len(result)} rows")

print("\nEmotion counts in final sample:")
for emotion in EMOTION_COLS:
    count = int(result[emotion].sum())
    status = "OK" if count >= ROWS_PER_EMOTION else "UNDER"
    print(f"  {emotion:<26}: {count} {status}")

print("\nBook distribution:")
print(result["book"].value_counts().to_string())

# -- Reorder columns and shuffle -----------------------------------------------

result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Put 'book' right after 'text'
cols = [c for c in result.columns if c != "book"]
text_pos = cols.index("text") + 1 if "text" in cols else len(cols)
cols.insert(text_pos, "book")
result = result[cols]

# -- Save ----------------------------------------------------------------------

result.to_csv(OUTPUT_FILE, sep=";", index=False)
print(f"\nSaved to: {OUTPUT_FILE}")