import os
import re
import shutil
import tempfile
import pandas as pd
from pypdf import PdfReader
from transformers import AutoTokenizer
from setfit import SetFitModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Final model/

PDF_PATH    = os.path.join(SCRIPT_DIR, "The Idiot (pages 7-612).pdf")
MODEL_PATH  = os.path.join(PARENT_DIR, "setfit-mpnet-final")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "the_idiot_predictions.csv")

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_TOKENS    = 512
OVERLAP_TOKENS = 10   # small overlap so sentences at chunk edges aren't cut off

LABEL_COLUMNS = [
    "Death Anxiety", "Death Acceptance", "Loneliness", "Solitude",
    "Identity Confusion", "Identity Synthesis", "Freedom Paralysis",
    "Freedom Responsibility", "Meaninglessness", "Engagement"
]

# ── Text cleaning helper ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Fix common PDF-extraction artefacts before chunking."""
    # 1. Fix hyphenated line-breaks: "some- thing" / "some -thing" -> "something"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    # 2. Normalise curly/smart quotes to straight ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    # 3. Normalise em/en dashes to spaced hyphen
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # 4. Normalise ellipsis character
    text = text.replace("\u2026", "...")
    # 5. Remove trademark symbol and replace bullet with asterisk
    text = text.replace("\u2122", "").replace("\u2022", "*")
    # 6. Remove replacement characters (U+FFFD) left by bad encoding
    text = text.replace("\ufffd", "")
    # 7. Collapse multiple spaces / stray newlines into a single space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


if __name__ == "__main__":
    # ── Step 1: Extract text from PDF ─────────────────────────────────────────────
    print("--- Step 1: Extracting text from PDF ---")
    reader = PdfReader(PDF_PATH)
    page_texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)                    # ← clean immediately after extraction
        page_texts.append((i + 1, text))           # (page_number, text)
    print(f"  Extracted text from {len(page_texts)} pages.")

    # ── Step 2: Chunk into 512-token windows ──────────────────────────────────────
    print("\n--- Step 2: Tokenising and chunking into 512-token windows ---")

    # Use the same tokeniser that mpnet uses
    TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    chunks = []   # list of dicts: {chunk_id, page_start, page_end, text}

    # Build a flat list of (page_num, word) so we can track page provenance
    words_with_pages = []
    for page_num, text in page_texts:
        for word in text.split():
            words_with_pages.append((page_num, word))

    # Slide a window over the words, tokenising each candidate chunk
    i = 0
    chunk_id = 1
    while i < len(words_with_pages):
        # Expand window until we hit MAX_TOKENS or end of document
        j = i
        current_tokens = 0
        while j < len(words_with_pages):
            word = words_with_pages[j][1]
            n_toks = len(tokenizer.encode(word, add_special_tokens=False))
            if current_tokens + n_toks > MAX_TOKENS:
                break
            current_tokens += n_toks
            j += 1

        if j == i:          # single word already exceeds limit — include it anyway
            j = i + 1

        chunk_words = [w for _, w in words_with_pages[i:j]]
        chunk_pages = [p for p, _ in words_with_pages[i:j]]
        chunk_text  = " ".join(chunk_words)

        chunks.append({
            "chunk_id":   chunk_id,
            "page_start": chunk_pages[0],
            "page_end":   chunk_pages[-1],
            "text":       chunk_text,
        })

        chunk_id += 1
        # Move forward, stepping back by OVERLAP_TOKENS words for continuity
        step = max(1, j - i - OVERLAP_TOKENS)
        i += step

    print(f"  Created {len(chunks)} chunks "
          f"(max {MAX_TOKENS} tokens each, {OVERLAP_TOKENS}-token overlap).")

    # ── Step 3: Load model ─────────────────────────────────────────────────────────
    print("\n--- Step 3: Loading setfit-mpnet-final model ---")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found:\n  {MODEL_PATH}\n"
            "Make sure 'setfit-mpnet-final' exists in the 'Final model' folder."
        )

    # Copy to a space-free temp path (huggingface_hub rejects paths with spaces)
    _tmp_dir   = tempfile.mkdtemp()
    _tmp_model = os.path.join(_tmp_dir, "setfit-mpnet-final")
    try:
        shutil.copytree(MODEL_PATH, _tmp_model)
        model = SetFitModel.from_pretrained(_tmp_model)
    finally:
        shutil.rmtree(_tmp_dir, ignore_errors=True)
    print("  Model loaded.")

    # ── Step 4: Run inference ──────────────────────────────────────────────────────
    print("\n--- Step 4: Running predictions ---")
    texts = [c["text"] for c in chunks]
    preds = model.predict(texts)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    print(f"  Done — {len(preds)} predictions generated.")

    # ── Step 5: Build output DataFrame ────────────────────────────────────────────
    print("\n--- Step 5: Saving results ---")
    rows = []
    for chunk, pred in zip(chunks, preds):
        row = {
            "chunk_id":   chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end":   chunk["page_end"],
            "text":       chunk["text"],
        }
        for label, val in zip(LABEL_COLUMNS, pred):
            row[label] = int(val)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False, sep=";")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"\nDone! {len(out_df)} chunks predicted.")
    print(out_df[["chunk_id", "page_start", "page_end"] + LABEL_COLUMNS].to_string(index=False))