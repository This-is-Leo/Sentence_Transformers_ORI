# app.py
import re
import unicodedata
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# User Configurable Variables
# -----------------------------
INTERNAL_NAME_COL = "internal_name"
INTERNAL_TEXT_COL = "expertise_summary"
INTERNAL_DEPT_COL = "Department"

EXTERNAL_NAME_COL = "external_name"
EXTERNAL_TEXT_COL = "research_interest_summary"

TOP_K = 5

# -----------------------------
# Lightweight encoding helper
def fix_encoding(text):
    """Try to fix common latin1->utf-8 mis-encodings, otherwise return input."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode('latin1').decode('utf-8')
    except Exception:
        return text


# Data cleaning (single place for normalization)
def data_cleaner(text):
    """Normalize unicode, remove combining marks, strip punctuation, lowercase and collapse spaces."""
    if pd.isna(text):
        return ""
    s = str(text)
    s = fix_encoding(s)
    # normalize and remove diacritics
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # convert to ascii where possible, then to lower
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = s.strip().lower()
    # replace commas with space and remove other non-alphanum chars
    s = s.replace(',', ' ')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Data Processing
def load_and_preprocess(df, name_col, text_col, dept_col=None):
    """Validate presence of columns and apply cleaning. Returns df and the effective column names."""
    if not isinstance(name_col, str) or not isinstance(text_col, str):
        raise TypeError("Column names must be strings.")

    missing = [c for c in [name_col, text_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Apply cleaning once (data_cleaner includes fix_encoding and normalization)
    df[name_col] = df[name_col].fillna("").astype(str).apply(data_cleaner)
    df[text_col] = df[text_col].fillna("").astype(str).apply(data_cleaner)

    if dept_col and isinstance(dept_col, str) and dept_col in df.columns:
        df[dept_col] = df[dept_col].fillna("").astype(str).apply(data_cleaner)
    else:
        df['__dept_tmp__'] = ""
        dept_col = '__dept_tmp__'

    return df, name_col, text_col, dept_col


# instantiate model once (module-level)
_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def compute_top_k_matches(internal_df, external_df,
                          internal_name_col, internal_text_col, internal_dept_col,
                          external_name_col, external_text_col,
                          k=TOP_K, similarity_threshold=0.0):
    """
    Compute top-k internal matches for each external researcher.
    Uses module-level _MODEL to avoid reloading the transformer repeatedly.
    """
    internal_texts = internal_df[internal_text_col].tolist()
    external_texts = external_df[external_text_col].tolist()

    internal_emb = _MODEL.encode(internal_texts, convert_to_numpy=True, show_progress_bar=True,
                                 normalize_embeddings=True)
    external_emb = _MODEL.encode(external_texts, convert_to_numpy=True, show_progress_bar=True,
                                 normalize_embeddings=True)

    sim = np.dot(external_emb, internal_emb.T)

    rows = []
    for i_ext in range(sim.shape[0]):
        scores = sim[i_ext]
        top_k_idx = np.argsort(scores)[::-1][:k]
        for rank, j in enumerate(top_k_idx, start=1):
            score = float(scores[j])
            if score < similarity_threshold:
                continue
            rows.append({
                # store cleaned names (presentation formatting should be handled by UI)
                'external_name': external_df.loc[i_ext, external_name_col],
                'best_internal_match': internal_df.loc[j, internal_name_col],
                'similarity_score': round(score, 4),
                'internal_department': internal_df.loc[j, internal_dept_col] if internal_dept_col in internal_df.columns else "",
            })

    out_df = pd.DataFrame(rows).sort_values(by="similarity_score", ascending=False).reset_index(drop=True)
    # keep department as-is (cleaned). UI can title-case for display.
    out_df['internal_department'] = out_df['internal_department'].astype(str)
    return out_df


# -----------------------------
# Example Run (kept for script usage)
if __name__ == "__main__":
    internal_df = pd.read_csv("internal_dataset.csv")
    external_df = pd.read_csv("external_dataset.csv")

    internal_df, int_name, int_text, int_dept = load_and_preprocess(
        internal_df, INTERNAL_NAME_COL, INTERNAL_TEXT_COL, INTERNAL_DEPT_COL
    )
    external_df, ext_name, ext_text, _ = load_and_preprocess(
        external_df, EXTERNAL_NAME_COL, EXTERNAL_TEXT_COL
    )

    results_df = compute_top_k_matches(
        internal_df, external_df,
        int_name, int_text, int_dept,
        ext_name, ext_text,
        k=TOP_K
    )

    results_df.to_csv("matched_results.csv", index=False)
    print("Matching complete. Results saved to 'matched_results.csv'.")
