from __future__ import annotations

"""Sentiment analysis data pipeline.

This module was adapted from the original Jupyter notebook. It provides:

* ``load_raw_data`` – read the raw CSV with explicit dtypes
* ``prepare_dataframe`` – clean & tokenize tweets
* ``get_processed_dataframe`` – cached accessor
* ``main`` – console entry printing dataset info (exposed via ``sentiment-pipeline`` CLI)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path
from string import punctuation
from typing import List

# Public ---------------------------------------------------------------------
__all__ = [
    "load_raw_data",
    "prepare_dataframe",
    "get_processed_dataframe",
    "top_tokens",
    "main",
]

# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------


def load_raw_data(csv_path: str = "data/judge_1377884607_tweet_product_company.csv") -> pd.DataFrame:
    """Load the raw CSV with explicit dtypes."""
    dtype_map = {
        "tweet_text": "string",
        "emotion_in_tweet_is_directed_at": "category",
        "is_there_an_emotion_directed_at_a_brand_or_product": "category",
    }
    return pd.read_csv(csv_path, dtype=dtype_map)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_nltk() -> None:
    """Download required NLTK corpora once (quietly)."""
    import nltk  # noqa: WPS433

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# Caching path
_CACHE_PATH = Path("data/processed_df.parquet")

# Regex pattern compiled once for performance
_TOKEN_PATTERN = r"\b\w+(?:'\w+)?\b"


def _get_stop_words() -> set[str]:
    _ensure_nltk()
    from nltk.corpus import stopwords  # noqa: WPS433

    return set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------------


def remove_html_urls_mentions(text: str) -> str:
    pattern_html = re.compile("<.*?>")
    text = pattern_html.sub(r"", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+\s*", "", text)
    return text


def encode_emojis(text: str) -> str:
    return text.encode("unicode-escape").decode("utf-8")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def tokenize_series(text_series: pd.Series, n_jobs: int = 1) -> pd.Series:
    tokens_series = text_series.str.lower().str.findall(_TOKEN_PATTERN)
    stop_words = _get_stop_words()

    if n_jobs == 1:
        return tokens_series.apply(
            lambda toks: [t for t in toks if t not in stop_words and t not in punctuation]
        )

    from joblib import Parallel, delayed  # noqa: WPS433

    def _filter(tok_list: List[str]) -> List[str]:
        return [t for t in tok_list if t not in stop_words and t not in punctuation]

    processed = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_filter)(tok_list) for tok_list in tokens_series
    )
    return pd.Series(processed, index=text_series.index)


def top_tokens(token_series: pd.Series, top_n: int = 20):
    flat = (tok for sub in token_series for tok in sub)
    return Counter(flat).most_common(top_n)


# ---------------------------------------------------------------------------
# Main dataframe transformation
# ---------------------------------------------------------------------------


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    df = df.rename(
        columns={
            "tweet_text": "tweet",
            "emotion_in_tweet_is_directed_at": "brand_product",
            "is_there_an_emotion_directed_at_a_brand_or_product": "emotion",
        }
    )

    # Normalise neutral label
    df.loc[df["emotion"] == "No emotion toward brand or product", "emotion"] = "Neutral emotion"

    # Infer brand when missing
    categories = np.array(["iPad", "Apple", "iPhone", "Google", "Android"])
    for idx, row in df.iterrows():
        if pd.isnull(row["brand_product"]):
            for cat in np.concatenate((categories, np.char.lower(categories))):
                if cat in row["tweet"]:
                    df.at[idx, "brand_product"] = cat
                    break

    # Drop NAs, duplicates, ambiguous rows
    df = df.dropna().drop_duplicates()
    df = df[df["emotion"] != "I can't tell"].reset_index(drop=True)

    # Collapse brand categories to Apple vs Google
    apple_cats = [
        "iPad",
        "Apple",
        "iPhone",
        "iPad or iPhone App",
        "ipad",
        "apple",
        "iphone",
        "Other Apple product or service",
    ]
    google_cats = [
        "Google",
        "google",
        "Other Google product or service",
        "Android",
        "Android App",
        "android",
    ]
    for idx, row in df.iterrows():
        if row["brand_product"] in apple_cats:
            df.at[idx, "brand_product"] = "Apple"
        elif row["brand_product"] in google_cats:
            df.at[idx, "brand_product"] = "Google"

    # Clean & tokenize
    df["cleaned_tweet"] = df["tweet"].apply(lambda t: encode_emojis(remove_html_urls_mentions(t)))
    df["tokenized_tweets"] = tokenize_series(df["cleaned_tweet"], n_jobs=1)

    return df[["tweet", "cleaned_tweet", "tokenized_tweets", "brand_product", "emotion"]]


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


def get_processed_dataframe(force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh and _CACHE_PATH.exists():
        return pd.read_parquet(_CACHE_PATH)

    raw = load_raw_data()
    processed = prepare_dataframe(raw)

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(_CACHE_PATH, index=False)
    except Exception:
        processed.to_pickle(_CACHE_PATH.with_suffix(".pkl"))

    return processed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    df = get_processed_dataframe()
    print("Processed dataset shape:", df.shape)
    print(df.head())


if __name__ == "__main__":  # noqa: D401
    main()