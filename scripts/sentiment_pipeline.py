import pandas as pd
import numpy as np
import re
from string import punctuation
from pathlib import Path
from collections import Counter


def load_raw_data(csv_path: str = "data/judge_1377884607_tweet_product_company.csv") -> pd.DataFrame:
    """Load the raw CSV file with explicit dtypes to speed parsing & cut RAM usage."""
    dtype_map = {
        "tweet_text": "string",
        "emotion_in_tweet_is_directed_at": "category",
        "is_there_an_emotion_directed_at_a_brand_or_product": "category",
    }
    return pd.read_csv(csv_path, dtype=dtype_map)


# -----------------------------------------------------------------------------
# Cleaning helpers (taken verbatim from the notebook)
# -----------------------------------------------------------------------------

def remove_html_urls_mentions(input_text: str) -> str:
    """Remove HTML tags, URLs and Twitter @mentions from a tweet."""
    pattern_html = re.compile("<.*?>")
    text_without_html = pattern_html.sub(r"", input_text)

    text_without_urls = re.sub(r"http\S+|www\S+|https\S+", "", text_without_html)

    text_without_mentions = re.sub(r"@\w+\s*", "", text_without_urls)

    return text_without_mentions


def encode_emojis(text: str) -> str:
    """Unicode-escape the emojis in *text* (matches notebook behaviour)."""
    return text.encode("unicode-escape").decode("utf-8")


def process_tweet(tweet: str) -> list[str]:
    """Lower-case, tokenize and strip stop-words & punctuation (lazy NLTK import)."""
    # Heavy NLTK imports are moved inside the function so the module itself is cheap to import.
    from nltk.tokenize import RegexpTokenizer  # noqa: WPS433 (allow internal import)
    from nltk.corpus import stopwords  # noqa: WPS433

    tweet = tweet.lower()
    pattern = r"\b\w+(?:'\w+)?\b"
    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(tweet)

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    return tokens


# -----------------------------------------------------------------------------
# Text processing utilities (vectorised + optional parallelism)
# -----------------------------------------------------------------------------

_TOKEN_PATTERN = r"\b\w+(?:'\w+)?\b"  # pre‐compiled pattern for performance


def _get_stop_words() -> set[str]:
    """Return a cached set of English stop-words (downloads once)."""
    _ensure_nltk()
    from nltk.corpus import stopwords  # noqa: WPS433 (lazy heavy import)

    return set(stopwords.words("english"))


def tokenize_series(text_series: pd.Series, n_jobs: int = 1) -> pd.Series:
    """Vectorised tokenisation of a pandas Series.

    Parameters
    ----------
    text_series: pd.Series
        Series of strings to tokenize.
    n_jobs: int, default 1
        Number of parallel processes. If >1, uses joblib.Parallel.
    """
    # Lower-case + regex token extraction (vectorised, no Python loop)
    tokens_series = text_series.str.lower().str.findall(_TOKEN_PATTERN)

    stop_words = _get_stop_words()

    if n_jobs == 1:
        return tokens_series.apply(
            lambda toks: [t for t in toks if t not in stop_words and t not in punctuation]
        )

    # Parallel branch (lazy import of joblib to avoid overhead if unused)
    from joblib import Parallel, delayed  # noqa: WPS433

    def _filter_tokens(tok_list):
        return [t for t in tok_list if t not in stop_words and t not in punctuation]

    processed = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_filter_tokens)(tok_list) for tok_list in tokens_series
    )
    return pd.Series(processed, index=text_series.index)


# Simple frequency helper using Counter (drop-in replacement for NLTK.FreqDist)

def top_tokens(token_series: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    """Return *top_n* most common tokens across a Series of token lists."""
    flat_tokens = (tok for sublist in token_series for tok in sublist)
    return Counter(flat_tokens).most_common(top_n)


# -----------------------------------------------------------------------------
# End-to-end transformation
# -----------------------------------------------------------------------------

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the series of transformations performed in the notebook."""
    # Shorten column names
    df = df.rename(
        columns={
            "tweet_text": "tweet",
            "emotion_in_tweet_is_directed_at": "brand_product",
            "is_there_an_emotion_directed_at_a_brand_or_product": "emotion",
        }
    )

    # Harmonise neutral label
    df.loc[df["emotion"] == "No emotion toward brand or product", "emotion"] = (
        "Neutral emotion"
    )

    # Infer *brand_product* from tweet text when missing
    categories = np.array(["iPad", "Apple", "iPad", "iPhone", "Google", "Android"])
    for i, row in df.iterrows():
        if pd.isnull(row["brand_product"]):
            for category in np.concatenate((categories, np.char.lower(categories))):
                if category in row["tweet"]:
                    df.loc[i, "brand_product"] = category
                    break

    # Drop rows still containing NAs or duplicates
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Drop ambiguous rows
    df = df[df["emotion"] != "I can't tell"].reset_index(drop=True)

    # Merge brand categories into Apple vs Google only
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

    # Text cleaning pipeline
    df["cleaned_tweet"] = df["tweet"].apply(
        lambda text: encode_emojis(remove_html_urls_mentions(text))
    )

    # Vectorised tokenisation; set n_jobs>1 for parallel processing
    df["tokenized_tweets"] = tokenize_series(df["cleaned_tweet"], n_jobs=1)

    # Re-order columns for convenience
    df = df[["tweet", "cleaned_tweet", "tokenized_tweets", "brand_product", "emotion"]]

    return df


CACHE_PATH = Path("data/processed_df.parquet")


def _ensure_nltk() -> None:  # moved outside to reuse in caching
    """Download required NLTK corpora once (quietly)."""
    import nltk  # noqa: WPS433

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_processed_dataframe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned dataframe, using on-disk cache when available.

    Parameters
    ----------
    force_refresh: bool, default False
        If True, ignore any cached file and rebuild from raw CSV.
    """
    if not force_refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    # Rebuild and store cache
    _ensure_nltk()
    raw = load_raw_data()
    processed = prepare_dataframe(raw)

    # Attempt Parquet first (needs pyarrow or fastparquet)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(CACHE_PATH, index=False)
    except Exception:  # pragma: no cover – fall back when backend missing
        fallback = CACHE_PATH.with_suffix(".pkl")
        processed.to_pickle(fallback)
        print(f"[sentiment_pipeline] Parquet backend missing, cached as {fallback.name}")

    return processed


if __name__ == "__main__":
    df = get_processed_dataframe()

    # Show basic info as in the notebook
    print("Processed dataset shape:", df.shape)
    print(df.head())