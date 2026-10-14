"""
Stage 1: Data Preprocessing
Justification for each step is commented inline (required by guidelines).
"""
import pandas as pd
import re
import numpy as np
from pathlib import Path

# Paths are resolved relative to this file, not the working directory, so the
# script runs correctly from anywhere (terminal, notebook, IDE).
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw' / 'flipkart_product.csv'
CLEANED = ROOT / 'data' / 'processed' / 'flipkart_cleaned.csv'
CLEANED.parent.mkdir(parents=True, exist_ok=True)

# Load with latin1 encoding -- the raw file has non-UTF8 bytes
# (likely from currency symbols / emoji corrupted during original scraping)
df = pd.read_csv(RAW, encoding='latin1')
print(f"Raw shape: {df.shape}")

# --- Step 1: Remove rows with corrupted/misaligned columns ---
# A small number of rows have non-numeric values in 'Rate' (e.g. product
# names leaked into this column), caused by unescaped commas in the
# original scrape breaking column alignment. These rows are unusable
# for supervised labeling, so we drop them.
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
before = len(df)
df = df.dropna(subset=['Rate'])
df['Rate'] = df['Rate'].astype(int)
df = df[df['Rate'].between(1, 5)]  # ratings must be 1-5
print(f"Dropped {before - len(df)} rows with invalid/corrupted Rate values")

# --- Step 2: Drop rows with missing review text ---
# Summary is our primary text field for sentiment/topic analysis;
# a review with no text carries no analyzable signal.
before = len(df)
df = df.dropna(subset=['Summary'])
print(f"Dropped {before - len(df)} rows with missing Summary text")

# --- Step 3: Clean product names and text fields ---
# The raw text contains mojibake artifacts (e.g. '?\xff?\xff', 'Â Â')
# from the original encoding corruption. We strip non-ASCII noise while
# preserving actual review content.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII mojibake
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['ProductName_clean'] = df['ProductName'].apply(clean_text)
df['Summary_clean'] = df['Summary'].apply(clean_text)
df['Review_clean'] = df['Review'].apply(clean_text)

# --- Step 4: Clean Price to numeric ---
# Price has a garbled currency-symbol prefix and comma thousand-separators
# (e.g. '??3,999' -> 3999). Needed if we want price as a feature later.
def clean_price(p):
    if pd.isna(p):
        return np.nan
    digits = re.sub(r'[^\d]', '', str(p))
    return int(digits) if digits else np.nan

df['Price_clean'] = df['Price'].apply(clean_price)

# --- Step 5a: Record how often each review text repeats, BEFORE dedup ---
# This MUST happen before Step 5 below. Step 5 deletes the repeated rows,
# so counting repeats afterwards would return 1 for almost every row and
# the feature would carry no signal at all.
# Two counts are recorded:
#   text_freq_global  - times this text appears anywhere in the dataset
#                       (across all products). Catches generic one-word
#                       reviews copy-pasted over hundreds of products.
#   text_freq_product - times this text repeats on the SAME product.
# Both are used as anomaly-detection features in 04_clustering_anomaly.py.
text_norm = df['Summary_clean'].str.lower().str.strip()
df['text_freq_global'] = text_norm.map(text_norm.value_counts())
df['text_freq_product'] = text_norm.groupby(
    [df['ProductName_clean'], text_norm]).transform('size')
print(f"\nPre-dedup text frequency recorded:")
print(f"  max global repeats of one review text: {df['text_freq_global'].max()}")
print(f"  reviews whose text appears >1 time globally: "
      f"{(df['text_freq_global'] > 1).sum()} ({100*(df['text_freq_global'] > 1).mean():.1f}%)")

# --- Step 5: Remove exact duplicate reviews ---
# Duplicate rows (same product + same review text) likely represent
# scraping artifacts rather than genuinely distinct customer reviews.
before = len(df)
df = df.drop_duplicates(subset=['ProductName_clean', 'Summary_clean'])
print(f"Dropped {before - len(df)} exact duplicate reviews")

# --- Step 6: Derive sentiment label from star rating (ground truth) ---
# Standard mapping used in review-analytics literature:
# 4-5 stars = positive, 3 = neutral, 1-2 = negative.
def rate_to_sentiment(r):
    if r >= 4:
        return 'positive'
    elif r == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment_label'] = df['Rate'].apply(rate_to_sentiment)

# --- Step 7: Filter out extremely short/uninformative reviews for topic modeling ---
# Reviews under 3 words (e.g. "Nice", "Good") carry no topical signal
# and would only add noise to LDA topic discovery.
df['word_count'] = df['Summary_clean'].apply(lambda x: len(x.split()))

print(f"\nFinal cleaned shape: {df.shape}")
print(f"\nSentiment label distribution:\n{df['sentiment_label'].value_counts()}")
print(f"\nWord count stats:\n{df['word_count'].describe()}")

df.to_csv(CLEANED, index=False)
print(f"\nSaved cleaned data to {CLEANED.relative_to(ROOT)}")
