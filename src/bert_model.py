import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading dataset...")
df = pd.read_csv("data/Spotify Million Song Dataset_exported.csv")
df = df[['artist', 'song', 'text']]
df.dropna(subset=['text'], inplace=True)

# 🔑 IMPORTANT FIX: keep only first 60 words
def truncate_lyrics(text, max_words=60):
    words = text.split()
    return " ".join(words[:max_words])

df['short_text'] = df['text'].apply(truncate_lyrics)

print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding lyrics (optimized)...")
lyrics_embeddings = model.encode(
    df['short_text'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("BERT model ready!")
