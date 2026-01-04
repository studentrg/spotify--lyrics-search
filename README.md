# 🎵 Spotify Lyric Search using Sentence-BERT

A Machine Learning / NLP project that identifies the **song title and artist** from a **small snippet of lyrics**, using **semantic search with Sentence-BERT**.

This project is inspired by how platforms like **Spotify** and **Genius** perform lyric-based search.

---

## 📌 Objective

To build a text-identification system that:
- Takes a short lyric snippet as input
- Predicts the most relevant **song name and artist**
- Uses **semantic understanding**, not just keyword matching

---

## 🧠 Approach

Two approaches were explored:
1. **TF-IDF + Cosine Similarity** (baseline)
2. **Sentence-BERT (final implementation)** ✅

The final version uses **Sentence-BERT**, which captures the **meaning of lyrics**, resulting in much higher accuracy.

---

## 🗂️ Dataset

- **Spotify Songs Lyrics Dataset**
- Size: 50,000+ songs
- Columns used:
  - `artist` → Artist name
  - `song` → Song title
  - `text` → Full lyrics

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Sentence-Transformers
- PyTorch

---

## 🔑 Key Optimization

Embedding full lyrics caused **semantic dilution** due to model token limits.  
To solve this, only the **first 50–60 words of each song** were embedded, preserving song identity and improving accuracy.

This technique significantly boosts real-world performance.

---

## 📁 Project Structure

spotify-lyric-search/
│
├── data/
│ └── spotify_lyrics.csv
│
├── src/
│ ├── init.py
│ ├── bert_model.py
│ └── bert_predict.py
│
├── bert_main.py
├── requirements.txt

---
## ▶️ How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

## ▶️  Run
python bert_main.py

Sample Input
we were both young when i first saw you

Sample Output
🎯 BERT Top Predictions:
1. 🎵 Love Story - Taylor Swift (0.57)
2. 🎵 Bitter Wine - Bon Jovi (0.56)
3. 🎵 Catch My Breath - Westlife (0.55)

## ▶️Results
Correct song ranked at Top-1
Confidence scores between 0.5 – 0.7 indicate strong semantic matches
Handles paraphrased or partial lyrics effectively
