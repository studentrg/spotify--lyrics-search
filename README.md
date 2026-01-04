# 🎵 Spotify Lyric Search using Sentence-BERT

Project Overview

The Spotify Lyric Search project is a text identification system that predicts the song title and artist when provided with a short snippet of lyrics.
The model is trained on the Spotify 50k+ Songs Dataset using Natural Language Processing (NLP) techniques.

## 📌 Objective

To build a machine learning model that:

Accepts a partial lyric input
Processes and vectorizes text data
Identifies the most likely song and artist
Demonstrates text similarity or classification performance

## 🧠 Approach



Input lyric snippet → model inference → ranked song matches

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

Future Improvements

Use Word2Vec / BERT embeddings
Improve accuracy using LSTM / Transformer models
Add REST API for real-time lyric search
Deploy as a web application

Author
Rinki Ghosh
Btech CSE
