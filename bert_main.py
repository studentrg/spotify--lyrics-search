from src.bert_model import df, model, lyrics_embeddings
from src.bert_predict import predict_song_bert

snippet = input("Enter lyric snippet: ")

results = predict_song_bert(snippet, model, lyrics_embeddings, df)

print("\n🎯 BERT Top Predictions:")
for i, res in enumerate(results, 1):
    print(f"{i}. 🎵 {res['song']} - {res['artist']} ({res['confidence']:.3f})")
