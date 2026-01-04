import numpy as np
from sentence_transformers.util import cos_sim

def predict_song_bert(snippet, model, embeddings, df, top_k=5):
    snippet_embedding = model.encode(snippet, convert_to_numpy=True)

    scores = cos_sim(snippet_embedding, embeddings)[0].cpu().numpy()

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "song": df.iloc[idx]['song'],
            "artist": df.iloc[idx]['artist'],
            "confidence": float(scores[idx])
        })

    return results
