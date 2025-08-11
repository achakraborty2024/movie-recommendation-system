# --- Save All Essential Models & Data for Frontend Integration ---

import os
import joblib
import pandas as pd

# Optionally, for SentenceTransformer or Keras models:
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    from tensorflow import keras
except ImportError:
    keras = None

# ---- Step 1: Define Save Directory ----
save_dir = './recommendation_models'  # Change as needed
os.makedirs(save_dir, exist_ok=True)
print(f"Saving models to directory: {save_dir}")

# ---- Step 2: Save TF-IDF Vectorizer ----
if 'tfidf' in globals():
    tfidf_path = os.path.join(save_dir, 'tfidf_vectorizer.joblib')
    print("vocab size:", len(current_tfidf.vocabulary_))  # should be 86621
    print(current_tfidf.get_params())
    joblib.dump(current_tfidf, tfidf_path)
    print(f"Saved TF-IDF Vectorizer to {tfidf_path}")
else:
    print("TF-IDF Vectorizer not found. Skipping save.")

# ---- Step 3: Save Autoencoder Encoder Model ----
if 'current_encoder' in globals() and keras is not None:
    encoder_path = os.path.join(save_dir, 'autoencoder_encoder_model.keras')
    current_encoder.save(encoder_path)
    print(f"Saved Autoencoder Encoder Model to {encoder_path}")
else:
    print("Autoencoder Encoder Model not found or Keras not installed. Skipping save.")

# ---- Step 4: Save KMeans Model ----
if 'kmeans_model_tuned' in globals():
    kmeans_path = os.path.join(save_dir, 'kmeans_model.joblib')
    joblib.dump(kmeans_model_tuned, kmeans_path)
    print(f"Saved KMeans Model to {kmeans_path}")
else:
    print("KMeans Model not found. Skipping save.")

# ---- Step 5: Save Sentence-BERT or Embedding Model ----
if 'model' in globals():
    sentence_bert_path = os.path.join(save_dir, 'sentence_bert_model')
    # If it's a SentenceTransformer, save using its .save() method
    if SentenceTransformer is not None and isinstance(model, SentenceTransformer):
        model.save(sentence_bert_path)
        print(f"Saved Sentence-BERT Model to {sentence_bert_path}")
    # Otherwise, try saving as a Keras model (optional fallback)
    elif keras is not None and hasattr(model, "save"):
        model.save(sentence_bert_path)
        print(f"Saved model using Keras save() to {sentence_bert_path}")
    else:
        print("Model type unknown for 'model'. Not saved.")
else:
    print("Sentence-BERT Model not found. Skipping save.")

# ---- Step 6: Save Merged Movie Data (with essential columns) ----
if 'merged_df' in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
    essential_cols = [
        'id', 'title', 'genres', 'keywords', 'overview',
        'overview_sentiment_score', 'soup', 'enhanced_soup'
    ]
    cols_to_save = [col for col in essential_cols if col in merged_df.columns]
    if cols_to_save:
        merged_df_path = os.path.join(save_dir, 'merged_movie_data.csv')
        merged_df[cols_to_save].to_csv(merged_df_path, index=False)
        print(f"Saved essential merged_df columns to {merged_df_path}")
    else:
        print("No essential columns found in merged_df to save.")
else:
    print("merged_df not found or is empty. Skipping save.")

# ---- Step 7: Save Indices Series (for mapping in frontend/backend) ----
if 'indices' in globals() and hasattr(indices, 'empty') and not indices.empty:
    indices_path = os.path.join(save_dir, 'indices.pkl')
    indices.to_pickle(indices_path)
    print(f"Saved indices Series to {indices_path}")
else:
    print("indices Series not found or is empty. Skipping save.")


print("\nSaving process complete.")
print(f"All files are available in: {save_dir}")
print("To use these models in a separate application, load them using the respective libraries (joblib.load, keras.models.load_model, SentenceTransformer, pd.read_csv, pd.read_pickle, etc.).")
