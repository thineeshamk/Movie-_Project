# app/predict.py
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from app.models import MovieInput
from app.processing import clean_plot_text, get_longformer_embedding

# Global artifacts dictionary
artifacts = {
    "model": None,
    "encoder": None,
    "bert_scaler": None,
    "svd": None,
    "target_encoder": None,
    "feature_names": None,
    "longformer_tokenizer": None,
    "longformer_model": None,
    "device": None,
    "model_version": "1.0.0",
    "prediction_threshold": 0.48
}

def load_model_artifacts():
    """Load all trained model artifacts"""
    print("Loading model artifacts...")
    artifacts["model"] = joblib.load("artifacts/stacked_model.joblib")
    artifacts["encoder"] = joblib.load("artifacts/one_hot_encoder.joblib")
    artifacts["bert_scaler"] = joblib.load("artifacts/bert_scaler.joblib")
    artifacts["svd"] = joblib.load("artifacts/svd_transformer.joblib")
    artifacts["target_encoder"] = joblib.load("artifacts/target_encoder.joblib")
    artifacts["feature_names"] = joblib.load("artifacts/feature_names.joblib")
    print("Model artifacts loaded successfully.")

def load_longformer_model():
    """Load Longformer model for embeddings"""
    print("Loading Longformer model...")
    model_name = "allenai/longformer-base-4096"
    artifacts["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts["longformer_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    artifacts["longformer_model"] = AutoModel.from_pretrained(model_name).to(artifacts["device"])
    print("Longformer model loaded successfully.")

def normalize_genre(genre: str) -> str:
    """Map user-selected genre to model categories"""
    main_genres = ["Action", "Drama", "Comedy", "Biography"]
    return genre if genre in main_genres else "Other"

def normalize_mpa(mpa: str) -> str:
    """Map user-selected MPA rating to model categories"""
    main_mpas = ["PG-13", "R"]
    return mpa if mpa in main_mpas else "Other"

def make_prediction(input_data: MovieInput) -> dict:
    """Make a prediction for a movie"""
    
    # 1. Clean and embed the plot synopsis
    cleaned_plot = clean_plot_text(input_data.plot_synopsis)
    plot_embedding = get_longformer_embedding(
        cleaned_plot,
        artifacts["longformer_tokenizer"],
        artifacts["longformer_model"],
        artifacts["device"]
    )
    
    # 2. Scale and apply SVD to embedding
    embedding_scaled = artifacts["bert_scaler"].transform([plot_embedding])
    embedding_svd = artifacts["svd"].transform(embedding_scaled)
    embedding_df = pd.DataFrame(
        embedding_svd, 
        columns=[f"bert_svd_{i}" for i in range(embedding_svd.shape[1])]
    )
    
    # 3. Normalize and encode categorical features
    normalized_genre = normalize_genre(input_data.genre)
    normalized_mpa = normalize_mpa(input_data.mpa_rating)
    
    categorical_data = pd.DataFrame(
        [[normalized_mpa, normalized_genre]], 
        columns=["MPA", "1st Genre"]
    )
    encoded_cats = artifacts["encoder"].transform(categorical_data)
    encoded_cats_df = pd.DataFrame(
        encoded_cats, 
        columns=artifacts["encoder"].get_feature_names_out(["MPA", "1st Genre"])
    )

    # 4. Prepare numerical features
    numerical_data = {
        'budget': input_data.budget,
        'Duration_Minutes': input_data.duration_minutes,
        'First Actor Avg': input_data.first_actor_avg,
        'Second Actor Avg': input_data.second_actor_avg,
        'Average IMDb Rating': input_data.average_imdb_rating
    }
    numerical_df = pd.DataFrame([numerical_data])

    # 5. Combine all features
    final_df = pd.concat([numerical_df, encoded_cats_df, embedding_df], axis=1)
    final_df = final_df[artifacts["feature_names"]]

    # 6. Make prediction
    probability = artifacts["model"].predict_proba(final_df)[0][1]
    prediction_int = 1 if probability >= artifacts["prediction_threshold"] else 0
    prediction_label = artifacts["target_encoder"].inverse_transform([prediction_int])[0]

    return {
        "prediction": prediction_label,
        "probability_success": float(probability),
        "model_version": artifacts["model_version"]
    }