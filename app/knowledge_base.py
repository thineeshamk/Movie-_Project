# app/knowledge_base.py
import pandas as pd
import numpy as np
import faiss
import ast
import torch
from pathlib import Path

from app.processing import clean_plot_text

# Centralized file path
DATA_FILE_PATH = Path(r"C:\Users\Dell\Movie project\DATA\successful_movies_embeddings.xlsx")


class KnowledgeBase:
    """Manages the successful movies knowledge base"""

    def __init__(self):
        self.df = None
        self.index = None
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self, tokenizer, model, device, file_path: Path = DATA_FILE_PATH):
        """Load the knowledge base from Excel file"""

        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

        print(f"Loading knowledge base from {file_path}...")

        self.df = pd.read_excel(file_path)
        print(f"Loaded {len(self.df)} successful movies")

        # Parse embeddings
        if 'embedding_str' in self.df.columns:
            self.df["embedding"] = self.df["embedding_str"].apply(
                lambda x: np.array(ast.literal_eval(x), dtype="float32")
            )
        elif 'embedding' in self.df.columns:
            pass
        else:
            raise ValueError("No embedding column found in the data!")

        # Build FAISS index
        embedding_dim = len(self.df["embedding"].iloc[0])
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.vstack(self.df["embedding"].values))
        print(f"FAISS index built with {self.index.ntotal} vectors")

        # Store model components
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def get_embedding_for_text(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
        cleaned_text = clean_plot_text(text)

        inputs = self.tokenizer(
            cleaned_text,
            max_length=4096,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy().flatten().astype('float32')

    def find_similar_movies(self, movie_data: dict, k: int = 5):
        """Find k most similar movies to the given movie"""
        combined_text = f"""Title: {movie_data.get('title', 'Unknown')}.
Genre: {movie_data['genre']}.
Budget: {movie_data['budget']}.
Duration: {movie_data['duration_minutes']} minutes.
MPA Rating: {movie_data['mpa_rating']}.
First Actor IMDb Rating: {movie_data['first_actor_avg']}.
Second Actor IMDb Rating: {movie_data['second_actor_avg']}.
Director IMDb Rating: {movie_data['average_imdb_rating']}.
Plot: {movie_data['plot_synopsis']}"""

        embedding = self.get_embedding_for_text(combined_text)

        # Search in FAISS
        D, I = self.index.search(np.array([embedding]), k)

        similar_movies = []
        for idx, dist in zip(I[0], D[0]):
            movie = self.df.iloc[idx]
            similar_movies.append({
                'title': movie['Title'],
                'genre': movie['1st Genre'],
                'budget': float(movie['budget']),
                'duration_minutes': int(movie['Duration_Minutes']),
                'first_actor_avg': float(movie['First Actor Avg']),
                'second_actor_avg': float(movie['Second Actor Avg']),
                'director_avg': float(movie['Director Avg']),
                'mpa': movie['MPA'],
                'plot': movie['Movie Plot'],
                'distance': float(dist)
            })

        return similar_movies

    def get_movie_by_title(self, title: str):
        """Get a specific movie by title"""
        matches = self.df[self.df['Title'].str.lower().str.contains(title.lower(), na=False)]
        if len(matches) > 0:
            movie = matches.iloc[0]
            return {
                'title': movie['Title'],
                'genre': movie['1st Genre'],
                'budget': float(movie['budget']),
                'duration_minutes': int(movie['Duration_Minutes']),
                'first_actor_avg': float(movie['First Actor Avg']),
                'second_actor_avg': float(movie['Second Actor Avg']),
                'director_avg': float(movie['Director Avg']),
                'mpa': movie['MPA'],
                'plot': movie['Movie Plot']
            }
        return None

    def get_statistics(self, genre: str = None):
        """Get statistics from knowledge base"""
        filtered_df = self.df[self.df['1st Genre'] == genre] if genre else self.df

        if len(filtered_df) == 0:
            return None

        return {
            'count': len(filtered_df),
            'avg_budget': float(filtered_df['budget'].mean()),
            'avg_duration': float(filtered_df['Duration_Minutes'].mean()),
            'avg_director_rating': float(filtered_df['Director Avg'].mean()),
            'avg_first_actor_rating': float(filtered_df['First Actor Avg'].mean()),
            'avg_second_actor_rating': float(filtered_df['Second Actor Avg'].mean())
        }