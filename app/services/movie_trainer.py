import os
import numpy as np
import joblib
from dotenv import load_dotenv
import pandas as pd
from app.utils.logger_config import get_logger
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MovieTrainer:
    def __init__(self):
        self.logger = get_logger("MovieTrainer")
        self.similarity_matrix = None
        self.indices = None

    def load_movies_from_db(self):
        self.logger.info("Loading movies from MongoDB.")
        mongo_uri = "mongodb://user:pass@localhost:27017" #os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        db = client["movieDB"]
        collection = db["movieDetails"]
        movies = list(collection.find())
        self.logger.info(f"Loaded {len(movies)} movies from the database.")
        return pd.DataFrame(movies)

    def create_soup(self, movies_df):
        self.logger.info("Creating soup for feature extraction.")
        
        genres = movies_df["genres"].apply(
            lambda x: [i["name"] for i in x]
        )
        genres = genres.apply(
            lambda x: x * 2
        )
        cast = movies_df["cast"].apply(
            lambda x: [i["name"] for i in x] if isinstance(x, list) else []
        )
        cast = cast.apply(
            lambda x: x[:3] if len(x) >= 3 else x
        )
        cast = cast.apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )
        director = movies_df["crew"].apply(
            lambda x: next((i["name"]
                           for i in x if i["job"] == "Director"), np.nan)
        )
        director = director.astype("str").apply(
            lambda x: [str.lower(x.replace(" ", ""))] * 3
        )
        keywords = movies_df["keywords"].apply(
            lambda x: [i["name"] for i in x] if isinstance(x, list) else []
        )
        keywords = keywords.apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )

        soup = (keywords + cast + genres + director)
        soup = soup.apply(lambda x: " ".join(x))

        self.logger.info(f"Soup created for {len(movies_df)} movies.")

        return soup

    def prepare_vectors(self, movies_df):
        self.logger.info("Preparing feature vectors.")
        vectorizer = CountVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=0.0, stop_words="english"
        )
        soup = self.create_soup(movies_df)
        feature_matrix = vectorizer.fit_transform(soup)
        self.logger.info("Feature vectors prepared.")

        return feature_matrix

    def compute_similarity(self, all_movies):
        self.logger.info("Computing cosine similarity matrix.")
        similarity_matrices = self.prepare_vectors(all_movies)
        cosine_sim = cosine_similarity(similarity_matrices)
        all_movies = all_movies.reset_index()
        self.indices = dict(zip(all_movies["_id"], all_movies.index))
        self.logger.info("Cosine similarity matrix computation completed.")

        return cosine_sim

    def train_model(self, all_movies):
        self.logger.info("Training model by computing similarity matrix.")
        self.similarity_matrix = self.compute_similarity(all_movies)
        self.logger.info("Model training completed. Similarity matrix is ready.")

    def save_model(self, model_path):
        model = {
            "similarity_matrix": self.similarity_matrix,
            "indices": self.indices
        }

        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")