import os
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MovieTrainer:
    @staticmethod
    def load_movies_from_db():
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        db = client["movieDB"]
        collection = db["movieDetails"]

        projection = {
            "_id": 1,
            "title": 1,
            "genres": 1,
            "cast": 1,
            "crew": 1,
            "keywords": 1
        }

        movies = list(collection.find({}, projection))
        return pd.DataFrame(movies)

    @staticmethod
    def create_soup(movie):
        genres = movie['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        cast = movie['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        keywords = movie['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        directors = movie['crew'].apply(
            lambda x: [i['name'] for i in x if isinstance(i, dict) and i.get("job") == "Director"]
            if isinstance(x, list) else []
        )

        movie['soup'] = (genres + cast + keywords + directors).apply(lambda x: " ".join(x))

        return movie['soup']

    def prepare_vectors(self, movies_df):
        movies_df = movies_df.copy()
        vectorizer = CountVectorizer(stop_words='english')
        soup = self.create_soup(movies_df)
        feature_matrix = vectorizer.fit_transform(soup)

        return feature_matrix

    def compute_similarity(self, all_movies):
        similarity_matrices = self.prepare_vectors(all_movies)
        cosine_sim = cosine_similarity(similarity_matrices)

        return cosine_sim

    def train_model(self, all_movies):
        similarity_matrix = self.compute_similarity(all_movies)
        return similarity_matrix
