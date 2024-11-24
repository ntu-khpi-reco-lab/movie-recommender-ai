import os
from dotenv import load_dotenv
import joblib
import logging
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

load_dotenv()

class MovieRecommender:
    def __init__(self):
        self.logger = self.setup_logger()
        self.all_movie_ids = []
        self.all_movie_titles = {}
        self.selected_movies = pd.DataFrame()


    @staticmethod
    def setup_logger():
        logger = logging.getLogger('MovieRecommender')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


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


    def prepare_data(self, movies_df):
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()


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


    def compute_average_similarity(self, similarity_matrix):
        predictions = []

        for j, movie_id in enumerate(self.all_movie_ids):
            similarity_scores = similarity_matrix[:, j]
            average_similarity = np.nanmean(similarity_scores)
            title = self.all_movie_titles.get(movie_id, "Unknown Title")

            predictions.append({
                "movie_id": movie_id,
                "average_similarity": average_similarity,
                "title": title
            })

        return sorted(predictions, key=lambda x: x['average_similarity'], reverse=True)


    def train_and_save_model(self, all_movies, model_path):
        similarity_matrix = self.compute_similarity(all_movies)
        joblib.dump(similarity_matrix, model_path)
        self.logger.info(f"Model saved to {model_path}")
        return similarity_matrix


    @staticmethod
    def load_model(model_path):
        return joblib.load(model_path)


def main():
    recommender = MovieRecommender()
    all_movies = recommender.load_movies_from_db()

    movies = all_movies[:101]

    liked_movie_ids = [105, 680, 569094, 574, 5874]
    recommender.prepare_data(movies)

    model_path = "movie_similarity_model.pkl"
    similarity_matrix = recommender.train_and_save_model(movies, model_path)

    print("Cosine similarity matrix:")
    print(similarity_matrix)

    predictions = recommender.compute_average_similarity(similarity_matrix)

    for prediction in predictions[:20]:
        print(f"Movie ID: {prediction['movie_id']}, Title: {prediction['title']}, "
              f"Similarity: {prediction['average_similarity']:.2f}")


if __name__ == "__main__":
    main()
