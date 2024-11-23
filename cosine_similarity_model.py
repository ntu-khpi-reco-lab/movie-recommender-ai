import joblib
import logging
import numpy as np
from pymongo import MongoClient
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


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
    def load_movies_from_db(movie_id=None):
        client = MongoClient("mongodb://user:pass@localhost:27017")
        db = client["movieDB"]
        collection = db["movieDetails"]

        query = {}
        if movie_id:
            query["_id"] = movie_id

        projection = {
            "_id": 1,
            "title": 1,
            "genres": 1,
            "cast": 1,
            "crew": 1,
            "keywords": 1
        }

        movies = list(collection.find(query, projection))
        return pd.DataFrame(movies)


    def prepare_data(self, movies_df, liked_movie_ids):
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()
        self.selected_movies = movies_df[movies_df["_id"].isin(liked_movie_ids)]


    @staticmethod
    def create_soup(movie):

        genres = " ".join([genre.get('name', '') for genre in movie.get("genres", []) if isinstance(genre, dict)])
        cast = " ".join([actor.get('name', '') for actor in movie.get("cast", []) if isinstance(actor, dict)])
        keywords = " ".join([keyword.get('name', '') for keyword in movie.get("keywords", [])
                             if isinstance(keyword, dict)])
        directors = " ".join([member.get("name", "") for member in movie.get("crew", [])
                              if isinstance(member, dict) and member.get("job") == "Director"])

        return f"{genres} {cast} {keywords} {directors}"


    def prepare_vectors(self, movies_df):
        movies_df = movies_df.copy()
        movies_df['soup'] = movies_df.apply(self.create_soup, axis=1)
        vectorizer = CountVectorizer(stop_words='english')

        feature_matrix = vectorizer.fit_transform(movies_df['soup'])

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
    recommender.prepare_data(movies, liked_movie_ids)

    model_path = "movie_similarity_model.pkl"
    similarity_matrix = recommender.train_and_save_model(movies, model_path)

    print("Cosine similarity matrix:")
    print(similarity_matrix)

    predictions = recommender.compute_average_similarity(similarity_matrix)
    print(predictions)

    for prediction in predictions[:20]:
        print(f"Movie ID: {prediction['movie_id']}, Title: {prediction['title']}, "
              f"Similarity: {prediction['average_similarity']:.2f}")


if __name__ == "__main__":
    main()
