from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MovieRecommender:
    def __init__(self):
        self.all_movie_ids = []
        self.all_movie_titles = {}

    def prepare_data(self, movies_df):
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()

    def compute_cosine_similarity(self, similarity_matrix, liked_movie_ids):
        liked_movie_indices = [self.all_movie_ids.index(movie_id) for movie_id in liked_movie_ids if movie_id in self.all_movie_ids]

        if not liked_movie_indices:
            return []

        liked_movies_vector = np.mean(similarity_matrix[:, liked_movie_indices], axis=1)
        cosine_sim = cosine_similarity(similarity_matrix.T, liked_movies_vector.reshape(1, -1))

        return cosine_sim.flatten()

    def generate_recommendations(self, liked_movie_ids, similarity_matrix):
        cosine_sim = self.compute_cosine_similarity(similarity_matrix, liked_movie_ids)

        predictions = []
        for idx, sim_score in enumerate(cosine_sim):
            movie_id = self.all_movie_ids[idx]
            title = self.all_movie_titles.get(movie_id, "Unknown Title")
            predictions.append({
                "movie_id": movie_id,
                "average_similarity": sim_score,
                "title": title
            })

        recommendations = sorted(predictions, key=lambda x: x['average_similarity'], reverse=True)
        recommended_movies = [recommendation for recommendation in recommendations
                              if recommendation["movie_id"] not in liked_movie_ids]

        return recommended_movies