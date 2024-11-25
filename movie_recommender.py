from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from logger_config import get_logger


class MovieRecommender:
    def __init__(self, movies_df):
        self.logger = get_logger("MovieTrainer")
        self.logger.info("Initializing MovieRecommender.")
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()
        self.logger.info(f"Loaded {len(self.all_movie_ids)} movies into the recommender.")

    def compute_cosine_similarity(self, similarity_matrix, liked_movie_ids):
        self.logger.info(f"Computing cosine similarity for {len(liked_movie_ids)} liked movies.")
        liked_movie_indices = [self.all_movie_ids.index(movie_id) for movie_id in liked_movie_ids if movie_id in self.all_movie_ids]

        if not liked_movie_indices:
            self.logger.warning("No liked movies found in the movie dataset.")
            return []

        self.logger.info(f"Found indices for {len(liked_movie_indices)} liked movies in the dataset.")
        liked_movies_vector = np.mean(similarity_matrix[:, liked_movie_indices], axis=1)
        cosine_sim = cosine_similarity(similarity_matrix.T, liked_movies_vector.reshape(1, -1))
        self.logger.info("Cosine similarity computation completed.")

        return cosine_sim.flatten()

    def generate_recommendations(self, liked_movie_ids, similarity_matrix):
        self.logger.info(f"Generating recommendations for {len(liked_movie_ids)} liked movies.")
        cosine_sim = self.compute_cosine_similarity(similarity_matrix, liked_movie_ids)

        if not cosine_sim.size:
            self.logger.warning("No cosine similarity results; no recommendations can be generated.")
            return []

        predictions = []
        for idx, sim_score in enumerate(cosine_sim):
            movie_id = self.all_movie_ids[idx]
            title = self.all_movie_titles.get(movie_id, "Unknown Title")
            predictions.append({
                "movie_id": movie_id,
                "average_similarity": sim_score,
                "title": title
            })

        self.logger.info(f"Generated similarity scores for {len(predictions)} movies.")

        recommendations = sorted(predictions, key=lambda x: x['average_similarity'], reverse=True)
        recommended_movies = [recommendation for recommendation in recommendations
                              if recommendation["movie_id"] not in liked_movie_ids]

        self.logger.info(f"Filtered out liked movies. {len(recommended_movies)} recommendations ready.")

        return recommended_movies