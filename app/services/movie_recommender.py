import numpy as np
from app.utils.logger_config import get_logger


class MovieRecommender:
    def __init__(self, movies_df):
        self.logger = get_logger("MovieTrainer")
        self.logger.info("Initializing MovieRecommender.")
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()
        self.logger.info(f"Loaded {len(self.all_movie_ids)} movies into the recommender.")

    def compute_cosine_similarity(self, liked_movie_indices, target_index, similarity_matrix):
        self.logger.info(f"Calculating similarity score for target movie index {target_index}.")
        try:
            similarity_scores = similarity_matrix[target_index, liked_movie_indices]
            avg_similarity_score = np.mean(similarity_scores)
            self.logger.info(
                f"Similarity scores for target movie index {target_index}: {similarity_scores}. "
                f"Average similarity: {avg_similarity_score:.4f}."
            )
            return avg_similarity_score
        except IndexError as e:
            self.logger.error(f"Index error during similarity calculation: {str(e)}", exc_info=True)
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error during similarity calculation: {str(e)}", exc_info=True)
            return 0.0

    def generate_recommendations(self, liked_movies, movies_df, similarity_matrix):
        liked_movie_indices = [self.all_movie_ids.index(movie) for movie in liked_movies if movie in self.all_movie_ids]
        if not liked_movie_indices:
            self.logger.warning("No liked movies found in the dataset.")
            return []

        recommendations = []
        for movie_id in movies_df:
            if movie_id in self.all_movie_ids:
                target_index = self.all_movie_ids.index(movie_id)
                avg_similarity_score = self.compute_cosine_similarity(liked_movie_indices, target_index, similarity_matrix)
                recommendations.append({
                    "movie_id": movie_id,
                    "title": self.all_movie_titles.get(movie_id, "Unknown Title"),
                    "average_similarity": avg_similarity_score
                })

        self.logger.info(f"Generated similarity scores for {len(recommendations)} movies.")

        recommendations = sorted(recommendations, key=lambda x: x['average_similarity'], reverse=True)

        self.logger.info(f"Filtered out liked movies. {len(recommendations)} recommendations ready.")

        return recommendations