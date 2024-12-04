import numpy as np
import joblib
from app.utils.logger_config import get_logger

class MovieRecommender:
    def __init__(self):
        self.logger = get_logger("MovieTrainer")
        self.logger.info("Initializing MovieRecommender.")
        
        self.indices = None
        self.similarity_matrix = None

    def compute_cosine_similarity(self, liked_movie_indices, target_index):
        self.logger.info(f"Calculating similarity score for target movie index {target_index}.")
        try:
            similarity_scores = self.similarity_matrix[target_index, liked_movie_indices]
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

    def generate_recommendations(self, liked_movies, movie_ids):
        self.logger.info(f"Computing cosine similarity for {len(liked_movies)} liked movies.")
        liked_movie_indices = [self.indices[movie_id] for movie_id in liked_movies if movie_id in self.indices]
        print(liked_movie_indices)
        if not liked_movie_indices:
            self.logger.warning("No liked movies found in the dataset.")
            return []

        recommendations = []
        for movie_id in movie_ids:
            target_index = self.indices.get(movie_id)
            avg_similarity_score = self.compute_cosine_similarity(liked_movie_indices, target_index)
            recommendations.append({
                "movie_id": movie_id,
                "average_similarity": avg_similarity_score
            })

        self.logger.info(f"Generated similarity scores for {len(recommendations)} movies.")

        recommendations = sorted(recommendations, key=lambda x: x['average_similarity'], reverse=True)

        self.logger.info(f"Filtered out liked movies. {len(recommendations)} recommendations ready.")

        return recommendations
    
    def load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}.")
        model = joblib.load(model_path)

        print(model)

        self.indices = model["indices"]
        self.similarity_matrix = model["similarity_matrix"]

        self.logger.info(f"Model loaded from {model_path}")
        return model
