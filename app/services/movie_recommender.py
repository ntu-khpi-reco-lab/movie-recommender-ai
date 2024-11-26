import numpy as np
from app.utils.logger_config import get_logger


class MovieRecommender:
    """
    A class responsible for generating movie recommendations based on cosine similarity scores.
    It calculates similarity between movies based on user preferences (liked movies) and
    generates a list of recommended movies accordingly.
    """
    def __init__(self, movies_df):
        """
        Initializes the MovieRecommender instance with the provided movie dataset.

        The initialization process extracts all movie IDs and titles from the dataset and stores them
        in memory for use during recommendation generation.

        :param movies_df: A DataFrame containing movie details, including "_id" and "title".
        """
        self.logger = get_logger("MovieTrainer")
        self.logger.info("Initializing MovieRecommender.")
        self.all_movie_ids = movies_df["_id"].tolist()
        self.all_movie_titles = movies_df.set_index("_id")["title"].to_dict()
        self.logger.info(f"Loaded {len(self.all_movie_ids)} movies into the recommender.")

    def compute_cosine_similarity(self, liked_movie_indices, target_index, similarity_matrix):
        """
        Computes the average cosine similarity score between the target movie and a list of liked movies.

        This method calculates the cosine similarity between the target movie and the liked movies,
        and returns the average similarity score.

        :param liked_movie_indices: List of indices representing the liked movies.
        :param target_index: The index of the target movie for which similarity is being calculated.
        :param similarity_matrix: The matrix containing pairwise cosine similarities between all movies.
        :return: The average similarity score between the target movie and the liked movies.
        """
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
        """
        Generates a list of recommended movies based on the cosine similarity of liked movies.

        This method computes the similarity scores between liked movies and all other movies,
        sorting them based on the highest average similarity to generate recommendations.

        :param liked_movies: List of movie titles that the user has liked.
        :param movies_df: DataFrame containing movie details (including "_id" and "title").
        :param similarity_matrix: The matrix containing pairwise cosine similarities between all movies.
        :return: A list of recommended movies sorted by their average similarity score.
        """
        self.logger.info(f"Computing cosine similarity for {len(liked_movies)} liked movies.")
        liked_movie_indices = [self.all_movie_ids.index(movie) for movie in liked_movies if movie in self.all_movie_ids]
        print(liked_movie_indices)
        if not liked_movie_indices:
            self.logger.warning("No liked movies found in the dataset.")
            return []

        recommendations = []
        for _, row in movies_df.iterrows():
            movie_id = row["_id"]
            if movie_id in self.all_movie_ids:
                target_index = self.all_movie_ids.index(movie_id)
                print(f"tar {target_index}")
                avg_similarity_score = self.compute_cosine_similarity(liked_movie_indices,
                                                                      target_index, similarity_matrix)
                print(f"avg {avg_similarity_score}")
                recommendations.append({
                    "movie_id": movie_id,
                    "title": row["title"],
                    "average_similarity": avg_similarity_score
                })

        self.logger.info(f"Generated similarity scores for {len(recommendations)} movies.")

        recommendations = sorted(recommendations, key=lambda x: x['average_similarity'], reverse=True)

        self.logger.info(f"Filtered out liked movies. {len(recommendations)} recommendations ready.")

        return recommendations