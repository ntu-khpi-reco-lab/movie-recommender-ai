from flask import Blueprint, jsonify, request
from app.services.movie_recommender import MovieRecommender
from app.services.movie_trainer import MovieTrainer
from app.services.model_saver import MovieSaver
from app.utils.logger_config import get_logger
import requests


recommend_routes = Blueprint('recommend_routes', __name__)

logger = get_logger("RecoEndpoint")

@recommend_routes.route("/recommend", methods=["POST"])
def recommend():
    """
    Handles a recommendation request based on the user's movie preferences.

    The endpoint receives a POST request with two lists: likedMovieIds and movieIds.
    Based on the provided movie IDs, it calculates the similarity score between the
    user's liked movies and a target set of movies.

    1. Parses the request data to extract likedMovieIds and movieIds.
    2. Loads movie data and the pre-trained movie similarity model.
    3. For each movie in movieIds, calculates the average similarity score with the likedMovieIds.
    4. Returns a list of predictions with movie IDs and their corresponding similarity scores.

    :return: A JSON response with the recommended movies or an error message.
        - In case of success, returns a list of recommended movies.
        - In case of failure, returns an error message in JSON format.
    """
    logger.info("Received a recommendation request")
    saver = MovieSaver()
    trainer = MovieTrainer()

    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")

        liked_movie_ids = data.get("likedMovieIds")
        logger.info(f"Liked movie IDs: {liked_movie_ids}")

        movies_df = trainer.load_movies_from_db()
        model_path = "./movie_similarity_model.pkl"
        loaded_similarity_matrix = saver.load_model(model_path)

        if loaded_similarity_matrix is None:
            logger.error("Similarity matrix is None")
            return jsonify({"message": "Model not found"}), 500

        logger.info("Generating movie recommendations")
        recommender = MovieRecommender(movies_df)
        recommendations = recommender.generate_recommendations(liked_movie_ids, loaded_similarity_matrix)

        logger.info("Recommendations generated successfully")
        return jsonify(recommendations)

    except requests.RequestException as api_error:
        logger.error(f"API request error: {str(api_error)}")
        return jsonify({"error": f"API error: {str(api_error)}"}), 500
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
