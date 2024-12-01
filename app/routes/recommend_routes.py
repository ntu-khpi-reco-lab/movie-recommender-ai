from flask import Blueprint, jsonify, request
from app.services.movie_recommender import MovieRecommender
from app.services.movie_trainer import MovieTrainer
from app.services.model_saver import MovieSaver
from app.utils.logger_config import get_logger
import requests


recommend_routes = Blueprint('recommend_routes', __name__)

logger = get_logger("RecoEndpoint")

API_URL="http://localhost:9002/api/v1/favorites"

@recommend_routes.route("/recommend", methods=["POST"])
def recommend():
    """
    Handles a recommendation request based on the user's movie preferences.

    1. Fetches the IDs of movies liked by the user via an external API.
    2. Loads the movie similarity model.
    3. Generates a list of recommended movies based on the user's liked movies and the similarity matrix.

    :return: A JSON response with the recommended movies or an error message.
        - In case of success, returns a list of recommended movies.
        - In case of failure, returns an error message in JSON format.
    """
    logger.info("Received a recommendation request")
    saver = MovieSaver()
    trainer = MovieTrainer()

    try:
        data = request.json
        logger.info(f"Request data: {data}")

        user_id = data.get("user_id")
        # response = requests.get(API_URL)
        # data = response.json()
        #
        for item in data:
            print(item)

        if not user_id:
            logger.warning("User ID not provided in the request")
            return jsonify({"message": "User ID is required"}), 400

        logger.info(f"Fetching liked movies for user_id: {user_id}")
        response = requests.get(f"{API_URL}/{user_id}")
        if response.status_code != 200:
            logger.error(f"Failed to fetch data from API: {response.status_code}, {response.text}")
            return jsonify({"message": "Failed to fetch user favorites"}), 500

        response_data = response.json()
        print(f"response_data {response_data}")
        liked_movie_ids = response_data
        logger.info(f"Liked movie IDs: {liked_movie_ids}")

        movies_df = trainer.load_movies_from_db()
        model_path = "./movie_similarity_model.pkl"
        loaded_similarity_matrix = saver.load_model(model_path)

        if movies_df is None:
            logger.error("Movies dataframe is None")
            return jsonify({"message": "Movie data not found"}), 500
        if loaded_similarity_matrix is None:
            logger.error("Similarity matrix is None")
            return jsonify({"message": "Model not found"}), 500

        logger.info("Generating movie recommendations")
        recommender = MovieRecommender(movies_df)
        recommendations = recommender.generate_recommendations(liked_movie_ids, movies_df, loaded_similarity_matrix)

        logger.info("Displaying top 20 recommendations.")
        recommendation_logs = "\n".join(
            [f"Movie ID: {rec['movie_id']}, Similarity: {rec['average_similarity']:.2f}"
             for rec in recommendations[:20]]
        )
        logger.info(f"Top 20 Recommendations:\n{recommendation_logs}")

        logger.info("Recommendations generated successfully")
        return jsonify(recommendations)


    except requests.RequestException as api_error:
        logger.error(f"API request error: {str(api_error)}")
        return jsonify({"error": f"API error: {str(api_error)}"}), 500
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
