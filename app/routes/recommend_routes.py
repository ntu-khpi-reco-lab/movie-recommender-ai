from flask import Blueprint, jsonify, request
from app.services.movie_recommender import MovieRecommender
from app.services.movie_trainer import MovieTrainer
from app.services.model_saver import MovieSaver
from app.utils.logger_config import get_logger
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

load_dotenv()

recommend_routes = Blueprint('recommend_routes', __name__)

logger = get_logger("RecoEndpoint")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
}

@recommend_routes.route("/recommend", methods=["POST"])
def recommend():
    logger.info("Received a recommendation request")
    saver = MovieSaver()
    trainer = MovieTrainer()

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            logger.info("Connected to the database")
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                query = "SELECT movie_id FROM user_favorite_movies WHERE user_id = %s"
                user_id = request.json.get("user_id")
                if not user_id:
                    logger.warning(f"User ID {user_id} not provided in the request")
                    return jsonify({"message": "User ID is required"}), 400

                logger.info(f"Fetching liked movies for user_id: {user_id}")
                cursor.execute(query, (user_id,))
                liked_movie_ids = [row["movie_id"] for row in cursor.fetchall()]

                logger.debug(f"Liked movie IDs: {liked_movie_ids}")

                if not liked_movie_ids:
                    logger.warning(f"No liked movies found for user_id: {user_id}")
                    return jsonify({"message": "No liked movies found for the user"}), 404

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
            [f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}"
             for rec in recommendations[:20]]
        )
        logger.info(f"Top 20 Recommendations:\n{recommendation_logs}")

        logger.info("Recommendations generated successfully")
        return jsonify(recommendations)

    except psycopg2.Error as db_error:
        logger.error(f"Database error: {str(db_error)}")
        return jsonify({"error": f"Database error: {str(db_error)}"}), 500
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
