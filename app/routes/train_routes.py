from flask import Blueprint, jsonify
from app.services.movie_trainer import MovieTrainer
from app.services.model_saver import MovieSaver
import logging

train_routes = Blueprint('train_routes', __name__)

logger = logging.getLogger("TrainEndpoint")

@train_routes.route('/train', methods=['POST'])
def train_model():
    try:
        trainer = MovieTrainer()
        saver = MovieSaver()

        logger.info("Loading movies from database...")
        movies_df = trainer.load_movies_from_db()

        logger.info("Training model...")
        similarity_matrix = trainer.train_model(movies_df)

        model_path = "movie_similarity_model.pkl"
        logger.info(f"Saving model to {model_path}...")
        saver.save_model(similarity_matrix, model_path)

        return jsonify({
            'message': 'Model retrained successfully',
            'model_path': model_path
        }), 200

    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        return jsonify({
            'message': f"An error occurred: {e}"
        }), 500
