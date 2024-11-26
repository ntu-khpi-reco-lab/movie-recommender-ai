from flask import Blueprint, jsonify
from app.services.movie_trainer import MovieTrainer
from app.services.model_saver import MovieSaver
from app.utils.logger_config import get_logger

train_routes = Blueprint('train_routes', __name__)

logger = get_logger("TrainEndpoint")

@train_routes.route('/train', methods=['POST'])
def train_model():
    logger.info("Training process started.")
    try:
        trainer = MovieTrainer()
        saver = MovieSaver()

        logger.info("Loading movies from database.")
        movies_df = trainer.load_movies_from_db()

        if movies_df is None or movies_df.empty:
            logger.warning("No movies found in the database.")
            return jsonify({'message': 'No movies found in the database.'}), 400

        logger.info("Movies loaded successfully, starting model training.")
        similarity_matrix = trainer.train_model(movies_df)

        if similarity_matrix is None:
            logger.error("Failed to train model, similarity matrix is None.")
            return jsonify({'message': 'Failed to train model.'}), 500

        model_path = "movie_similarity_model.pkl"
        logger.info(f"Saving model to {model_path}.")
        saver.save_model(similarity_matrix, model_path)

        logger.info("Model retrained and saved successfully.")
        return jsonify({
            'message': 'Model retrained successfully',
            'model_path': model_path
        }), 200

    except Exception as e:
        logger.error(f"Error occurred during training: {e}", exc_info=True)
        return jsonify({
            'message': f"An error occurred: {e}"
        }), 500
