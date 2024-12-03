from flask import Blueprint, jsonify
from app.services.movie_trainer import MovieTrainer
from app.utils.logger_config import get_logger

train_routes = Blueprint('train_routes', __name__)

logger = get_logger("TrainEndpoint")

@train_routes.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint to train the movie recommendation model.

    This route handles the process of training the recommendation model. It loads the movies
    from the database, trains the model using the data, and then saves the trained model
    to a file. The response includes a success message or an error message if something goes wrong.

    1. Load movies from the database.
    2. Train the recommendation model using the movie's data.
    3. Save the trained model to a file.

    :return: JSON response with a success or error message.
    """
    logger.info("Training process started.")
    try:
        trainer = MovieTrainer()

        logger.info("Loading movies from database.")
        movies_df = trainer.load_movies_from_db()

        if movies_df is None or movies_df.empty:
            logger.warning("No movies found in the database.")
            return jsonify({'message': 'No movies found in the database.'}), 400

        logger.info("Movies loaded successfully, starting model training.")
        trainer.train_model(movies_df)

        model_path = "movie_similarity_model.pkl"
        logger.info(f"Saving model to {model_path}.")
        trainer.save_model(model_path)

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
