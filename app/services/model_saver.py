import joblib
from app.utils.logger_config import get_logger


class MovieSaver:
    """
    A class to handle the saving and loading of movie recommendation models using joblib.
    Provides methods to save a model to disk and load a model from a given file path.
    """
    def __init__(self):
        """
        Initializes the MovieSaver instance with a logger for tracking operations.
        """
        self.logger = get_logger("MovieSaver")

    def save_model(self, model, model_path):
        """
        Saves the trained movie recommendation model to a specified file path.

        This method uses joblib to serialize and store the model on disk, making it
        possible to reload the model later for use in predictions or evaluations.

        :param model: The model object to be saved.
        :param model_path: The file path where the model will be saved.
        """
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
        Loads a previously saved movie recommendation model from a specified file path.

        This method deserializes the model stored on disk and returns it for use.

        :param model_path: The file path from which the model will be loaded.
        :return: The loaded model object.
        """
        model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return model
