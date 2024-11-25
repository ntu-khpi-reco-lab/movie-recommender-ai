import joblib
from logger_config import get_logger


class MovieSaver:
    def __init__(self):
        self.logger = get_logger("MovieSaver")

    def save_model(self, model, model_path):
        try:
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {model_path}: {str(e)}", exc_info=True)

    def load_model(self, model_path):
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {model_path}.")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
        return None
