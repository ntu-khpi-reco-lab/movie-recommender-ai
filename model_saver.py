import joblib
from logger_config import get_logger


class MovieSaver:
    def __init__(self):
        self.logger = get_logger("MovieSaver")

    def save_model(self, model, model_path):
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return model
