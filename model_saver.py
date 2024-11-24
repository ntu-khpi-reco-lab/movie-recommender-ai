import joblib
import logging


class MovieSaver:
    def __init__(self):
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        logger = logging.getLogger('MovieSaver')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def save_model(model, model_path):
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path):
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
