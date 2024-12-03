from flask import Flask
from app.routes.train_routes import train_routes
from app.routes.main_routes import main_routes
from app.routes.recommend_routes import recommend_routes
from app.utils.logger_config import get_logger

app = Flask(__name__)
logger = get_logger("Main")

app.register_blueprint(main_routes)
app.register_blueprint(train_routes)
app.register_blueprint(recommend_routes)

def main():
    """
     Entry point for starting the Flask application server.

    This function initializes the Flask application, sets up logging, and
    registers route blueprints for handling different endpoints. The server
    starts in debug mode, listening on all available interfaces (0.0.0.0) and
    port 5000.

    Features:
        - `train_routes`: Handles training-related API routes.
        - `main_routes`: Handles general API routes.
        - `recommend_routes`: Handles recommendation-related API routes.
        - Logs server startup messages for better traceability.

    Example Usage:
        Run this script directly to start the Flask server:
        $ python main.py

        Test the endpoints using the following commands:
        - Train route:
            Invoke-WebRequest -Uri http://localhost:5000/train -Method POST
        - Recommend route:
            Invoke-RestMethod -Uri "http://localhost:5000/recommend" -Method POST -ContentType "application/json" -Body '{"movieIds" : [680, 105, 6, 8, 10, 13, 16], "likedMovieIds": [25, 105, 569094, 574, 6]}'

    :return: None
    """
    logger.info("Starting Flask server for movie recommender.")
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
