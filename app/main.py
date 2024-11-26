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
    logger.info("Starting Flask server for movie recommender.")
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()

#  Invoke-WebRequest -Uri http://127.0.0.1:5000/train -Method POST
#  Invoke-RestMethod -Uri "http://127.0.0.1:5000/recommend" -Method POST -ContentType "application/json" -Body '{"user_id": 1}'