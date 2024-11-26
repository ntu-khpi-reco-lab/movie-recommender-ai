from flask import Blueprint

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/')
def home():
    """
    Home route for the Movie Recommender App.

    This route serves the landing page of the application. It returns a welcome message
    to the user when they visit the root URL.

    :return: A simple welcome message.
    """
    return "Welcome to the Movie Recommender App!"
