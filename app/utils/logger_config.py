import logging

LOG_FORMAT = '%(name)s - %(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'logger.log'


def get_logger(logger_name="application"):
    """
    Configures and returns a logger instance with both file and console handlers.

    The logger is set up with a specific name and ensures that log messages are formatted
    and written to both a log file and the console. If a logger with the same name
    already exists, it avoids adding duplicate handlers.

    :param logger_name: (str) The name of the logger. Defaults to "application".
    :return: (logging.Logger) A configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
