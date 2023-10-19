import logging

logging.basicConfig(filename='logfile.txt', level=logging.DEBUG, format='%(asctime)s - %(message)s')

logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")