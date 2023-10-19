import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러 생성 및 설정
file_handler = logging.FileHandler('logfile.txt')
file_format = logging.Formatter('%(asctime)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# 콘솔 핸들러 생성 및 설정
stream_handler = logging.StreamHandler()
stream_format = logging.Formatter('%(levelname)s: %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)

# 로그 메시지 작성
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")