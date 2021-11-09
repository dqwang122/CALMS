import sys
import logging


class Log(object):
    @staticmethod
    def getLogger(name, filepath=None):
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)  # 也可以直接给formatter赋值
        console_handler.setLevel(logging.INFO)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)
        if filepath is not None:
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

