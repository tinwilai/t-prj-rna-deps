import logging
import sys

logger = logging.getLogger("T_prj.rna")
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(stream_handler)


null_logger = logging.Logger("null")
null_logger.addHandler(logging.NullHandler())
