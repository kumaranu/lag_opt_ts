import logging


def get_logger(logger_name, log_file=None, log_mode='a'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)

    # add file handler if log_file is provided
    if log_file:
        fh = logging.FileHandler(log_file, mode=log_mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
