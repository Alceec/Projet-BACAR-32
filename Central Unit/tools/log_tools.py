import logging
from os import path

def create_logger(filename="default.log", scriptname="%(name)s"):
    logging.basicConfig(filename=filename,filemode='w',level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - ' + scriptname + ' - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    logging.info('logger created [%s]'%path.abspath(filename))


if __name__=='__main__':

    create_logger()
