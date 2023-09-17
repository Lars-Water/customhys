import logging
import sys
import os

logging.Formatter.format

class CustomFormatter(logging.Formatter):
    # def formatDebug(self, exc_info):
    #     """
    #     Format an exception so that it prints on a single line.
    #     """
    #     result = super().formatException(exc_info)
    #     return repr(result)  # or format into one line however you want to

    def format(self, record):
        s = super().format(record)
        # Include function name and line number for debug messages for better information
        if record.levelname == 'DEBUG':
            tmp_line = s.split("]")
            tmp = tmp_line[0]+"] [l." + str(record.lineno) + " | " + str(record.funcName) + "()]"
            s = s.replace(tmp_line[0] + "]", tmp)
        # Do proper formatting of multiline logging lines: Print empty brackets to start equally
        lengthSpace = len(s.split("]")[0][1:])
        s = s.replace(
            '\n',
            '\n[' + ( lengthSpace * ' ' ) + ']  '
        )
        return s


def logger(name, outfolder, print_stdout=False):
    outputfile = os.path.join(outfolder, 'log_'+str(name)+'.txt')
    formatter = CustomFormatter(
        fmt='[ %(name)s  %(asctime)s  %(levelname)-8s ]  %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S'
    )

    handler = logging.FileHandler(outputfile, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if (print_stdout):
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    return logger


def setLevelLogger(logger, debugLevel):
    dbglvl = logger.getEffectiveLevel()
    if debugLevel.upper() == "CRITICAL":
        dbglvl = logging.CRITICAL
    elif debugLevel.upper() == "ERROR":
        dbglvl = logging.ERROR
    elif debugLevel.upper() == "WARNING":
        dbglvl = logging.WARNING
    elif debugLevel.upper() == "INFO":
        dbglvl = logging.INFO
    elif debugLevel.upper() == "DEBUG":
        dbglvl = logging.DEBUG
    logger.setLevel(dbglvl)
    return logger
