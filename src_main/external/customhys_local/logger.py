import logging
import sys
import os
from rich.logging import RichHandler

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

def prep_prefix(prefix):
    prefix = prefix.strip()
    if prefix != "":
        prefix = " [ " + prefix + " ]"
    return prefix

def default_formatter (prefix=""):
    prefix = prep_prefix(prefix)
    return CustomFormatter(
        fmt='[ %(name)s  %(asctime)s  %(levelname)-8s ]' + prefix + ' %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )

def rich_formatter (prefix=""):
    prefix = prep_prefix(prefix)
    return CustomFormatter(
        fmt='[ ' + prefix + ' ] %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )

def loggerSTDOUT(name):
    formatter = default_formatter()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)

    return logger

def loggerRICH(name):
    formatter = rich_formatter()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.addHandler(RichHandler(level="NOTSET"))

    return logger

def loggerShutdown(logger):
    logger.warn("Shutdown logger with Handlers:")
    logger.warn(logger.handlers[:])

    if logger.hasHandlers():
        for handler in logger.handlers[:]: 
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)

    logger.handlers.clear()

def logger(name, outfolder, print_stdout=False, rich_handler=False, disabled=False, prefix=""):
    os.makedirs(outfolder, exist_ok=True)
    outputfile = os.path.join(outfolder, 'log_'+str(name)+'.txt')
    formatter = default_formatter(prefix=prefix)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.disabled = disabled
    logger.handlers.clear()

    if not disabled:
        handler = logging.FileHandler(outputfile, mode='w')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if (print_stdout):
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)

    if rich_handler:
        rich_handler = RichHandler(level="NOTSET")
        rich_handler.setFormatter(rich_formatter(prefix=prefix))
        logger.addHandler(rich_handler)

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
