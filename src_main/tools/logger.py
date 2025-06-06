import logging
import sys
import os
from rich.logging import RichHandler
import multiprocessing
import logging.handlers

# Global queue for multiprocessing logging
_log_queue = None
_listener = None


class CentralizedQueueListener(logging.handlers.QueueListener):
    """
    A custom QueueListener that allows adding handlers after initialization.
    This is necessary because the standard QueueListener only accepts handlers
    at construction time.
    """
    def addHandler(self, handler):
        """
        Adds a handler to the listener.
        The internal 'handlers' attribute is a tuple, so we create a new one.
        """
        self.handlers = self.handlers + (handler,)


def setup_multiprocess_logging(log_path):
    """
    Initializes a queue and a listener for centralized logging from multiple processes.
    This should be called ONCE from the main process before any child processes are spawned.
    """
    global _log_queue, _listener
    if _listener is not None:
        return # Already configured

    _log_queue = multiprocessing.Queue(-1)

    # We use our custom listener that allows adding handlers on the fly.
    # It's initialized with no handlers. They will be added as loggers are created.
    _listener = CentralizedQueueListener(_log_queue)
    _listener.start()


def stop_multiprocess_logging():
    """
    Stops the queue listener. Should be called at the end of the main process.
    """
    global _listener
    if _listener:
        _listener.stop()
        _listener = None


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
        fmt='[%(processName)-10s] [%(name)s  %(asctime)s  %(levelname)-8s ]' + prefix + ' %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )

def rich_formatter (prefix=""):
    prefix = prep_prefix(prefix)
    return CustomFormatter(
        fmt='[%(processName)-10s]' + prefix + ' %(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )

def loggerSTDOUT(name, prefix=""):
    formatter = default_formatter(prefix=prefix)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)

    return logger

def loggerRICH(name, prefix=""):
    formatter = rich_formatter(prefix=prefix)

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
    """
    Configures and returns a logger.
    This function is process-aware and configures the logger differently based on
    whether it's in a single-process, a multiprocessing main process, or a child process.
    """
    global _log_queue, _listener
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Clear existing handlers

    # Case 1: Child process of a multiprocessing application.
    # All log messages are sent to the central queue.
    if _log_queue is not None and multiprocessing.current_process().name != 'MainProcess':
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        logger.addHandler(queue_handler)
        return logger

    # For the main process or single-process cases, create the handlers.
    os.makedirs(outfolder, exist_ok=True)
    outputfile = os.path.join(outfolder, 'log_'+str(name)+'.txt')
    formatter = default_formatter(prefix=prefix)
    rich_formatter_inst = rich_formatter(prefix=prefix)
    logger.disabled = disabled

    # Prepare handlers
    file_h = logging.FileHandler(outputfile, mode='w') if not disabled else None
    if file_h:
        file_h.setFormatter(formatter)

    stream_h = logging.StreamHandler(stream=sys.stdout) if print_stdout else None
    if stream_h:
        stream_h.setFormatter(formatter)

    rich_h = RichHandler(level="NOTSET") if rich_handler else None
    if rich_h:
        rich_h.setFormatter(rich_formatter_inst)

    # Case 2: Main process of a multiprocessing application.
    # Handlers are added to the central listener, not the logger itself.
    if _listener is not None:
        if file_h:
            _listener.addHandler(file_h)
        if stream_h:
            _listener.addHandler(stream_h)
        if rich_h:
            _listener.addHandler(rich_h)
    # Case 3: Single-process application.
    # Handlers are added directly to the logger.
    else:
        if file_h:
            logger.addHandler(file_h)
        if stream_h:
            logger.addHandler(stream_h)
        if rich_h:
            logger.addHandler(rich_h)

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
