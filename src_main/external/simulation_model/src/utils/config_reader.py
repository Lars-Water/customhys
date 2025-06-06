import os
import type_enforced
from pathlib import Path, PosixPath
from src_main.tools import logger as logger
import json
import errno
import pathlib
import configparser
import shutil

@type_enforced.Enforcer
class Config:
    logger = None
    defaultFile: PosixPath = Path(
        str(pathlib.Path(__file__).parent.resolve())+"/default.cfg"
    )
    configFilePath: PosixPath = None
    _config = None

    def __init__(self, configFilePath: PosixPath, outputfolderpath = None, name: str = None, disabledLoggers: bool=False):
        if outputfolderpath is not None and name is not None:
            self.logger = logger.logger(name, outputfolderpath, disabled=disabledLoggers)
        elif name is not None:
            self.logger = logger.loggerRICH(name, disabled=disabledLoggers)
        else:
            self.logger = logger.loggerRICH("Config", disabled=disabledLoggers)

        self.logger.info("Default config file: "+str(self.defaultFile))
        if os.path.exists(configFilePath) and os.path.isfile(configFilePath):
            self.configFilePath = configFilePath
            self.logger.info("Using config file: "+str(self.configFilePath))
        else:
            self.configFilePath = self.defaultFile
        self.readConfig(self.configFilePath)

    def createLogger(self, outputfolderpath: PosixPath, name: str):
        self.logger = logger.logger(name, outputfolderpath)
        self.logger.info("Init logger later then init()")
        self.logger.info("Reread config file.")
        self.readConfig(self.configFilePath)

    def readConfig(self, configFilePath: PosixPath = defaultFile):
        self.logger.info("Loading config file: "+str(configFilePath))
        if os.path.exists(configFilePath) and os.path.isfile(configFilePath):
            f = open(configFilePath)
            self._config = json.load(f)
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                configFilePath
            )
    
    # Custom code by Herget. Close Logging file handlers manually since otherwise we open to many files.
    def closeLoggerFileHandlers(self):
        logger.loggerShutdown(self.logger)

    def conf(self):
        if self._config == None:
            raise Exception("Config was not loaded.")
        return self._config

    def tryGet(self, *args):
        conf = self._config
        try:
            for arg in args:
                conf = conf[arg]
            return conf
        except Exception as err:
            self.logger.warn("Could not get: "+ str(args))
            self.logger.exception(str(err))
            return None

if __name__ == "__main__":
    config = Config(Path("../config.cfg"), Path("../../tmp"))
    config.logger.debug(config.conf()["MockupReader"])
