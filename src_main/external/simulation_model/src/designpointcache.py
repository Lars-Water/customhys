import os
import hashlib

from enum import Enum
from pathlib import Path

from src.utils.config_reader import Config
from src.utils.logger import logger
from src.utils.hash import md5_dir
from src.utils.stats import Stats


class SimStatus(Enum):
    WAITING = 0
    PROCESSING = 1
    FINISHED = 2
    CACHED = 3

    @staticmethod
    def list_values():
        return list(map(lambda c: c.value, SimStatus))

    @staticmethod
    def has_status(status):
        if status in SimStatus.list_values():
            return True
        else:
            False


class DesignPointCache:
    logger = None
    cnf = None
    sims_path = None
    uids_status = None
    stats = None
    cached_hashes = {}

    def __init__(self, config_path, logs_path):
        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)

        self.stats = Stats()

        self.logger = logger("sim_cache", Path(self.logs_path), disabled=False)

        self.logger.info("Reading in config file: {}".format(config_path))
        self.cnf = Config(Path(config_path), Path(self.logs_path), "config_sim_cache")

        self.sims_path = self.cnf.tryGet("sims_path")
        self.logger.info("Sims path: {}".format(self.sims_path))

        self.uids_status = {}

    def __has_sim_status(self, sim_instance):
        if sim_instance.uid in self.uids_status.keys():
            return True
        else:
            return False

    def __get_sim_status(self, sim_instance):
        if self.__has_sim_status(sim_instance):
            return self.uids_status[sim_instance.uid]
        else:
            return -1

    def __is_sim_status(self, sim_instance, sim_status):
        # self.logger.debug(f"is sim status {sim_instance.uid} // {self.__get_sim_status(sim_instance)} // {self.uids_status.keys()}")
        if self.__get_sim_status(sim_instance) == sim_status:
            return True
        else:
            return False

    def __set_sim_state(self, sim_instance, sim_status):
        if SimStatus.has_status(sim_status.value):
            self.uids_status[sim_instance.uid] = sim_status
        else:
            raise Exception("Trying to set cache sim status to invalid state")

    def is_sim_waiting(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.WAITING)

    def is_sim_processing(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.PROCESSING)

    def is_sim_finished(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.FINISHED)

    def set_sim_waiting(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.WAITING)

    def set_sim_cached(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.CACHED)

    def set_sim_processing(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.PROCESSING)

    def set_sim_finished(self, sim_instance):
        if self.cnf.tryGet("cache", "cache_only_finished_sim_instances"):
            self._set_hash_sim(sim_instance.getCacheHash(), sim_instance)
        self.__set_sim_state(sim_instance, SimStatus.FINISHED)

    def set_sims_waiting(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_waiting(sim_instance)

    def set_sims_processing(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_processing(sim_instance)

    def set_sims_finished(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_finished(sim_instance)

    def __filter_status(self, sim_instances, sim_status):
        return [sim_instance for sim_instance in sim_instances if not self.__is_sim_status(sim_instance, sim_status)]

    def __filter_statuses(self, sim_instances, sim_statuses):
        return [sim_instance for sim_instance in sim_instances if not sum([self.__is_sim_status(sim_instance, sim_status) for sim_status in sim_statuses])]

    def filter_waiting(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.WAITING)

    def filter_processing(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.PROCESSING)

    def filter_finished(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.FINISHED)

    # Filter all sim_instances which are already in cache
    def filter_design_point_cache(self, sim_instances):
        cached_sim_instances = []
        if self.cnf.tryGet("cache", "files") is not None and len(self.cnf.tryGet("cache", "files")) > 0:
            for sim_instance in sim_instances:
                hash = self.calc_design_point_hash_filelist(sim_instance, self.cnf.tryGet("cache", "files"))
                if hash not in self.cached_hashes.keys():
                    if not self.cnf.tryGet("cache", "cache_only_finished_sim_instances"):
                        self._set_hash_sim(hash, sim_instance)
                    sim_instance.setCacheHash(hash)
                    self.logger.debug(f"New simulation config found with hash: {hash} (ID: {sim_instance.uid})")
                else:
                    cached_sim_instances.append([sim_instance, self.cached_hashes[hash]])
                    self.set_sim_cached(sim_instance)


        return [sim_instance for sim_instance in sim_instances if not self.__has_sim_status(sim_instance)], \
                cached_sim_instances
               

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def shutdown(self):
        return

    def _set_hash_sim(self, hash, sim_instance):
        self.cached_hashes[hash] = sim_instance
    
    def calc_design_point_hash_single_file(self, sim_instance, filename):
        file_path = os.path.join(sim_instance.path, filename)
        hash = self.md5_file_list([file_path])
        return hash
    
    def calc_design_point_hash_filelist(self, sim_instance, filenames):
        file_paths = [os.path.join(sim_instance.path, filename) for filename in filenames]
        hash = self.md5_file_list(file_paths)
        return hash

    
    def md5_file_list(self, filenames):
        hash = hashlib.md5()
        for fn in filenames:
            try:
                hash.update(Path(fn).read_bytes())
            except IsADirectoryError:
                pass
        return hash.digest()