import os
import hashlib

from enum import Enum
from pathlib import Path

from src.utils.config_reader import Config
from src.utils.logger import logger
from src.utils.hash import md5_dir
from src.utils.stats import Stats

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main

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

    @requires_main_process
    def __init__(self, config_path, logs_path, doCache=True):
        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)

        self.stats = Stats()

        self.logger = logger("sim_cache", Path(self.logs_path), disabled=False)

        self.logger.info("Reading in config file: {}".format(config_path))
        self.cnf = Config(Path(config_path), Path(self.logs_path), "config_sim_cache")

        self.sims_path = self.cnf.tryGet("sims_path")
        self.logger.info("Sims path: {}".format(self.sims_path))

        self.uids_status = {}

        self.doCache = doCache

    @requires_main_process
    def __has_sim_status(self, sim_instance):
        return sim_instance.uid in self.uids_status.keys()

    @requires_main_process
    def __get_sim_status(self, sim_instance):
        if self.__has_sim_status(sim_instance):
            return self.uids_status[sim_instance.uid]
        else:
            return -1

    @requires_main_process
    def __is_sim_status(self, sim_instance, sim_status):
        return self.__get_sim_status(sim_instance) == sim_status

    @requires_main_process
    def __set_sim_state(self, sim_instance, sim_status):
        if SimStatus.has_status(sim_status.value):
            self.uids_status[sim_instance.uid] = sim_status
        else:
            raise Exception("Trying to set cache sim status to invalid state")

    @requires_main_process
    def is_sim_waiting(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.WAITING)

    @requires_main_process
    def is_sim_processing(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.PROCESSING)

    @requires_main_process
    def is_sim_finished(self, sim_instance):
        return self.__is_sim_status(sim_instance, SimStatus.FINISHED)

    @requires_main_process
    def set_sim_waiting(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.WAITING)

    @requires_main_process
    def set_sim_cached(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.CACHED)

    @requires_main_process
    def set_sim_processing(self, sim_instance):
        self.__set_sim_state(sim_instance, SimStatus.PROCESSING)

    @requires_main_process
    def set_sim_finished(self, sim_instance):
        if self.doCache and self.cnf.tryGet("cache", "cache_only_finished_sim_instances"):
            self._set_hash_sim(sim_instance.getCacheHash(), sim_instance)
        self.__set_sim_state(sim_instance, SimStatus.FINISHED)

    @requires_main_process
    def set_sims_waiting(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_waiting(sim_instance)

    @requires_main_process
    def set_sims_processing(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_processing(sim_instance)

    @requires_main_process
    def set_sims_finished(self, sim_instances):
        for sim_instance in sim_instances:
            self.set_sim_finished(sim_instance)

    @requires_main_process
    def __filter_status(self, sim_instances, sim_status):
        return [sim_instance for sim_instance in sim_instances if not self.__is_sim_status(sim_instance, sim_status)]

    @requires_main_process
    def __filter_statuses(self, sim_instances, sim_statuses):
        return [sim_instance for sim_instance in sim_instances if not sum([self.__is_sim_status(sim_instance, sim_status) for sim_status in sim_statuses])]

    @requires_main_process
    def filter_waiting(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.WAITING)

    @requires_main_process
    def filter_processing(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.PROCESSING)

    @requires_main_process
    def filter_finished(self, sim_instances):
        return self.__filter_status(sim_instances, SimStatus.FINISHED)

    @requires_main_process
    def filter_design_point_cache(self, sim_instances):
        cached_sim_instances = []
        non_cached_sim_instances = []

        if self.doCache and self.cnf.tryGet("cache", "files") is not None and len(self.cnf.tryGet("cache", "files")) > 0:
            for sim_instance in sim_instances:
                current_sim_hash = self.calc_design_point_hash_filelist(sim_instance, self.cnf.tryGet("cache", "files"))
                sim_instance.setCacheHash(current_sim_hash)

                if current_sim_hash not in self.cached_hashes.keys():
                    if not self.cnf.tryGet("cache", "cache_only_finished_sim_instances"):
                        self._set_hash_sim(current_sim_hash, sim_instance)
                    self.logger.debug(f"New simulation config found with hash: {current_sim_hash} (ID: {sim_instance.uid})")
                    non_cached_sim_instances.append(sim_instance)
                else:
                    original_sim_uid = self.cached_hashes[current_sim_hash]
                    cached_sim_instances.append((sim_instance, original_sim_uid))
                    self.set_sim_cached(sim_instance)
        else:
            non_cached_sim_instances = sim_instances

        return non_cached_sim_instances, cached_sim_instances
               

    @requires_main_process
    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    @requires_main_process
    def shutdown(self):
        return

    @requires_main_process
    def _set_hash_sim(self, hash, sim_instance):
        if self.doCache:
            self.cached_hashes[hash] = sim_instance.uid
    
    @requires_main_process
    def calc_design_point_hash_single_file(self, sim_instance, filename):
        if self.doCache:
            file_path = os.path.join(sim_instance.path, filename)
            hash = self.md5_file_list([file_path])
            return hash
        else:
            return None
    
    @requires_main_process
    def calc_design_point_hash_filelist(self, sim_instance, filenames):
        if self.doCache:
            file_paths = [os.path.join(sim_instance.path, filename) for filename in filenames]
            hash = self.md5_file_list(file_paths)
            return hash
        else:
            return None
    
    @requires_main_process
    def md5_file_list(self, filenames):
        if self.doCache:
            hash = hashlib.md5()
            for fn in filenames:
                try:
                    hash.update(Path(fn).read_bytes())
                except IsADirectoryError:
                    pass
            return hash.digest()
        else:
            return None