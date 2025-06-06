import json
import os

from pathlib import Path, PosixPath

from src_main.tools.logger import logger
from src.utils.config_reader import Config
from src.utils.stats import Stats

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main, managed_by_main_process

@managed_by_main_process
class DesignPointQueue:
    id = None
    priority = None
    logger = None
    cnf = None
    queue = None
    queue_cached = []
    sorting_algo = None
    stats = None

    supported_sorting_algos = ["FIFO"]

    def __init__(self, config_path, logs_path, id):
        self.id = id
        # TODO: validate that 'id' can be used in file/folder names

        self.stats = Stats()

        self.logs_path = logs_path
        os.makedirs(self.logs_path, exist_ok=True)

        self.logger = logger("design_point_queue_{}".format(self.id), Path(self.logs_path), disabled=False)

        self.logger.info("ID of design point queue: {}".format(self.id))

        self.logger.info("Reading in config file: " + str(config_path))
        self.cnf = Config(Path(config_path), Path(self.logs_path), "config_design_point_queue_{}".format(self.id))

        self.queue = []
        self.queue_cached = []

        if (self.cnf.tryGet("design_point_queues")):
            self.logger.info("Design point queues config data present in config file")

            if (self.cnf.tryGet("design_point_queues", self.id)):
                self.priority = self.cnf.tryGet("design_point_queues", self.id, "priority")
                sorting_algo = self.cnf.tryGet("design_point_queues", self.id, "sorting_algo")
                self.logger.info("Design point queue has priority {}".format(self.priority))
                self.logger.info("Design point queue has sorting algorithm {}".format(sorting_algo))
                self.set_sorting_algo(sorting_algo)
            else:
                self.logger.warn("No design point queue config present in config file for id: {}".format(self.id))
                raise Exception("No design point queue config present in config file for id: {}".format(self.id))
        else:
            self.logger.warn("No design point queues config data present in config file")
            raise Exception("No design point queues config data present in config file")

    def __is_valid_sorting_algo(self, sorting_algo):
        self.logger.info("Checking validity of sorting algorithm param: {}".format(sorting_algo))
        if (sorting_algo == "FIFO"):
            self.logger.info("{} is a valid sorting algorithm".format(sorting_algo))
            return True
        else:
            return False

    def insert(self, design_point):
        if (self.sorting_algo == "FIFO"):
            self.queue.append(design_point)
            self.logger.debug("Inserting design point at the back of the queue")
        else:
            pass

        self.stats.inc_stat("general", "num_dp_queued")

    def insert_list(self, design_points):
        if (self.sorting_algo == "FIFO"):
            self.queue += design_points
            self.logger.debug("Inserting design point at the back of the queue")
        else:
            pass

        self.stats.inc_stat("general", "num_dp_queued")

    def pop(self):
        if (self.sorting_algo == "FIFO"):
            self.logger.debug("Popping design point at the front of the queue")
            return self.queue.pop(0)
        else:
            pass

        self.stats.inc_stat("general", "num_dp_dequeued")

    def get(self, n):
        if self.size() < n:
            n = self.size()

        self.stats.add_stat(n, "general", "num_dp_dequeued")
        if (self.sorting_algo == "FIFO"):
            self.logger.debug("Popping {} design points from the front of the queue".format(n))
            design_points = self.queue[:n]
            self.queue = self.queue[n:]
            return design_points
        else:
            pass


    def get_list(self):
        return self.queue

    def empty(self):
        current_queue = self.queue
        self.queue = []

        self.stats.add_stat(len(current_queue), "general", "num_dp_dequeued")

        return current_queue

    def set_sorting_algo(self, new_algo):
        if (self.__is_valid_sorting_algo(new_algo)):
            self.sorting_algo = new_algo
            self.logger.debug("Setting design point queue sorting algorithm to: {}".format(self.sorting_algo))
            self.sort()
        else:
            self.logger.warn("Received invalid new sorting algorithm. Not changing the current sorting algorithm: {}".format(self.sorting_algo))

    def size(self):
        self.logger.debug("The current size of the design point queue: {}".format(len(self.queue)))
        return len(self.queue)

    def sort(self):
        self.logger.debug("Sorting the queue according to the sorting algorithm: {}".format(self.sorting_algo))
        if (self.sorting_algo == "FIFO"):
            self.logger.debug("{} sorting algorithm does not change the current order of the queue".format(self.sorting_algo))
            return
        else:
            pass

    def cached_insert(self, design_point):
        if (self.sorting_algo == "FIFO"):
            self.queue_cached.append(design_point)
        else:
            pass

    def cached_insert_list(self, design_points):
        if (self.sorting_algo == "FIFO"):
            self.queue_cached += design_points
        else:
            pass

    def cached_get_all(self):
        tmp = self.queue_cached[:]
        self.queue_cached = []
        return tmp
    
    def has_chached(self):
        return self.chached_amount() != 0

    def chached_amount(self):
        return len(self.queue_cached)

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def shutdown(self):
        return
