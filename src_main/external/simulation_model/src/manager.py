import json
import os
import time

from multiprocessing import Process
from pathlib import Path, PosixPath
from types import SimpleNamespace

from src.designpointqueue import DesignPointQueue
from src.siminstance import Siminstance
from src.resourcecontroller import ResourceController
from src.outputhandler import OutputHandler
from src.designpointcache import DesignPointCache

from src_main.tools.logger import logger
from src.utils.config_reader import Config
from src.utils.stats import Stats

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main

class Manager:
    logger = None
    cnf = None
    start_time = None
    stats = None

    @requires_main_process
    def __init__(self, config_path, logs_path, stats_file=None, data_collector = None):
        if stats_file is None:
            stats_file = os.path.join(logs_path, "statistics.json")
        self.stats_file = stats_file
        self.stats = Stats(file_path=self.stats_file)
        self.stats.record_time_stat("general", "environment_start")
        self.start_time = time.time()
        os.makedirs(logs_path, exist_ok=True)
        self.logger = logger("manager", Path(logs_path), disabled=False)

        self.logger.info("Reading in config file: {}".format(config_path))
        self.cnf = Config(Path(config_path), Path(logs_path), "config_manager")
        self.num_task_completed = 0


        self.logger.info("Creating design point queue(s)")
        self.design_point_queues = {}
        self.data_collector = data_collector

        if self.cnf.tryGet("design_point_queues"):
            for design_point_queue_id in self.cnf.tryGet("design_point_queues").keys():
                self.logger.info("Creating design point queue with id: {}".format(design_point_queue_id))
                self.design_point_queues[design_point_queue_id] = DesignPointQueue(config_path, logs_path, design_point_queue_id)
        else:
            raise Exception("No design point queue present in config")

        # A higher priority number will be in front
        design_queue_ids = self.design_point_queues.keys()
        design_queue_prios = [self.design_point_queues[design_queue_id].priority for design_queue_id in design_queue_ids]

        design_queue_id_priority_order = sorted(list(zip(design_queue_ids, design_queue_prios)), key=lambda x: x[1], reverse=True)

        self.design_queue_id_order = [design_queue_id for design_queue_id, design_queue_prio in design_queue_id_priority_order]

        self.logger.info("Design point queue order (highest priority first):")
        for design_queue_id, design_queue_prio in design_queue_id_priority_order:
            self.logger.info("{} with priority {}".format(design_queue_id, design_queue_prio))

        self.logger.info("Creating resource controller")
        self.resource_controller = ResourceController(config_path, logs_path)

        self.logger.info("Creating output handler")
        self.output_handler = OutputHandler(config_path, logs_path)

        self.logger.info("Creating sim cache")
        self.design_point_cache = DesignPointCache(config_path, logs_path)

    def set_stats_file(self, stats_file):
        self.stats_file = stats_file
        self.stats.set_file_path(self.stats_file)
        
        
    @requires_main_process
    def set_data_collector(self, data_collector):
        self.data_collector = data_collector
        
    @requires_main_process
    def __has_queue(self, queue_id):
        return queue_id in self.design_point_queues.keys()

    @requires_main_process
    def __get_queue(self, queue_id):
        if self.__has_queue(queue_id):
            return self.design_point_queues[queue_id]
        else:
            self.logger.info("Queue for given id does not exist")
            return None

    @requires_main_process
    def __get_highest_prio_queue(self):
        return self.design_point_queues[self.design_queue_id_order[0]]

    @requires_main_process
    def __get_first_non_empty_queue(self):
        for design_point_queue_id in self.design_queue_id_order:
            design_point_queue = self.design_point_queues[design_point_queue_id]
            if design_point_queue.size() != 0:
                return design_point_queue_id

        return None

    @requires_main_process
    def __get_first_from_design_point_queue(self, queue_id):
        if self.__has_queue(queue_id):
            return self.design_point_queues[queue_id]
        else:
            raise Exception("Passed invalid design queue id")

    @requires_main_process
    def __get_first_from_design_point_queues(self):
        design_point_queue_id = self.__get_first_non_empty_queue()
        if design_point_queue:
            design_point_queue = self.__get_queue(design_point_queue_id)
            return design_point_queue.pop()
        else:
            return None

    @requires_main_process
    def __insert_design_point_queue(self, sim_instances, queue_id):
        design_point_queue = self.__get_queue(queue_id)

        if design_point_queue:
            self.logger.info("Inserting sim instances ({}) in design point queue {} (Total: {})".format(len(sim_instances), queue_id, design_point_queue.size()+len(sim_instances)))
            self.sim_queue.insert_list(sim_instances)

    @requires_main_process
    def __filter_unique_sim_instances(self, sim_instances):
        num_received = len(sim_instances)

        unique = []
        unique_uids = []
        for sim_instance in sim_instances:
            if not sim_instance.uid in unique_uids:
                unique.append(sim_instance)
                unique_uids.append(sim_instance.uid)

        num_unique = len(unique)
        self.logger.info("Filtered {} duplicate sim instances".format(num_received - num_unique))

        return unique

    @requires_main_process
    def __filter_cached_sim_instances(self, sim_instances):
        num_received = len(sim_instances)
        sim_instances, cached_sim_instances = self.design_point_cache.filter_design_point_cache(sim_instances)
        num_not_in_cache = len(sim_instances)
        self.logger.info("Filtered {} sim instances based on cache".format(num_received - num_not_in_cache))
        return sim_instances, cached_sim_instances

    @requires_main_process
    def __set_sim_instances_time_stat(self, sim_instances, *args):
        sims = []
        for sim_instance in sim_instances:
            sim_instance.record_time_stat(*args)
            sims += [sim_instance]
        return sims

    @requires_main_process
    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def enqueue_tasks(self, sim_instances, queue_id=None, metadata=None):
        self.logger.debug("Manager received sim instances ({})".format(len(sim_instances)))
        self.stats.add_stat(len(sim_instances), "general", "num_dp")
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "environment_entry")

        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            design_point_queue = self.__get_highest_prio_queue()

        sim_instances = self.__filter_unique_sim_instances(sim_instances)
        sim_instances, cached_sim_instances = self.__filter_cached_sim_instances(sim_instances)
        self.stats.add_stat(len(sim_instances), "general", "num_dp_unique")
        self.stats.add_stat(len(cached_sim_instances), "general", "num_dp_cached")
        self.logger.warn("num_dp_cached+=" + str(len(cached_sim_instances))+" (total: "+str(self.stats.get_stat("general", "num_dp_cached"))+")")
        
        if metadata is not None:
            if type(metadata) == dict:
                metadata = self._combineDictToList(metadata)
            if type(metadata) == list:
                self.stats.add_stat(len(cached_sim_instances), "metadata", *metadata, "num_dp_cached")
                self.stats.add_stat(len(sim_instances), "metadata", *metadata, "num_dp_unique")

        if (len(cached_sim_instances) > 0):
            design_point_queue.cached_insert_list(cached_sim_instances)
        else:
            self.logger.debug("No cached sim instances to process")

        if (len(sim_instances) > 0):
            self.logger.debug("Inserting sim instances ({}) in design point queue {}".format(len(sim_instances), queue_id))
            sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "insert_design_point_queue_start")
            design_point_queue.insert_list(sim_instances)
            sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "insert_design_point_queue_end")
            self.design_point_cache.set_sims_waiting(sim_instances)
        else:
            self.logger.debug("No sim instances to enqueue")

        self.stats.write_stats_to_file()

    # Does not wait for results
    @requires_main_process
    def submit_queue_all(self, queue_id):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        n = design_point_queue.size()

        return self.submit_queue(queue_id, n=n)

    # Does not wait for result
    @requires_main_process
    def submit_queue(self, queue_id, n=1):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        self.logger.debug("Requesting {} sim instances from design point queue {}".format(n, queue_id))
        sim_instances = design_point_queue.get(n)
        self.logger.debug("Recieved {} sim instances from design point queue {}".format(len(sim_instances), queue_id))
        self.stats.add_stat(len(sim_instances), "general", "num_submit_rc")

        self.logger.debug("Sending sim instances ({}) to resource controller".format(len(sim_instances)))
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "manager_submit_resource_controller_start")
        self.resource_controller.submit_tasks(sim_instances)
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "manager_submit_resource_controller_end")

        self.logger.debug("Setting sim instances ({}) to processing state in cache".format(len(sim_instances)))
        self.design_point_cache.set_sims_processing(sim_instances)

        return sim_instances

    # Does not wait for result
    @requires_main_process
    def submit(self, n):
        self.logger.debug("Requesting {} sim instances from design point queues (highest priority first)".format(n))
        sim_instances = []

        for design_point_queue_id in self.design_queue_id_order:
            if n <= 0:
                self.logger.debug("All sim instances extracted from design point queues")
                break

            queue_sim_instances = self.submit_queue(design_point_queue_id, n)
            sim_instances += queue_sim_instances
            n = n - len(queue_sim_instances)

        return sim_instances

    # Does not wait for results
    @requires_main_process
    def submit_all(self):
        self.logger.debug("Requesting all sim instances from design point queues (highest priority first)")
        sim_instances = []

        for design_point_queue_id in self.design_queue_id_order:
            queue_sim_instances = self.submit_queue_all(design_point_queue_id)
            sim_instances += queue_sim_instances

        return sim_instances

    # Waits for results
    def evaluate_queue_all(self, queue_id):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        n = design_point_queue.size()

        sim_instances = self.evaluate_queue(queue_id, n=n)

        # self.logger.warn(f"Evaluatring queue {queue_id}: {design_point_queue.has_chached()} {str(design_point_queue.chached_amount())}")

        # Function returns true if all cached are currently processing, hence we wait until all are done
        i = 0
        while self.evaluate_cached_sims(design_point_queue) and design_point_queue.has_chached():
            self.logger.warn(f"+++ {queue_id} Waiting {i}")
            time.sleep(5)           
            i += 1     
        return sim_instances
    
    @requires_main_process
    def evaluate_cached_sims(self, design_point_queue):
        all_are_processing = 0
        cached_sim_instances = design_point_queue.cached_get_all()
        if self.data_collector:
            self.data_collector.append_caching_status_to_design_point_metrics_lazy([sim_tuple[0].uid for sim_tuple in cached_sim_instances], True)
        
        for current_sim_instance, original_sim_uid in cached_sim_instances:
            original_cached_sim_placeholder = SimpleNamespace(uid=original_sim_uid)

            if self.design_point_cache.is_sim_finished(original_cached_sim_placeholder):
                self.output_handler.copy_sim_results(current_sim_instance, original_cached_sim_placeholder)
                self.design_point_cache.set_sim_finished(current_sim_instance)

            elif self.design_point_cache.is_sim_processing(original_cached_sim_placeholder):
                design_point_queue.cached_insert((current_sim_instance, original_sim_uid))
                all_are_processing += 1
                
            else:
                self.logger.warn(f"Original cached sim {original_sim_uid} for {current_sim_instance.uid} is in an unexpected state. Re-queuing.")
                design_point_queue.cached_insert((current_sim_instance, original_sim_uid))

        return (all_are_processing == design_point_queue.chached_amount() and design_point_queue.has_chached())

    # Waits for results
    def evaluate_queue(self, queue_id, n=1):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        self.logger.debug("Requesting {} sim instances from design point queue {}".format(n, queue_id))
        sim_instances = design_point_queue.get(n)
        self.logger.debug("Recieved {} sim instances from design point queue {}".format(len(sim_instances), queue_id))

        # TODO: move this to the resource_controller?
        self.logger.debug("Setting sim instances ({}) to processing state in cache".format(len(sim_instances)))
        self.design_point_cache.set_sims_processing(sim_instances)
        self.stats.add_stat(len(sim_instances), "general", "num_eval_rc")

        self.logger.debug("Sending sim instances ({}) to resource controller".format(len(sim_instances)))
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "manager_evaluate_resource_controller_start")
        sim_instances = self.resource_controller.evaluate_tasks(sim_instances)
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "manager_evaluate_resource_controller_end")

        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "output_handler_start")
        sim_instances = self.output_handler.clean_sims(sim_instances)
        sim_instances = self.__set_sim_instances_time_stat(sim_instances, "general", "output_handler_end")
        self.design_point_cache.set_sims_finished(sim_instances)
        self.stats.add_stat(len(sim_instances), "general", "num_dp_finished")
        self.logger.debug("num_dp_finished+=" + str(len(sim_instances))+" (total: "+str(self.stats.get_stat("general", "num_dp_finished"))+")")

        cached = self.evaluate_cached_sims(design_point_queue)      

        return sim_instances

    # Waits for results
    def evaluate(self, n=1):
        self.logger.debug("Requesting {} sim instances from design point queues (highest priority first)".format(n))
        sim_instances = []

        for design_point_queue_id in self.design_queue_id_order:
            if n <= 0:
                self.logger.debug("All sim instances extracted from design point queues")
                break

            queue_sim_instances = self.evaluate_queue(design_point_queue_id, n)
            sim_instances += queue_sim_instances
            n = n - len(queue_sim_instances)

        return sim_instances

    # Waits for results
    def evaluate_all(self):
        self.logger.info("Requesting all sim instances from design point queues (highest priority first)")
        sim_instances = []

        for design_point_queue_id in self.design_queue_id_order:
            queue_sim_instances = self.evaluate_queue_all(design_point_queue_id)
            sim_instances += queue_sim_instances

        return sim_instances

    @requires_main_process
    def shutdown(self):
        self.design_point_cache.shutdown()
        design_point_runtime = self.design_point_cache.get_runtime_stats()
        self.output_handler.output_design_point_cache_runtime(design_point_runtime)

        for queue in self.design_point_queues.values():
            queue.shutdown()
            queue_runtime = queue.get_runtime_stats()
            self.output_handler.output_design_point_queue_runtime(queue.id, queue_runtime)

        self.resource_controller.shutdown()
        resource_controller_runtime = self.resource_controller.get_runtime_stats()
        self.output_handler.output_resource_controller_runtime(resource_controller_runtime)

        self.stats.record_time_stat("general", "environment_end")
        self.stats.record_stat(self.stats.get_stat("general", "environment_end") - self.stats.get_stat("general", "environment_start"), "general", "environment_time")

        manager_runtime = self.get_runtime_stats()
        self.output_handler.output_manager_runtime(manager_runtime)

        output_handler_runtime = self.output_handler.get_runtime_stats()
        self.output_handler.output_output_handler_runtime(output_handler_runtime)

        self.output_handler.shutdown()

        end_time = time.time()
        print("Started at {}".format(time.strftime("%H:%M:%S %d/%m/%Y", time.gmtime(self.start_time))))
        print("Number of design points received: {}".format(self.stats.get_stat("general", "num_dp") or 0))
        print("Number of unique design points received: {}".format(self.stats.get_stat("general", "num_dp_unique") or 0))
        print("Number of unique design points processed: {}".format(self.stats.get_stat("general", "num_dp_finished") or 0))
        print("Number of design points which were already cached: {}".format(self.stats.get_stat("general", "num_dp_cached") or 0))
        print("Throughput unique design points: {:.3f} evaluations per second".format((self.stats.get_stat("general", "num_dp_finished") or 0)/(end_time - self.start_time)))
        print("Shutdown at {}".format(time.strftime("%H:%M:%S %d/%m/%Y", time.gmtime(end_time))))

        self.stats.write_stats_to_file()
    
    def _combineDictToList(self, metadata, delimiter="_", oldkey=None):
        ret_list = []
        for key in metadata.keys():
            val = metadata[key]
            if oldkey is not None:
                key = str(oldkey)+delimiter+str(key)
            if type(val) == dict:
                val = self._combineDictToList(val, delimiter, key)
                ret_list += val
            elif type(val) == list:
                val = [str(key)+delimiter+str(x) for x in val]
                ret_list += val
            elif val == None:
                ret_list.append(str(key))
            else:
                ret_list.append(str(key)+delimiter+str(val))
        return ret_list

# TODO: add dequeue, wait_for, get_results, etc. functions