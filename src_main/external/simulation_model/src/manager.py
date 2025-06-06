import json
import os
import time
import threading
from multiprocessing import Process, Queue
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

from src_main.tools.local_parallelization.rpc import requires_main_process, callable_from_main, managed_by_main_process

@managed_by_main_process
class Manager:
    logger = None
    cnf = None
    start_time = None
    stats = None

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
        
        
    def set_data_collector(self, data_collector):
        self.data_collector = data_collector
        
    def __has_queue(self, queue_id):
        return queue_id in self.design_point_queues.keys()

    def __get_queue(self, queue_id):
        if self.__has_queue(queue_id):
            return self.design_point_queues[queue_id]
        else:
            self.logger.info("Queue for given id does not exist")
            return None

    def __get_highest_prio_queue(self):
        return self.design_point_queues[self.design_queue_id_order[0]]

    def __get_first_non_empty_queue(self):
        for design_point_queue_id in self.design_queue_id_order:
            design_point_queue = self.design_point_queues[design_point_queue_id]
            if design_point_queue.size() != 0:
                return design_point_queue_id

        return None

    def __get_first_from_design_point_queue(self, queue_id):
        if self.__has_queue(queue_id):
            return self.design_point_queues[queue_id]
        else:
            raise Exception("Passed invalid design queue id")

    def __get_first_from_design_point_queues(self):
        design_point_queue_id = self.__get_first_non_empty_queue()
        if design_point_queue:
            design_point_queue = self.__get_queue(design_point_queue_id)
            return design_point_queue.pop()
        else:
            return None

    def __insert_design_point_queue(self, sim_instances, queue_id):
        design_point_queue = self.__get_queue(queue_id)

        if design_point_queue:
            self.logger.info("Inserting sim instances ({}) in design point queue {} (Total: {})".format(len(sim_instances), queue_id, design_point_queue.size()+len(sim_instances)))
            self.sim_queue.insert_list(sim_instances)

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

    def __filter_cached_sim_instances(self, sim_instances):
        num_received = len(sim_instances)
        sim_instances, cached_sim_instances = self.design_point_cache.filter_design_point_cache(sim_instances)
        num_not_in_cache = len(sim_instances)
        self.logger.info("Filtered {} sim instances based on cache".format(num_received - num_not_in_cache))
        return sim_instances, cached_sim_instances

    def set_sim_instances_time_stat(self, sim_instances, *args):
        sims = []
        for sim_instance in sim_instances:
            sim_instance.record_time_stat(*args)
            sims += [sim_instance]
        return sims

    def get_runtime_stats(self):
        return self.stats.get_stats_dict()

    def enqueue_tasks(self, sim_instances, queue_id=None, metadata=None):
        self.logger.debug("Manager received sim instances ({})".format(len(sim_instances)))
        self.stats.add_stat(len(sim_instances), "general", "num_dp")
        sim_instances = self.set_sim_instances_time_stat(sim_instances, "general", "environment_entry")

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
            sim_instances = self.set_sim_instances_time_stat(sim_instances, "general", "insert_design_point_queue_start")
            design_point_queue.insert_list(sim_instances)
            sim_instances = self.set_sim_instances_time_stat(sim_instances, "general", "insert_design_point_queue_end")
            self.design_point_cache.set_sims_waiting(sim_instances)
        else:
            self.logger.debug("No sim instances to enqueue")

        self.stats.write_stats_to_file()

    # Does not wait for results
    def submit_queue_all(self, queue_id):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        n = design_point_queue.size()

        return self.submit_queue(queue_id, n=n)

    # Does not wait for result
    def submit_queue(self, queue_id, n=1):
        design_point_queue = self.__get_queue(queue_id)

        if not design_point_queue:
            raise Exception("Passed invalid design queue id")

        self.logger.debug("Requesting {} sim instances from design point queue {}".format(n, queue_id))
        sim_instances = design_point_queue.get(n)
        self.logger.debug("Recieved {} sim instances from design point queue {}".format(len(sim_instances), queue_id))
        self.stats.add_stat(len(sim_instances), "general", "num_submit_rc")

        self.logger.debug("Sending sim instances ({}) to resource controller".format(len(sim_instances)))
        sim_instances = self.set_sim_instances_time_stat(sim_instances, "general", "manager_submit_resource_controller_start")
        self.resource_controller.submit_tasks(sim_instances)
        sim_instances = self.set_sim_instances_time_stat(sim_instances, "general", "manager_submit_resource_controller_end")

        self.logger.debug("Setting sim instances ({}) to processing state in cache".format(len(sim_instances)))
        self.design_point_cache.set_sims_processing(sim_instances)

        return sim_instances

    # Does not wait for result
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
    def submit_all(self):
        self.logger.debug("Requesting all sim instances from design point queues (highest priority first)")
        sim_instances = []

        for design_point_queue_id in self.design_queue_id_order:
            queue_sim_instances = self.submit_queue_all(design_point_queue_id)
            sim_instances += queue_sim_instances

        return sim_instances

    def get_queue_object(self, queue_id):
        """Returns the actual queue object for a given ID."""
        return self.__get_queue(queue_id)

    def is_queue_empty_and_finished(self, queue_id):
        """Checks if a specific queue is empty and the resource controller has no active tasks for it."""
        # This is a simplified check. A more robust implementation might need
        # the resource controller to track tasks per queue_id.
        return self.__get_queue(queue_id).size() == 0 and not self.resource_controller.has_active_tasks()

    def get_all_completed_from_queue(self, queue_id):
        """
        Retrieves all completed simulation instances.
        This method relies on the resource controller to have a way to provide these.
        """
        # This assumes the resource controller can return all completed tasks.
        # This might need a new method in ResourceController like `get_completed_tasks`.
        completed = self.resource_controller.get_all_completed()
        
        # You might need to filter these results based on queue_id if the
        # resource controller manages tasks from multiple queues at once.
        return completed

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

    def _wait_and_put_results(self, sim_instances, result_queue):
        """
        This function runs in a background thread in the main manager process.
        It blocks until the Dask tasks are complete, then puts the results
        into the queue provided by the child process.
        """
        try:
            self.logger.debug(f"Background thread started, waiting for {len(sim_instances)} tasks.")
            completed = self.resource_controller.wait_for_tasks(sim_instances)
            self.logger.debug(f"Background thread: tasks finished, putting {len(completed)} results into queue.")
            result_queue.put(completed)
        except Exception as e:
            self.logger.exception(f"Exception in manager's background result thread: {e}")
            # Put the exception in the queue so the child process can see it
            result_queue.put(e)

    def evaluate_tasks_async(self, sim_instances, result_queue):
        """
        A non-blocking call for worker processes. It submits tasks and
        spawns a background thread to wait for the results and put them
        in the provided queue.
        """
        self.logger.debug(f"Manager received async evaluation request for {len(sim_instances)} instances.")
        
        # Set processing state in the main process
        self.design_point_cache.set_sims_processing(sim_instances)
        
        # Submit tasks non-blockingly to Dask
        self.resource_controller.submit_tasks(sim_instances)
        
        # Start a background thread to wait for the results
        wait_thread = threading.Thread(
            target=self._wait_and_put_results,
            args=(sim_instances, result_queue)
        )
        wait_thread.daemon = True # Allows main program to exit even if thread is running
        wait_thread.start()

    def evaluate_queue_all(self, queue_id):
        """
        A blocking, synchronous method to evaluate all simulations in a given
        queue. This provides a consistent interface with ManagerChild for use
        when running directly in the main process.
        """
        design_point_queue = self.__get_queue(queue_id)
        if not design_point_queue:
            raise Exception(f"Passed invalid design queue id: {queue_id}")

        n = design_point_queue.size()
        sim_instances = design_point_queue.get(n)
        if not sim_instances:
            return []

        # Set state to processing
        self.design_point_cache.set_sims_processing(sim_instances)
        
        # Submit tasks before waiting for them
        self.resource_controller.submit_tasks(sim_instances)

        # Submit and wait for tasks to complete
        completed_sim_instances = self.resource_controller.wait_for_tasks(sim_instances)

        # Process output and set final state
        completed_sim_instances = self.output_handler.clean_sims(completed_sim_instances)
        self.design_point_cache.set_sims_finished(completed_sim_instances)

        return completed_sim_instances

# TODO: add dequeue, wait_for, get_results, etc. functions