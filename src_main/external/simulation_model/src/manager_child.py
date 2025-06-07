import time
from types import SimpleNamespace
from dask.distributed import wait
from multiprocessing import Process
from pathlib import Path, PosixPath
from multiprocessing.managers import BaseManager

from src.outputhandler import OutputHandler
from src.designpointqueue import DesignPointQueue
from src_main.tools.logger import loggerRICH

# --- Custom Manager for Shared Queues ---
class SharedObjectManager(BaseManager):
    pass
SharedObjectManager.register('get_queue')

class ManagerChild:
    """
    A class that runs within each child process, acting as a local interface
    to the main Manager in the parent process. It handles child-specific,
    blocking operations like waiting for results and processing output,
    while communicating with the main Manager via an RPC proxy for
    centralized state management.
    """
    def __init__(self, manager_proxy, coordinator_proxy, config_path, logs_path):
        """
        Initializes the ManagerChild.

        Args:
            manager_proxy: An RPCProxy object that connects to the main Manager instance.
            coordinator_proxy: An RPCProxy for the HeuristicSimulationCoordinator, used for creating shared objects.
            config_path: Path to the configuration file, needed for the local OutputHandler.
            logs_path: Path to the logs directory, needed for the local OutputHandler.
        """
        self.manager_proxy = manager_proxy
        self.coordinator_proxy = coordinator_proxy
        self.output_handler = OutputHandler(config_path, logs_path)
        
        # Connect to the shared object manager running in the main process
        address, authkey = self.coordinator_proxy.hh_base.get_shared_manager_address_and_authkey()
        self.shared_obj_manager = SharedObjectManager(address=address, authkey=authkey)
        self.shared_obj_manager.connect()

    def get_shared_obj_manager(self):
        return self.shared_obj_manager

    def set_sim_instances_time_stat(self, sim_instances, *args):
        """A local method to iterate over sim instances and record timing stats."""
        sims = []
        for sim_instance in sim_instances:
            sim_instance.record_time_stat(*args)
            sims += [sim_instance]
        return sims

    def enqueue_tasks(self, *args, **kwargs):
        return self.manager_proxy.enqueue_tasks(*args, **kwargs)

    def evaluate_queue_all(self, queue_id):
        sim_instances = self.evaluate_queue(queue_id, n=0)

        # self.logger.warn(f"Evaluatring queue {queue_id}: {design_point_queue.has_chached()} {str(design_point_queue.chached_amount())}")

        result_queue = self.shared_obj_manager.get_queue(queue_id+"_cached_sims_queue")
        self.manager_proxy.evaluate_cached_sims_async(queue_id, result_queue)

        i = result_queue.get()
        return sim_instances

    def evaluate_queue(self, queue_id, n=1):
        """
        Gets N tasks from a queue, submits them for evaluation, waits for them
        to complete, and processes the output.
        This is the core evaluation loop for the child process.
        """

        sim_instances_to_eval = self.manager_proxy.evaluate_queue_pre(queue_id, n)


        result_queue = self.shared_obj_manager.get_queue(queue_id+"_result_queue")

        # This is a non-blocking RPC call that submits the tasks and
        # provides a queue for the main process to send the results back.
        self.manager_proxy.evaluate_tasks_async(sim_instances_to_eval, result_queue)
        
        # Block efficiently until the results are available in the queue.
        # The main process's background thread will put the results here when done.
        results = result_queue.get()
        if isinstance(results, Exception):
            raise results

        results = self.manager_proxy.evaluate_queue_post(queue_id, results)
        results = self.set_sim_instances_time_stat(results, "general", "output_handler_start")
        results = self.output_handler.clean_sims(results)
        results = self.set_sim_instances_time_stat(results, "general", "output_handler_end")

        return results

    def evaluate_all(self):
        """Evaluates all tasks from all queues."""
        sim_instances = []
        for design_point_queue_id in self.manager_proxy.design_queue_id_order:
            queue_sim_instances = self.evaluate_queue_all(design_point_queue_id)
            sim_instances += queue_sim_instances
        return sim_instances
        
    def evaluate(self, n=1):
        """Evaluates N tasks, pulling from queues based on priority."""
        sim_instances = []
        for design_point_queue_id in self.manager_proxy.design_queue_id_order:
            if n <= 0:
                break
            queue_sim_instances = self.evaluate_queue(design_point_queue_id, n)
            sim_instances += queue_sim_instances
            n -= len(queue_sim_instances)
        return sim_instances

    def evaluate_cached_sims(self, design_point_queue):
        """
        Handles the evaluation of cached simulation instances for a given queue.
        This involves checking their status with the main manager and processing
        them locally with the output handler.
        """
        # This loop is designed to handle cached items that might have been
        # waiting for an original simulation to finish.
        while design_point_queue.has_chached():
            all_are_processing = 0
            cached_sim_instances = design_point_queue.cached_get_all()

            for current_sim_instance, original_sim_uid in cached_sim_instances:
                original_cached_sim_placeholder = SimpleNamespace(uid=original_sim_uid)
                
                # Check the status of the original sim via RPC
                if self.manager_proxy.design_point_cache.is_sim_finished(original_cached_sim_placeholder):
                    # If finished, process the output locally
                    self.output_handler.copy_sim_results(current_sim_instance, original_cached_sim_placeholder)
                    self.manager_proxy.design_point_cache.set_sim_finished(current_sim_instance) # Update state in parent
                
                elif self.manager_proxy.design_point_cache.is_sim_processing(original_cached_sim_placeholder):
                    # If still processing, re-queue it and check again later
                    design_point_queue.cached_insert((current_sim_instance, original_sim_uid))
                    all_are_processing += 1
                else:
                    # If in an unexpected state, re-queue to avoid losing it
                    design_point_queue.cached_insert((current_sim_instance, original_sim_uid))

            if all_are_processing == design_point_queue.chached_amount():
                # If all remaining cached items are waiting on processing sims, we can break
                break
            time.sleep(2) 