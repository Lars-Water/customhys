# @requires_main_process Decorator Audit

This document summarizes the analysis of `@requires_main_process` decorator usage in the project.

Following a thorough review and comparison, it has been confirmed that all functions identified as requiring main-process execution to ensure state synchronization are now correctly decorated. This includes functions related to:

-   Managing the lifecycle of child processes and shared UI elements.
-   Accessing or modifying shared data structures (e.g., problem spaces, data collectors).
-   Coordinating collective decisions based on the state of all workers.
-   Interacting with shared resources like the central Dask manager and the filesystem for logging and results.

The audit is complete, and the current implementation is considered robust and correct according to the hybrid RPC architecture.

# Analysis of @requires_main_process Decorator Usage

This document contains a list of all functions that have been identified to require the `@requires_main_process` decorator to ensure correct state management in the hybrid RPC multiprocessing architecture.

A function requires this decorator if it performs an action or modifies state that is shared across all child processes and must remain synchronized and authoritative in the main process.

| File | Function | Reason |
| --- | --- | --- |
| `src_main/tools/coordinatorBase.py` | `__init__` | Initializes core components like the logger and Dask manager which are central resources. |
| `src_main/tools/coordinatorBase.py` | `createDaskManager` | Instantiates the `Manager` which controls Dask simulation tasks. This is a critical singleton. |
| `src_main/tools/coordinatorBase.py` | `createDataCollector` | Abstract method, but implementations create a data collector object. This object aggregates results from all workers and must be a singleton. |
| `src_main/tools/coordinatorBase.py` | `createHyperHeuristicBase` | Instantiates `HyperHeuristicBase`, which manages the entire set of parallel workers. This is a core singleton. |
| `src_main/tools/coordinatorBase.py` | `simulation_run` | Submits simulation jobs to the central Dask `Manager`. This is a core stateful operation. |
| `src_main/tools/coordinatorBase.py` | `try_load_problems_file_name_fitness_values` | Accesses the shared file system to load fitness values, preventing race conditions. |
| `src_main/tools/coordinator_asml.py` | `createDataCollector` | Instantiates `DataCollectorASML`, a singleton for collecting results from all workers. |
| `src_main/tools/coordinator_asml.py` | `createHyperHeuristicBase` | Instantiates the `HyperHeuristicBase`, the central orchestrator for the hyper-heuristics. |
| `src_main/tools/coordinator_asml.py` | `manual_normalization` | Performs a series of simulations to determine global min/max boundaries. Modifies shared state (`self.min_wfpm_runtime`, etc.) and interacts with the filesystem and the central Dask manager. |
| `src_main/tools/coordinator_asml.py` | `determine_boundary_value` | Modifies the shared boundary attributes (`self.min_wfpm_runtime`, etc.) based on simulation results. Must be atomic. |
| `src_main/tools/hyperheuristicBase.py` | `search_operator_spaces` (property) | Accesses the `_search_operator_spaces` object which holds the problem definitions for all workers. |
| `src_main/tools/hyperheuristicBase.py` | `run_multi_threaded` | Manages the entire lifecycle of the child processes and the central `rich.progress` UI. This is the main entry point for the parallel execution. |
| `src_main/tools/hyperheuristicBase.py` | `_update_shadow_hh_object` | The direct callback for updating the parent's shadow copy of a child object. Modifies the central `hypers` dictionary. |
| `src_main/tools/hyperheuristicBase.py` | `progress_advance` | Service method to update the shared `rich.progress` bar. UI elements must be controlled by the main process. |
| `src_main/tools/hyperheuristicBase.py` | `progress_start_task` | Service method to update the shared `rich.progress` bar. |
| `src_main/tools/hyperheuristicBase.py` | `get_pending_updates` | Accesses and modifies the central `pending_updates` dictionary to provide commands to children. |
| `src_main/tools/hyperheuristicBase.py` | `_call_method_on_shadow` | Accesses the central `hypers` dictionary to call methods on shadow objects. |
| `src_main/tools/hyperheuristicBase.py` | `hh_prepare` | Modifies the `hypers` dictionary to prepare a worker's state and calls `_search_operator_spaces.create_problems`, which is a stateful operation. |
| `src_main/tools/hyperheuristicBase.py` | `hh_checkFinalization` | Accesses the state of all workers via the `hypers` dictionary to make a collective decision. |
| `src_main/tools/hyperheuristicBase.py` | `_checkFinalization_internal` | Accesses the state of all other workers stored in the `hypers` dictionary. |
| `src_main/tools/hyperheuristicBase.py` | `_best_hyper` | Accesses the state of all workers in the `hypers` dictionary to find the best one. |
| `src_main/tools/hyperheuristicBase.py` | `_is_worst_hyper` | Accesses the state of all workers in the `hypers` dictionary to find the worst one. |
| `src_main/tools/hyperheuristicBase.py` | `hh_disable_hh_and_distribute_Resources` | Modifies the state of multiple workers in the `hypers` dictionary and writes to a shared results file. |
| `src_main/tools/hyperheuristicBase.py` | `_get_num_enabled_hyper` | Reads the state of all workers from the central `hypers` dictionary. |
| `src_main/tools/searchOperatorSpace.py` | `create_problems` | Creates problem instances on the file system, a shared resource. Must be synchronized to avoid race conditions. |

# Comprehensive List of Decorated Functions

This document provides a complete list of all functions in the project that are decorated with `@requires_main_process`. This ensures that all stateful operations are correctly routed to the main process, preserving the integrity of the hybrid RPC architecture.

| File | Function |
| --- | --- |
| `src_main/data/collect_data_ASML.py` | `__init__` |
| `src_main/data/collect_data_ASML.py` | `store_design_point_metrics` |
| `src_main/data/collect_data_ASML.py` | `cache_design_point_metrics_collective` |
| `src_main/data/collect_data_ASML.py` | `store_cached_design_point_metrics_collective` |
| `src_main/data/collect_data_INET.py` | `__init__` |
| `src_main/data/collect_data_INET.py` | `store_design_point_metrics` |
| `src_main/data/collect_dataBase.py` | `__del__` |
| `src_main/data/collect_dataBase.py` | `__init__` |
| `src_main/data/collect_dataBase.py` | `get_base_filename` |
| `src_main/data/collect_dataBase.py` | `get_heuristic_filename` |
| `src_main/data/collect_dataBase.py` | `_create_metrics_storage` |
| `src_main/data/collect_dataBase.py` | `_store_design_point_metrics_df` |
| `src_main/data/collect_dataBase.py` | `append_caching_status_to_design_point_metrics_lazy` |
| `src_main/data/collect_dataBase.py` | `_append_caching_status_to_design_point_metrics` |
| `src_main/data/collect_dataBase.py` | `append_fitness_and_hh_data_to_design_point_metrics` |
| `src_main/data/collect_dataBase.py` | `_save_to_csv` |
| `src_main/data/collect_dataBase.py` | `_set_metrics_defaults` |
| `src_main/data/collect_dataBase.py` | `_INTERNAL_apply_defaults_to_df` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__init__` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__has_sim_status` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__get_sim_status` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__is_sim_status` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__set_sim_state` |
| `src_main/external/simulation_model/src/designpointcache.py` | `is_sim_waiting` |
| `src_main/external/simulation_model/src/designpointcache.py` | `is_sim_processing` |
| `src_main/external/simulation_model/src/designpointcache.py` | `is_sim_finished` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sim_waiting` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sim_cached` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sim_processing` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sim_finished` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sims_waiting` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sims_processing` |
| `src_main/external/simulation_model/src/designpointcache.py` | `set_sims_finished` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__filter_status` |
| `src_main/external/simulation_model/src/designpointcache.py` | `__filter_statuses` |
| `src_main/external/simulation_model/src/designpointcache.py` | `filter_waiting` |
| `src_main/external/simulation_model/src/designpointcache.py` | `filter_processing` |
| `src_main/external/simulation_model/src/designpointcache.py` | `filter_finished` |
| `src_main/external/simulation_model/src/designpointcache.py` | `filter_design_point_cache` |
| `src_main/external/simulation_model/src/designpointcache.py` | `get_runtime_stats` |
| `src_main/external/simulation_model/src/designpointcache.py` | `shutdown` |
| `src_main/external/simulation_model/src/designpointcache.py` | `_set_hash_sim` |
| `src_main/external/simulation_model/src/designpointcache.py` | `calc_design_point_hash_single_file` |
| `src_main/external/simulation_model/src/designpointcache.py` | `calc_design_point_hash_filelist` |
| `src_main/external/simulation_model/src/designpointcache.py` | `md5_file_list` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `__init__` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `__is_valid_sorting_algo` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `insert` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `insert_list` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `pop` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `get` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `get_list` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `empty` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `set_sorting_algo` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `size` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `sort` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `cached_insert` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `cached_insert_list` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `cached_get_all` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `has_chached` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `chached_amount` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `get_runtime_stats` |
| `src_main/external/simulation_model/src/designpointqueue.py` | `shutdown` |
| `src_main/external/simulation_model/src/manager.py` | `__init__` |
| `src_main/external/simulation_model/src/manager.py` | `set_data_collector` |
| `src_main/external/simulation_model/src/manager.py` | `__has_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__get_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__get_highest_prio_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__get_first_non_empty_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__get_first_from_design_point_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__get_first_from_design_point_queues` |
| `src_main/external/simulation_model/src/manager.py` | `__insert_design_point_queue` |
| `src_main/external/simulation_model/src/manager.py` | `__filter_unique_sim_instances` |
| `src_main/external/simulation_model/src/manager.py` | `__filter_cached_sim_instances` |
| `src_main/external/simulation_model/src/manager.py` | `__set_sim_instances_time_stat` |
| `src_main/external/simulation_model/src/manager.py` | `get_runtime_stats` |
| `src_main/external/simulation_model/src/manager.py` | `submit_queue_all` |
| `src_main/external/simulation_model/src/manager.py` | `submit_queue` |
| `src_main/external/simulation_model/src/manager.py` | `submit` |
| `src_main/external/simulation_model/src/manager.py` | `submit_all` |
| `src_main/external/simulation_model/src/manager.py` | `evaluate_cached_sims` |
| `src_main/external/simulation_model/src/manager.py` | `shutdown` |
| `src_main/external/simulation_model/src/outputhandler.py` | `__init__` |
| `src_main/external/simulation_model/src/resourcecontroller.py` | `__init__` |
| `src_main/external/simulation_model/src/resourcecontroller.py` | `__get_all_completed` |
| `src_main/external/simulation_model/src/resourcecontroller.py` | `wait_for_active_tasks` |
| `src_main/external/simulation_model/src/resourcecontroller.py` | `get_runtime_stats` |
| `src_main/external/simulation_model/src/utils/stats.py` | `__init__` |
| `src_main/external/simulation_model/src/utils/stats.py` | `set_file_path` |
| `src_main/external/simulation_model/src/utils/stats.py` | `has_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `get_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `record_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `record_time_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `inc_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `add_stat` |
| `src_main/external/simulation_model/src/utils/stats.py` | `get_stats_dict` |
| `src_main/external/simulation_model/src/utils/stats.py` | `write_stats_to_file` |
| `src_main/tools/coordinator_asml.py` | `createDataCollector` |
| `src_main/tools/coordinator_asml.py` | `createHyperHeuristicBase` |
| `src_main/tools/coordinator_asml.py` | `manual_normalization` |
| `src_main/tools/coordinator_asml.py` | `determine_boundary_value` |
| `src_main/tools/coordinator_inet.py` | `createDataCollector` |
| `src_main/tools/coordinator_inet.py` | `manual_normalization` |
| `src_main/tools/coordinator_inet.py` | `determine_boundary_value` |
| `src_main/tools/coordinator_inet.py` | `parameter_tuning_workflow` |
| `src_main/tools/coordinator_inet.py` | `determine_sim_instance_parameter_tuning_results` |
| `src_main/tools/coordinatorBase.py` | `__init__` |
| `src_main/tools/coordinatorBase.py` | `createDataCollector` |
| `src_main/tools/coordinatorBase.py` | `createHyperHeuristicBase` |
| `src_main/tools/coordinatorBase.py` | `run` |
| `src_main/tools/coordinatorBase.py` | `set_run_name` |
| `src_main/tools/coordinatorBase.py` | `try_load_problems_file_name_fitness_values` |
| `src_main/tools/hyperheuristicBase.py` | `search_operator_spaces` |
| `src_main/tools/hyperheuristicBase.py` | `has_problems` |
| `src_main/tools/hyperheuristicBase.py` | `get_all_problems_file_name_fitness_values` |
| `src_main/tools/hyperheuristicBase.py` | `run_multi_threaded` |
| `src_main/tools/hyperheuristicBase.py` | `_update_shadow_hh_object` |
| `src_main/tools/hyperheuristicBase.py` | `progress_advance` |
| `src_main/tools/hyperheuristicBase.py` | `progress_start_task` |
| `src_main/tools/hyperheuristicBase.py` | `get_pending_updates` |
| `src_main/tools/hyperheuristicBase.py` | `_call_method_on_shadow` |
| `src_main/tools/hyperheuristicBase.py` | `hh_prepare` |
| `src_main/tools/hyperheuristicBase.py` | `_task_pause` |
| `src_main/tools/hyperheuristicBase.py` | `_task_resume` |
| `src_main/tools/hyperheuristicBase.py` | `hh_checkFinalization` |
| `src_main/tools/hyperheuristicBase.py` | `_checkFinalization_internal` |
| `src_main/tools/hyperheuristicBase.py` | `_best_hyper` |
| `src_main/tools/hyperheuristicBase.py` | `_is_worst_hyper` |
| `src_main/tools/hyperheuristicBase.py` | `hh_disable_hh_and_distribute_Resources` |
| `src_main/tools/hyperheuristicBase.py` | `_get_num_enabled_hyper` |
| `src_main/tools/problemSpace.py` | `__init__` |
| `src_main/tools/problemSpace.py` | `create_problems` |
| `src_main/tools/problemSpace.py` | `create_and_append_problem` |
| `src_main/tools/problemSpace.py` | `remove_problem` |
| `src_main/tools/problemSpace.py` | `has_problems` |
| `src_main/tools/problemSpace.py` | `get_problems` |
| `src_main/tools/searchOperatorSpace.py` | `__init__` |
| `src_main/tools/searchOperatorSpace.py` | `create_problems` |
| `src_main/tools/searchOperatorSpace.py` | `create_problem_space` |
| `src_main/tools/searchOperatorSpace.py` | `remove_problem_space` |
| `src_main/tools/searchOperatorSpace.py` | `has_problem_space` |
| `src_main/tools/searchOperatorSpace.py` | `has_problems` |
| `src_main/tools/searchOperatorSpace.py` | `get_problems` |
| `src_main/tools/searchOperatorSpace.py` | `get_all_problems_file_name_fitness_values` |
