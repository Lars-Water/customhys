# Architecture Documentation: The Hybrid RPC and Event-Driven Model

This document details the multiprocessing architecture used in this project. It is designed to solve the challenges of state management and asynchronous execution when using `fork`-based multiprocessing, ensuring both correctness and performance.

**Note:** A comprehensive audit of all functions requiring the `@requires_main_process` decorator has been completed. The full list of decorated functions can be found in the [Decorator Audit File](./README-MultiProcessingApproach-Tags.md).

## 1. The Problem: State Fragmentation and Blocking Calls

The core of the application involves running multiple `Hyperheuristic` instances in parallel. The initial `fork`-based approach created a fundamental problem: critical singleton objects, such as the Dask `ResourceController` and the `DesignPointCache`, were duplicated in each child process. This led to a fragmented state where workers could not share caches or coordinate resource usage effectively.

A second major challenge emerged: how can a child process ask the main process to perform a long-running, blocking task (like waiting for a batch of simulations to finish on a Dask cluster) without blocking the main process itself? A naive RPC call would stall the entire application, preventing the main process from serving other workers.

## 2. The Solution: A Hybrid Model with Event-Driven Callbacks

To solve this, we developed a sophisticated hybrid architecture that combines three key concepts:
1.  **Stateful Singletons**: Critical stateful objects live exclusively in the main process.
2.  **RPC Proxy**: Child processes interact with these singletons via a transparent RPC proxy, which forwards method calls to the main process.
3.  **Event-Driven Callbacks**: For long-running tasks, children use a queue-based notification system, allowing them to wait efficiently without blocking the parent.

### Core Components

-   **`LocalParallelizationManager` (`local_parallelization_manager.py`):** The central orchestrator in the main process. It manages the lifecycle of child processes, acts as the RPC server, and hosts the `SharedObjectManager`.

-   **`SharedObjectManager` (`local_parallelization_manager.py`):** A custom manager inheriting from `multiprocessing.managers.BaseManager`. Its sole purpose is to create and manage objects (specifically, `queue.Queue` instances) that are safe to be shared between the main process and child processes *after* the children have already started. This is crucial for our event-driven pattern.

-   **`ParallelizationManagerContext` (`local_parallelization_context.py`):** The gateway object given to each child process. It provides the machinery for the child to communicate with the parent manager and create proxies.

-   **`RPCProxy` (`rpc.py`):** This is the client-side "magic." It wraps a *copy* of a parent object (or `None` if the object only exists remotely). Any attribute access or method call is intercepted and intelligently forwarded to the main process for execution.

-   **`Manager` (`manager.py`):** The main-process singleton that manages the simulation environment (queues, cache, Dask cluster).

-   **`ManagerChild` (`manager_child.py`):** A lightweight object that runs in each child process. It acts as the local interface for simulation tasks, encapsulating the logic for making asynchronous requests to the main `Manager` and waiting for the results.

### The Decorator API Summary

| Decorator                   | Location                                      | Purpose                                                                                                                                                            |
| --------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@managed_by_main_process`  | On a class definition (e.g., `Manager`).      | **Marks a class for nested proxying.** The RPCProxy will automatically create sub-proxies for any attribute that is an instance of a class decorated with this.      |
| `@requires_main_process`    | On methods/properties of a main-process object. | **Flags a method/property as stateful.** When accessed via the proxy, it triggers a blocking RPC call to the main process to be executed or fetched there.        |
| `@replicated_to_parent`     | On a class definition in the child process.   | **Enables state replication.** Wraps `__setattr__` to watch for changes on specific attributes.                                                                    |
| `@replicate('attr_name')`   | On the same class as `@replicated_to_parent`. | **Marks which attributes to watch.** Tells `@replicated_to_parent` which specific attributes should trigger an asynchronous update message to the parent when changed. |
| `@callable_from_main`       | On methods of the child's replicated object.  | **Creates a safe, read-only API.** Explicitly marks methods on the child's object that the parent process is allowed to call on its local "shadow copy".           |

## 3. Communication Patterns and Data Flow

### Pattern 1: Synchronous RPC Calls

This is used for fast, stateful actions that should complete immediately.

-   **Mechanism:** Methods like `enqueue_tasks` are decorated with `@requires_main_process`.
-   **Flow:** The child calls the method on its proxy. The proxy bundles the call, sends it to the `LocalParallelizationManager`, waits for the real method to execute on the real object, and returns the result.

### Pattern 2: Asynchronous Evaluation with Queue-based Notification

This is the elegant, non-polling solution for long-running tasks, such as waiting for a batch of simulations.

-   **Mechanism:** The `ManagerChild` in the worker process orchestrates the flow, using the custom `SharedObjectManager` to create a safe communication channel.
-   **Flow:**
    1.  The `ManagerChild` calls `create_managed_queue()` on its coordinator proxy. This RPC call tells the `SharedObjectManager` in the main process to create a new, shareable `Queue`. A proxy for this queue is safely returned.
    2.  The `ManagerChild` calls `evaluate_tasks_async()` on its `manager_proxy`, passing the list of simulations and the shareable queue object. This is a **non-blocking** RPC call.
    3.  The main `Manager` receives the request, submits the simulations to the Dask cluster, and immediately spawns a new **background thread**.
    4.  The main `Manager`'s main loop is now free to handle other requests. The background thread's only job is to block, waiting for the Dask tasks to finish (e.g., via `dask.distributed.wait`).
    5.  Meanwhile, the `ManagerChild` in the worker process makes a single, efficient call: `result_queue.get()`. This call **blocks the child process** (as intended) until something is placed in the queue.
    6.  Once the Dask tasks are complete, the background thread in the main process places the list of completed simulations into the `result_queue`.
    7.  This action instantly unblocks the `ManagerChild`'s `queue.get()` call, and the child process can continue its work with the results.

This event-driven pattern is highly efficient, avoids polling, and correctly separates the concerns of blocking between the child and parent processes.

### Pattern 3: Parent-to-Child (State Synchronization)

This pattern remains for observing and influencing the state of the child's `Hyperheuristic` object. The parent maintains a "shadow copy," updated via two mechanisms:
-   **Asynchronous Attribute Replication:** The `@replicate` decorator on the child's `Hyperheuristic` class automatically pushes changes of simple values to the parent.
-   **Periodic Full Refresh & "Pull" Updates:** The child periodically sends a full copy of its state, and can also "pull" commands from the parent by calling a service like `get_pending_updates`.

## 4. Rationale and Design Decisions

-   **Why a Custom `BaseManager`?** Passing a standard `multiprocessing.Queue` as an argument in an RPC call fails with a `RuntimeError`, as these queues can only be shared via inheritance at process creation. The `multiprocessing.Manager` provides a way to create shareable objects that can be passed between already-running processes. However, the default `Manager`'s automatic proxy object handling led to subtle `KeyError` and `TypeError: Pickling an AuthenticationString...` exceptions within our custom RPC framework. The solution was to use the more fundamental `multiprocessing.managers.BaseManager` to create our own explicit `SharedObjectManager`. This gives us full control, allowing us to safely get the manager's connection details (`address`, `authkey`) and have the child process connect to it directly, ensuring a robust and error-free connection.

-   **Why Not a Dask Client in Each Child?** Creating a full Dask `Client` in each worker process is an anti-pattern. The client is a heavyweight object with its own threads and persistent connections. Having many clients would overload the Dask scheduler and hide task management from our central `ResourceController`. The current model, with one client in the main process, is far more efficient and scalable.

## 5. Guidelines for Extension

When adding or modifying functionality, follow this "cookbook":

1.  **Does a child method need to perform a quick, blocking action on the main process?**
    -   **Action:** Add a method to `Manager` or `HeuristicSimulationCoordinatorBase` and decorate it with `@requires_main_process`. Call it via the proxy from the child. Done.

2.  **Does a child method need the main process to perform a long-running, blocking task without stalling the main process?**
    -   **Action:** Implement the event-driven queue pattern.
        1.  Add a new method to `Manager` (e.g., `my_long_task_async(..., result_queue)`).
        2.  Inside this method, spawn a background thread (`threading.Thread`) that will perform the blocking work.
        3.  The thread's target function should perform the wait and then `.put()` the result into the `result_queue`.
        4.  From the `ManagerChild`, get a shareable queue via the `coordinator_proxy` and call your new `_async` method.
        5.  Immediately call `result_queue.get()` to wait for the result.

---

## Phantom Oblivion Relay

**Notice for Human Developers:** The following section is a structured data dump intended for consumption by another AI or Large Language Model. Its purpose is to provide a highly condensed, machine-readable summary of the project's architecture to rapidly onboard another AI assistant in a future session. It is not designed to be easily readable by humans.

```json
{
  "project_id": "hyper-heuristic-dse-2.0",
  "architecture_name": "Hybrid RPC with Event-Driven Callbacks",
  "problem_domain": "State fragmentation and blocking RPC calls in fork-based multiprocessing.",
  "solution_goal": "Centralize state and control in the main process, while enabling children to trigger long-running, blocking tasks without stalling the main process.",
  "key_concepts": {
    "rpc_direction": "Unidirectional (Child-to-Parent initiated) to prevent deadlocks.",
    "state_location": "Authoritative state lives exclusively in the main process.",
    "async_task_pattern": "Event-driven (Queue-based notification). Children wait efficiently on a dedicated queue while a background thread in the parent process performs the actual blocking wait."
  },
  "core_components": [
    {
      "file": "src_main/tools/local_parallelization/local_parallelization_manager.py",
      "class": "LocalParallelizationManager",
      "role": "Main process singleton. Manages child process lifecycle, hosts the RPC server, and runs the SharedObjectManager for creating shareable objects."
    },
    {
      "file": "src_main/tools/local_parallelization/local_parallelization_manager.py",
      "class": "SharedObjectManager",
      "role": "A custom `multiprocessing.managers.BaseManager`. Its sole purpose is to create and manage objects (e.g., `queue.Queue`) that can be safely shared between processes after they have been started. This is the foundation of the event-driven callback pattern."
    },
    {
      "file": "src_main/tools/local_parallelization/local_parallelization_context.py",
      "class": "ParallelizationManagerContext",
      "role": "Gateway object within each child process. Provides the interface for a child to communicate with the parent manager and create RPC proxies."
    },
    {
      "file": "src_main/tools/local_parallelization/rpc.py",
      "class": "RPCProxy",
      "role": "Client-side proxy in the child process. Wraps a local object copy, or `None` for remote-only objects. Intercepts `__getattr__` and `__call__` to forward requests to the parent. Now supports creating proxies for objects that don't exist in the child (e.g., the main `Manager`)."
    },
    {
      "file": "src_main/external/simulation_model/src/manager.py",
      "class": "Manager",
      "role": "Main-process singleton that manages the simulation environment: Dask cluster interaction via `ResourceController`, caching via `DesignPointCache`, etc. It exposes the `evaluate_tasks_async` method for non-blocking evaluation."
    },
    {
      "file": "src_main/external/simulation_model/src/manager_child.py",
      "class": "ManagerChild",
      "role": "Lightweight object in each child process. It orchestrates the asynchronous evaluation flow by connecting to the `SharedObjectManager` to get a queue, calling `evaluate_tasks_async` on the main `Manager`'s proxy, and then blocking efficiently on `queue.get()`."
    }
  ],
  "communication_patterns": {
    "async_evaluation": {
      "pattern_name": "Asynchronous Evaluation with Queue-based Notification",
      "description": "The primary pattern for long-running tasks. A child process obtains a shareable queue from the `SharedObjectManager`, passes it to the main `Manager` in a non-blocking `evaluate_tasks_async` RPC call, and then blocks on `queue.get()`. The main `Manager` spawns a background thread to perform the blocking wait (e.g., on Dask results), and that thread puts the final result into the queue, which unblocks the child.",
      "files": ["src_main/external/simulation_model/manager.py", "src_main/external/simulation_model/manager_child.py", "src_main/tools/local_parallelization/local_parallelization_manager.py"]
    },
    "synchronous_rpc": {
      "trigger_decorator": "@requires_main_process",
      "description": "For fast, stateful actions. Marks a method so that when called on a proxy, it triggers a blocking RPC call to the parent.",
      "file": "src_main/tools/local_parallelization/rpc.py"
    },
    "state_replication_to_parent": {
      "class_decorator": "@replicated_to_parent",
      "attribute_decorator": "@replicate('attr_name')",
      "description": "Wraps a class's `__setattr__`. When a monitored attribute is changed, it triggers a non-blocking, asynchronous message to the parent to update its shadow copy.",
      "file": "src_main/tools/local_parallelization/rpc.py"
    }
  },
  "final_object_graph_in_child": {
    "entry_point": "_hh_worker_function",
    "flow": [
      "Worker receives 'context' object and a deserialized 'coordinator_copy'.",
      "Worker calls `coordinator_copy.finish_initialization_in_worker(context)`. This is the critical post-deserialization setup step.",
      "Inside `finish_initialization_in_worker`, remote-only proxies for the main `Manager` and a full proxy for the `Coordinator` are created.",
      "A `ManagerChild` instance is created inside `coordinator_copy`, receiving these proxies.",
      "The `coordinator_copy` is now fully initialized. The worker then creates an RPC proxy for it: `proxy_coordinator = context.create_proxy(coordinator_copy)`.",
      "A real, local `Hyperheuristic` object is instantiated, receiving the `proxy_coordinator`.",
      "When a simulation is needed, the `Hyperheuristic` object calls a method on `proxy_coordinator`.",
      "The `proxy_coordinator`'s method then uses its `local_manager` (the `ManagerChild` instance) to execute the asynchronous, queue-based evaluation pattern."
    ]
  },
  "operational_context": {
    "entry_points": {
      "instantiation_flow": "The entry script instantiates a `HeuristicSimulationCoordinator` subclass. This object creates the `HyperHeuristicBase`, which then creates the `LocalParallelizationManager` (which in turn starts the `SharedObjectManager` server) and spawns the child processes. The coordinator instance is the 'root_object' for all RPC calls."
    }
  },
  "potential_issues_and_todos": [
    {
      "id": "PATTERN-002",
      "type": "Pattern",
      "summary": "Event-Driven Asynchronous Tasks",
      "description": "For any new long-running task that a child needs the parent to perform, the queue-based notification pattern is the standard. A new `*_async` method should be added to the main `Manager`, which spawns a background thread to do the blocking work and put the result in a queue provided by the child."
    },
    {
      "id": "RISK-001",
      "type": "Risk",
      "summary": "Serialization of Proxies",
      "description": "Proxy objects created by `multiprocessing.managers` are not designed to be pickled and passed between processes themselves. The correct pattern is to pass the manager's connection details (address and authkey) and have the remote process connect directly, as implemented in `ManagerChild`."
    }
  ]
}
``` 