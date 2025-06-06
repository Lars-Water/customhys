# Architecture Documentation: The Hybrid Execution RPC Model

This document details the multiprocessing architecture used in this project. It is designed to solve the challenges of state management when using `fork`-based multiprocessing, ensuring both correctness and performance.

## 1. The Problem: State Fragmentation

The core of the application involves running multiple `Hyperheuristic` instances in parallel to explore different search operator spaces. The initial approach used Python's `multiprocessing.Process` to create these parallel workers.

On POSIX-compliant systems (like Linux), `multiprocessing` defaults to the `fork()` start method. When a new process is "forked," it receives a complete, copy-on-write duplicate of the parent process's memory space.

This created a fundamental problem: critical singleton objects, such as the Dask `ResourceController` and the `DesignPointCache`, were duplicated in each child process. Each child had its own independent copy, leading to a fragmented state:
-   Simulations cached by one worker were not visible to others.
-   Dask client objects became corrupted, leading to deadlocks.
-   The parent process had no visibility into the state of the child processes, making it impossible to manage them effectively (e.g., re-distribute resources).

## 2. The Solution: A Hybrid Execution RPC Model

To solve this, we developed a sophisticated hybrid architecture that combines the safety of Remote Procedure Calls (RPC) with the performance of local execution.

The core principle is: **Stateful objects live exclusively in the main process. Child processes are given a proxy object that transparently decides whether to execute a method locally on a copied object or to forward the call to the authoritative object in the main process.**

This model provides a single source of truth for all shared state, eliminating fragmentation, while minimizing the performance overhead of inter-process communication.

### Core Components

-   **`LocalParallelizationManager` (`local_parallelization_manager.py`):** The central orchestrator that runs in the main process. It manages the lifecycle of child processes and acts as the RPC server, routing requests from children to the appropriate objects.

-   **`ParallelizationManagerContext` (`local_parallelization_context.py`):** The gateway object given to each child process. It does not contain any application state but provides the machinery for the child to communicate with the parent manager.

-   **`RPCProxy` (`rpc.py`):** This is the client-side "magic." When a child process starts, it wraps its *copy* of the main coordinator object in an `RPCProxy`. Any attribute access or method call on this proxy is intercepted. The proxy then intelligently decides whether to fulfill the request locally or forward it to the `LocalParallelizationManager`.

-   **The Decorator API (`rpc.py`):** A set of simple, powerful decorators are used to declare the behavior of methods and attributes, making the system's intent clear and maintainable.

### The Decorator API Summary

| Decorator                   | Location                                      | Purpose                                                                                                                                                            |
| --------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@requires_main_process`    | On methods/properties of the `root_object`.   | **Flags a method/property as stateful.** When accessed via the proxy, it triggers a blocking RPC call to the main process to be executed or fetched there.        |
| `@replicated_to_parent`     | On a class definition in the child process.   | **Enables state replication.** Wraps `__setattr__` to watch for changes on specific attributes.                                                                    |
| `@replicate('attr_name')`   | On the same class as `@replicated_to_parent`. | **Marks which attributes to watch.** Tells `@replicated_to_parent` which specific attributes should trigger an asynchronous update message to the parent when changed. |
| `@callable_from_main`       | On methods of the child's replicated object.  | **Creates a safe, read-only API.** Explicitly marks methods on the child's object that the parent process is allowed to call on its local "shadow copy".           |


## 3. Communication Patterns and Data Flow

We have two primary, safe communication patterns. Crucially, all communication is **initiated by the child process**, which prevents the deadlocks common in complex bidirectional systems.

### Pattern 1: Child-to-Parent (RPC Calls)

This is used when a child needs the parent to perform a stateful action.

-   **Mechanism:** Methods that modify the central state (e.g., `simulation_run`) are decorated with `@requires_main_process`.
-   **Flow:**
    1.  The child calls a method on its proxy object (e.g., `proxy.simulation_run(...)`).
    2.  The `RPCProxy` sees the `@requires_main_process` decorator on the method.
    3.  It bundles the method's path (e.g., `"heur_coordinator.simulation_run"`) and its arguments into a message.
    4.  It sends this message to the `LocalParallelizationManager` and waits for a response.
    5.  The manager receives the message, executes the real method on the real object, and sends the result back to the child.

### Pattern 2: Parent-to-Child (State Synchronization)

This is used when the parent needs to query or influence the state of a child. It works by having the parent read from a local "shadow copy" of the child's object, which is kept up-to-date by two child-initiated mechanisms.

-   **Mechanism A: Asynchronous Attribute Replication:**
    -   Key attributes in the child's `Hyperheuristic` class (like `num_agents`) are marked using `@replicate('num_agents')`.
    -   The class itself is decorated with `@replicated_to_parent`. This decorator wraps the class's `__setattr__` method.
    -   Whenever a replicated attribute is changed in the child, the wrapper automatically sends a small, non-blocking "fire-and-forget" update message to the parent.
    -   The parent manager receives this message and uses a callback (`_update_shadow_hh_object`) to update the specific attribute on its shadow copy. This is highly efficient for frequent updates.

-   **Mechanism B: Periodic Full Refresh & "Pull" Updates:**
    -   To ensure eventual consistency, every 10th RPC call from a child automatically "piggybacks" a full, serialized copy of its state object. The parent uses this to completely overwrite its old shadow copy.
    -   For parent-driven updates (like re-distributing agents), we use a "pull" model. The parent places the update in a holding area (`pending_updates`). The child periodically calls a service (`get_pending_updates`) to check for and pull any new instructions for itself.

## 4. Rationale and Design Decisions

-   **Why not `multiprocessing.Manager`?** A standard `multiprocessing.Manager` would create a bottleneck. Every single attribute access would become a blocking network call to the main process, serializing the application and losing all benefits of multiprocessing.
-   **Why not full Bidirectional RPC?** True parent-to-child RPC is incredibly complex. It requires the child to run its own server loop to listen for requests, drastically increasing the risk of deadlocks (e.g., parent calls child at the same time child calls parent). Our hybrid model avoids this entirely.
-   **Why Decorators?** Decorators provide a clean, declarative, and Pythonic way to specify the behavior of methods and attributes without cluttering the core application logic. It makes it very easy for a future developer to understand which parts of the code are involved in multiprocessing communication.

## 5. Limitations and Guidelines for Extension

This is a powerful architecture, but it requires developers to understand its rules to extend it safely.

### Limitations

-   **Pickleable Objects:** All data passed between processes (arguments, return values, and replicated objects) must be "pickleable" by Python's `pickle` module. This means no file handles, network connections, or complex non-serializable objects can be passed.
-   **Stale State:** The parent's "shadow copy" of a child's state is not guaranteed to be 100% up-to-the-millisecond. It is updated by asynchronous messages and periodic refreshes. Logic running in the parent should be robust to acting on slightly stale (but internally consistent) data.
-   **Shallow Replication:** The `@replicate` decorator is shallow. It only detects direct assignment to an attribute (e.g., `self.my_list = [1, 2]`). It will **not** detect in-place mutations (e.g., `self.my_list.append(3)`). For mutable attributes, you must re-assign the entire object to trigger the update.

### Guidelines for Extension

When adding or modifying functionality, follow this "cookbook":

1.  **Does a child method need to modify central state (e.g., the `Manager` or `ResourceController`)?**
    -   **Action:** Add the `@requires_main_process` decorator to the method in `coordinatorBase.py` or its subclasses. Call it via the proxy from the child. Done.

2.  **Does the parent need to know when a simple value in a child object changes?**
    -   **Action:** In the child's class, add the `@replicate('attribute_name')` decorator and ensure the class has `@replicated_to_parent`. The update will now be pushed automatically on assignment.

3.  **Does the parent need to read a value or call a simple, read-only method on a child object?**
    -   **Action:** In the child's class, decorate the target method with `@callable_from_main`. In the parent, use the `_call_method_on_shadow(...)` helper to access it safely. This prevents accidental calls to unsafe methods.

4.  **Does the parent need to send a command or update to a child?**
    -   **Action:** Use the "pull" mechanism.
        1.  In `HyperHeuristicBase`, add the data to the `pending_updates` dictionary, keyed by the target `hyper_space_name` and a unique variable name (e.g., `self.pending_updates[space_name]['new_command'] = 'go'`).
        2.  In the `solve()` loop in `customhys_local/hyperheuristic.py`, add logic to check for and process your new key from the dictionary returned by `get_pending_updates`.

---
---

## Phantom Oblivion Relay

**Notice for Human Developers:** The following section is a structured data dump intended for consumption by another AI or Large Language Model. Its purpose is to provide a highly condensed, machine-readable summary of the project's architecture to rapidly onboard another AI assistant in a future session. It is not designed to be easily readable by humans.

```json
{
  "project_id": "hyper-heuristic-dse-2.0",
  "architecture_name": "Proxy with Asynchronous State Replication",
  "problem_domain": "State fragmentation in fork-based multiprocessing.",
  "solution_goal": "Centralize state in the main process while enabling performant parallel execution in child processes, avoiding deadlocks.",
  "key_concepts": {
    "rpc_direction": "Unidirectional (Child-to-Parent initiated) to prevent deadlocks.",
    "state_location": "Authoritative state lives exclusively in the main process.",
    "child_state_view": "Children operate on local copies of objects for read-only/safe operations.",
    "parent_state_view": "Parent maintains 'shadow copies' of child-specific state objects (e.g., Hyperheuristic instances)."
  },
  "core_components": [
    {
      "file": "src_main/tools/local_parallelization/local_parallelization_manager.py",
      "class": "LocalParallelizationManager",
      "role": "Main process singleton. Acts as RPC server. Receives all messages from children via a multiprocessing.Queue. Resolves RPC paths dynamically using getattr on a 'root_object'. Uses a callback mechanism to update shadow objects."
    },
    {
      "file": "src_main/tools/local_parallelization/local_parallelization_context.py",
      "class": "ParallelizationManagerContext",
      "role": "Gateway object within each child process. Provides the interface for a child to communicate with the parent manager. Does not hold application state. Exposes `create_proxy()`."
    },
    {
      "file": "src_main/tools/local_parallelization/rpc.py",
      "class": "RPCProxy",
      "role": "Client-side proxy in the child process. Wraps a local copy of an object. Intercepts `__getattr__` and `__call__`. Decides whether to execute a call locally or forward it to the parent based on the presence of decorators."
    }
  ],
  "communication_patterns": {
    "child_to_parent_rpc": {
      "trigger_decorator": "@requires_main_process",
      "description": "Marks a method as stateful. When called on a proxy, it triggers a blocking RPC call to the parent. The path of the method (e.g., 'heur_coordinator.hh_base.simulation_run') and its arguments are sent.",
      "file": "src_main/tools/local_parallelization/rpc.py"
    },
    "state_replication_to_parent": {
      "class_decorator": "@replicated_to_parent",
      "attribute_decorator": "@replicate('attr_name')",
      "description": "Wraps a class's `__setattr__`. When a monitored attribute is changed, it triggers a non-blocking, asynchronous `UPDATE_ATTRIBUTE` message to the parent.",
      "file": "src_main/tools/local_parallelization/rpc.py"
    },
    "periodic_refresh": {
      "trigger": "Every Nth call to a `@requires_main_process` method.",
      "description": "The proxy serializes its entire wrapped object and 'piggybacks' it on an RPC call. The parent uses this to overwrite its shadow copy, ensuring eventual consistency.",
      "file": "src_main/tools/local_parallelization/rpc.py"
    },
    "parent_to_child_update": {
      "pattern": "Pull-based",
      "description": "Parent places updates in a 'pending_updates' dictionary. Child periodically calls a `@requires_main_process` getter service (`get_pending_updates`) to check for and pull its own updates.",
      "files": ["src_main/tools/hyperheuristicBase.py", "src_main/external/customhys_local/hyperheuristic.py"]
    },
    "parent_querying_child": {
      "trigger_decorator": "@callable_from_main",
      "description": "Marks a method on the child's object as read-only and safe to be called by the parent. The parent calls this method on its local shadow copy. A helper (`_call_method_on_shadow`) ensures only decorated methods can be accessed, preventing unsafe calls.",
      "files": ["src_main/tools/local_parallelization/rpc.py", "src_main/tools/hyperheuristicBase.py"]
    }
  },
  "final_object_graph_in_child": {
    "entry_point": "_hh_worker_function",
    "flow": [
      "Worker receives a 'context' and a 'coordinator_copy'.",
      "Worker instantiates a real, local 'Hyperheuristic' object, passing it the 'context' and 'coordinator_copy'.",
      "The local 'Hyperheuristic' object now has its `__setattr__` wrapped by the `@replicated_to_parent` decorator.",
      "The 'Hyperheuristic' object uses its reference to the 'coordinator_copy' to make proxied calls: `self.heur_coordinator.hh_base.some_method()`.",
      "This call is intercepted by the `RPCProxy`, which decides to execute locally or remotely."
    ]
  },
  "potential_issues_and_todos": [
    {
      "id": "TODO-001",
      "type": "Limitation",
      "summary": "Shallow Replication for Mutable Objects",
      "description": "The `@replicate` mechanism only triggers on direct attribute assignment (`self.x = ...`). It does not detect in-place mutations of mutable objects like lists or dictionaries (e.g., `self.my_list.append(...)`). A developer must remember to re-assign the attribute (`self.my_list = self.my_list`) to trigger replication for such changes."
    },
    {
      "id": "TODO-002",
      "type": "Risk",
      "summary": "Serialization Risk in Periodic Refresh",
      "description": "The periodic refresh serializes the entire root object from the child. If a non-pickleable attribute (e.g., a file handle, a thread lock) is added to a replicated object in the future, the refresh mechanism will fail and throw an exception."
    },
    {
      "id": "TODO-003",
      "type": "Inconsistency/Bug",
      "summary": "Progress Bar Updates Will Fail",
      "description": "The `updateMHProgress` lambda in `_hh_worker_function` calls `proxy_coordinator.hh_base.progress.advance(...)`. This attempts to make an RPC call on the `rich.progress` object's method. However, since we cannot add `@requires_main_process` to a third-party library's method, this call will be executed locally in the child on a non-functional copy and fail silently or throw an error. The correct solution is to wrap the progress calls in dedicated, decorated service methods on `HyperHeuristicBase` itself and call those services from the lambda."
    },
    {
        "id": "TODO-004",
        "type": "Enhancement",
        "summary": "Error Handling for Async Updates",
        "description": "The `@replicate` mechanism sends a 'fire-and-forget' update. If the parent process encounters an error while processing this update (e.g., a deserialization failure), the child process will never be notified. This is acceptable for the current design but could be enhanced with a separate error-reporting channel for critical applications."
    }
  ]
}
``` 