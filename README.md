hh-dse-2.0
==============================

An implementation that that enables the application of hyper-heuristic methodologies on a parallel discrete event simulation model.

Project Organization

NOTE: For codespaces, move omnet folder into master branch of the repo to run simulations. However, I seem to not be able to use it outside of the repo for some reason?
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── bin
    ├── config
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── raw
    ├── docs
    ├── notebooks
    ├── reports
    │   └── figures
    └── src
        ├── data
        ├── external
        ├── models
        ├── tools
        └── visualization

## CPU Optimization Features

This framework now includes advanced CPU optimization features to maximize performance:

### 1. CPU Affinity (Default)
Automatically distributes threads across CPU cores for better performance:
- **Level 1**: Search operator space threads are distributed across cores
- **Level 2**: Metaheuristic replica threads are distributed across cores
- **Requirements**: `psutil` library (auto-installed via `requirements.txt`)

### 2. Multiprocessing Option  
For compute-intensive workloads, use true multiprocessing instead of threading:
```python
hh_parameters = {
    'cardinality': 3,
    'num_iterations': 100,
    'num_agents': 30,
    'num_replicas': 50,
    'num_steps': 200,
    'evaluation_method': 'multiprocessing',  # Enable multiprocessing
    # ... other parameters
}
```

### Configuration Options
- `'evaluation_method': 'threading'` (default) - Threading with CPU affinity
- `'evaluation_method': 'multiprocessing'` - True multiprocessing for better CPU utilization

### Performance Recommendations
1. **For I/O-bound tasks**: Use default threading with CPU affinity
2. **For CPU-intensive tasks**: Use multiprocessing 
3. **For mixed workloads**: Test both options to find optimal performance

### System Requirements
- Linux/Unix systems for full CPU affinity support
- `psutil` library for CPU management
- Python multiprocessing support for the multiprocessing option
