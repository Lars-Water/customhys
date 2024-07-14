from distributed import span
import matplotlib.pyplot as plt
import json
import pandas as pd
import math

import os
import time
from pathlib import Path
import argparse

from src_main.tools.config_reader import Config
import src_main.tools.coordinator as coordinator
import src_main.models.model as model
from src_main.data import collect_data
from src_main.experiment_flows import experiment_1, experiment_2
from src_main.visualization import visualization


# TODO: Improve and place this function in a separate module.
import shutil
from datetime import datetime
def remove_directory(directory_path):
    '''
        Quick and dirty solution for running many heuristic runs after each other without clogging up memory.
    '''
    try:
        shutil.rmtree(directory_path)  # Remove the entire directory and all its contents
        print(f'Successfully removed {directory_path}')
    except Exception as e:
        print(f'Failed to delete {directory_path}. Reason: {e}')


# '''
#     Run the metaheuristic search operator.

#     Args:
#         heuristic_run_config (Config): The configuration object for the heuristic run.
#         heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.
#         heuristics_collection (list): The collection of heuristics to be used in the metaheuristic.
#         num_agents (int): The number of agents in the metaheuristic.
#         num_iterations (int): The number of iterations for the metaheuristic.
#         base_path (string): Root of the project.
#         verbose (bool): The flag to indicate if the metaheuristic should run in verbose mode.
# '''
# def run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, num_agents, num_iterations, base_path, verbose=False):

#     # Run the metaheuristic for every specified number of backbones.
#     runs_nr_of_backbones = heuristic_run_config.tryGet("runs_nr_of_backbones")
#     for run in runs_nr_of_backbones:

#         # Start timer for the heuristic run.
#         start_time = time.time()

#         # Create a problem instance for the metaheuristic.
#         prob = create_problem_instance(run, heur_sim_coordinator)

#         # Generate a metaheuristic search method for the model.
#         met = mh.Metaheuristic(prob, heuristics_collection, num_agents=num_agents, num_iterations=num_iterations)
#         met.verbose = verbose

#         # Reset the execution times before starting the heuristic run.
#         heur_sim_coordinator.coordinator_and_simulation_execution_time = 0
#         heur_sim_coordinator.simulation_execution_time = 0

#         # Run the metaheuristic on the problem. The fitness value is calculated at every iteration step by the run function in the SimulationModel class.
#         met.run()

#         # End timer for the heuristic run.
#         end_time = time.time()

#         heuristic_run_meta_data = calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator)

#         # Save the heuristic run data.
#         save_run_path = os.path.join(base_path, "data/raw/results/metaheuristic/", heuristic_run_config.tryGet('save_run_path_heuristic_name'))
#         collect_data.collect_mh_run(met,save_run_path,run,num_agents,num_iterations,heuristic_run_meta_data)

#     return


'''
    Calculate the distinct components of the heuristic run.

    Args:
        start_time (float): The start time of the heuristic run.
        end_time (float): The end time of the heuristic run.
        heur_sim_coordinator (HeuristicSimulationCoordinator): The coordinator object for the heuristic simulation workflow.

    Returns:
        dict: The distinct components of the heuristic run.
'''
def calculate_distinct_simulation_components(start_time, end_time, heur_sim_coordinator):

    # Caculate the distinct execution times for the distinct components of the heuristic run.
    total_execution_time = end_time - start_time
    heuristic_execution_time = total_execution_time - heur_sim_coordinator.coordinator_and_simulation_execution_time
    coordinator_and_simulation_execution_time = heur_sim_coordinator.coordinator_and_simulation_execution_time - heur_sim_coordinator.simulation_execution_time
    simulation_execution_time = heur_sim_coordinator.simulation_execution_time

    return {
        "total_execution_time": total_execution_time,
        "heuristic_execution_time": heuristic_execution_time,
        "coordinator_and_simulation_execution_time": coordinator_and_simulation_execution_time,
        "simulation_execution_time": simulation_execution_time
    }


def quick_and_dirty_hh_multiplot(directory_path):
    all_historical_fitness = []

    # List all files in the directory
    for filename in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is a JSON file
        if filename.endswith('.json') and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    # Load JSON content
                    data = json.load(json_file)
                    historical_fitness = data['details']['historical'][0]['fitness']
                    all_historical_fitness.append(historical_fitness)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")

    plt.figure(figsize=(10, 6))

    for idx, fitness_values in enumerate(all_historical_fitness):
        plt.plot(fitness_values, label=f'HH Step: {idx + 1}')

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Optimal Fitness Every Iteration')
    plt.ylim(0.5, 0.7)
    plt.legend()

    # Save the plot in the same directory
    plot_file_path = os.path.join(directory_path, 'historical_fitness_plot.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Plot saved as {plot_file_path}")


def quick_and_dirty_rescale_back_positions(positions, centre_boundaries, span_boundaries):
    return [math.ceil((centre + position * (span / 2)) * 4) if math.ceil((centre + position * (span / 2)) * 4) != 0 else 1 for centre, position, span in zip(centre_boundaries, positions, span_boundaries)]


def quick_and_dirty_save_hh_positions_to_xlsx(positions_data, rescale=False):
    centre_boundaries = [0.5] * 50
    span_boundaries = [1.0] * 50
    # Define the positions
    # positions_data = {
    #     "Swarm": {
    #         "Worst": [-0.43839575098067607, 0.4962124805931715, -0.5976269034130429, 0.48809253714211387, -0.05788224867897773, -0.7351597470476258, 0.8600975041007879, 0.22382354559590395, 0.30542017200112986, 0.3214428173821823, -0.7643364409160542, -0.6145714136241985, -0.7079998989669304, 0.7629755878371387, 0.21193470798085917, 0.672303383947763, -0.7097764137801252, -0.4971412881049039, -0.6964817849811272, 0.3189354244274531, 0.14782428512657678, -0.4738837198040101, 0.8336004435302143, 0.5798588169587346, -0.7168282419483467, -0.4990074157056891, 0.21299427472249408, 0.21037132849475842, 0.002044334652873593, 0.3226237933028945, -0.26377547032194754, 0.9217295051154588, -0.03908072911250766, 0.14692136551738588, 0.8510033475459965, -0.555003367041518, 0.12077564990789152, 0.7864163072970966, -0.16936161992888502, 0.7583764172596038, -0.15191177883945067, 0.14591499539697989, -0.871128966567528, 0.0032524271752620085, 0.6636369547632192, 0.7455217971686623, 0.38129003383599946, -0.3203316178790086, 0.7294633918624683],
    #         "Medium": [1.0, 0.6276416280041418, -1.0, 0.7472803880635206, 1.0, -1.0, 1.0, 1.0, -1.0, 0.7303213845736264, 1.0, 0.6779379193153727, 0.9055631511414352, 0.17668638571857898, 1.0, 0.9893716434497842, 0.241385448695549, -1.0, 0.7600828379855297, 0.006807389005694507, 1.0, 0.5366372868004667, 1.0, -0.7745285128822165, -0.07730691185214489, 0.8434372909552829, 1.0, 1.0, -1.0, 0.9401129068340184, 0.10118930419064902, 0.24649154633237694, 0.9688100141294063, 0.28168297756003513, 0.9856867644765419, 1.0, 1.0, 0.6555628648031726, -0.2716084377365081, -1.0, 0.9303378718676938, 1.0, 0.7156660099032066, 1.0, -1.0, 0.11502545760242613, 0.2986700007184376, 1.0, 1.0],
    #         "Best": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4050402198032189, 1.0, 1.0, 0.6674797312217415, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7260736893837213, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

    #     },
    #     "Gravitational": {
    #         "Worst": [0.009078295016381688, 0.9917374566648525, -0.3741061030957258, 0.9834656311218541, -0.9428524910434062, 0.8801989013012543, 0.7645296660708942, 0.09420944905994744, 0.2439900598758744, -0.4911483632547178, -0.8761806284045048, 0.9990574418537543, -0.4924851735092879, -0.041758338864958056, -0.699852421933717, 0.25381860665580147, 0.28356708987553336, -0.3170613296362226, 0.6929883718502967, -0.8653832762387557, 0.35929001209320033, -0.3852835481202217, 0.5266847945804296, 0.5851578199305532, 0.08352391514918311, -0.26738147196653506, 0.05504744131731609, -0.6950782464446359, -0.2764237991003182, -0.43909628793470823, 0.13049942643592471, -0.7995236325714448, 0.31187738307576396, -0.43085091164785916, -0.13246146809971737, -0.4708869522740571, -0.0727287756977697, -0.2606098251671445, 0.9482492451998963, 0.7615906458115027, 0.028441899694919837, 0.4612130464129811, -0.499700774099594, -0.6063989990245016, -0.41091803149771944, 0.1315063744821121, -0.23423918051517023, 0.6018383291509539, 0.795860575088883],
    #         "Best": [-0.3734079884924098, -0.3054205383653441, 0.08334822595546436, -0.028566803916807895, 0.01586438946795706, -0.3466662157460246, 0.25186170231305866, -0.23169511077906335, 0.013964828120560605, -0.03626159334831878, 0.031026321951121948, -0.03247116728531867, 0.45713820334540584, -0.11057947193495092, 0.22783040851677433, 0.0013147124454981618, 0.2999778183171208, 0.37983128059688476, 0.06885337259411925, 0.2455360495887131, 0.35421714772731705, 0.127423679291823, 0.26684706903679195, 0.1341918051580468, -0.06511101977854844, -0.2675822087426275, 0.4086733489129829, -0.1326125628192202, 0.26341588208037126, 0.07019000675762553, -0.11584616327157293, 0.042342045940587186, 0.044185101536576635, 0.010318555209217137, 0.37585081996325537, 0.22052791670355887, 0.48895705634037295, 0.08888871264965983, -0.1791009381657855, 0.2403457596728794, 0.2244965860694526, -0.23212682950998556, 0.21756780718722518, 0.23017113659598687, -0.15966541174589396, 0.08773174608305438, 0.04212115605925683, -0.21924421435651462, -0.008273432266488377]

    #     },
    #     "Mutation": {
    #         "Worst": [0.8780304846360898, -0.14801329468295543, 1.0, -1.0, 1.0, -0.20760664152155742, -0.8951827222199404, 1.0, -0.984147903039928, 0.5052307225405274, 1.0, 0.7786464267846626, -0.006065930295061457, 1.0, 0.6208766423201626, 0.2164402805520613, -1.0, 0.9878813808156709, 0.04195814581862991, 0.7073550515314794, -0.21287011254492025, -0.37768889236564823, -0.22003297222362278, 0.41833818796137506, 0.8823060731557683, -1.0, -0.5848192831050554, 1.0, -0.2561359834996587, 0.6960346585696137, 0.8804528465407852, -0.8603167955316208, 0.8204781437552305, 0.6752474775012427, 1.0, 0.6275423819653142, 0.5003755372381242, 0.3108594269415752, 1.0, 1.0, 0.765382366442193, 0.0981731051220682, -1.0, -0.9108737334386574, 1.0, 0.1308995922956561, 1.0, 0.5376809015014428, -0.4464633858023343],
    #         "Best": [0.2157217896233122, 0.176161917180303, 0.22428213624462798, 0.26097361955998893, 0.13242016813744445, 0.29714647347870055, 0.26070003858194857, 0.3196619503960391, -0.006550178809794147, -0.4050617865524412, -0.19839901091449277, 0.15033326369428313, 0.4410518540070826, 0.1419747045355958, -0.33918087322837076, -0.37866783979222746, 0.35139493729358606, -0.06512893444272679, 0.4929665070648653, -0.32631485397910975, -0.2586680058335703, 0.36672089007117914, 0.15267407614569736, 0.04040558864838986, 0.17733627635259233, 0.31138192424564193, 0.11606148412806104, -0.20858192304033332, -0.4703803658788259, -0.038867514542623494, 0.3069319411036457, 0.021735986430112163, -0.0862773020165339, -0.16999683217497735, 0.2703109650075741, 0.3751748497839407, 0.05456819493753087, -0.2846079727797437, 0.13760517998407426, 0.42302287726118054, 0.33399182173675296, 0.3484579819323045, 0.3863687461900332, -0.43287427312381577, 0.09570411180960953, 0.10588584542002424, -0.28289634665104946, 0.10134087612190809, 0.016664389274043796]

    #     },
    #     "Crossover": {
    #         "Worst": [0.5464518124009579, 0.27160005042194424, 0.8641534642254884, 0.28216867682564417, 0.902826755184196, 0.6886784823671213, 0.06644156851309413, 0.7410623578382756, 0.6537357725877133, -0.2981174367287347, -0.3226857195237185, -0.6493784950470745, -0.6302932963629073, -0.9775119797448784, -0.8434389694224462, 0.035166839476338474, 0.254175132268214, 0.779988582143023, 0.8983290288513619, 0.7674329282016832, -0.5982820586172723, -0.12829953433371943, 0.36312734477422826, 0.7152809638240423, 0.8775957684664346, 0.940055720980083, -0.5098715052873162, -0.7385657905624119, 0.8457861776205897, -0.21497535212715468, 0.10101825314369761, 0.7591443381551211, 0.448101056540698, 0.10242044228311253, 0.9301485960682858, 0.08114556495540604, 0.03925213967089136, -0.796680101314998, 0.24648240847341185, -0.37104721658513795, -0.43890072554851356, -0.7116411578259918, 0.5576837643274504, 0.8137345458384624, -0.7259510712506605, 0.9621821085045761, -0.13155868612824984, 0.7532267847673226, -0.20627317726817807],
    #         "Best": [-0.32750114409697273, -0.3223890998684106, -0.19640077812240547, 0.15931259703667208, -0.022535543843164016, 0.36273467338637483, 0.4219244059618271, 0.10110439977688485, 0.40985971391840226, 0.22709716346565617, -0.33582035816159306, -0.11763707550342846, -0.10555539370921377, 0.1297886277447725, -0.027075210363023416, 0.27729679055635814, 0.32948624096514345, 0.31371507120817527, -0.023495877856167503, -0.09997028438244512, -0.17752780960156853, 0.27199025433152546, -0.32654182168571433, -0.15942003906054225, 0.09158193732898787, 0.46783418167554547, 0.20460336851696626, -0.19251610140007228, -0.037871583059580036, -0.10806083224748522, 0.1452908448290182, -0.44008949306069806, -0.1343006914041406, 0.26542603718305585, 0.1795273697212829, 0.37154769900557766, 0.06993686349867194, 0.3019741706525578, -0.24897077430616377, -0.1283352569911947, -0.27290601544709164, -0.4633330298314464, -0.4940546971108059, -0.3268633345116534, -0.391894877442753, 0.33799578282091497, -0.24545393438725238, -0.47280549248322357, 0.22819086630563726]
    #     }
    # }

    # Initialize a list to store the rescaled or original positions
    positions_list = []

    # Loop through the positions
    for method, positions in positions_data.items():
        for category, position_list in positions.items():
            if rescale:
                processed_positions = quick_and_dirty_rescale_back_positions(position_list, centre_boundaries, span_boundaries)
            else:
                processed_positions = [math.ceil(position * 4) if math.ceil(position * 4) != 0 else 1 for position in position_list]
            positions_list.append([method, category] + processed_positions)

    # Convert to a DataFrame
    columns = ['Operator', 'Type'] + [f'Position_{i+1}' for i in range(50)]
    positions_df = pd.DataFrame(positions_list, columns=columns)

    # Write the DataFrame to an Excel file
    excel_filename = 'rescaled_positions.xlsx'
    positions_df.to_excel(excel_filename, index=False)


'''
    Run the heuristic simulation workflow.

    Args:
        base_path (str): The base path for the project.
        coordinator_config_file_path (Path): The path to the coordinator configuration file.
        heur_run_config_file_path (Path): The path to the heuristic run configuration file.
        parameter_tuning (bool): The flag to indicate if parameter tuning is enabled.
        design_space_plot (bool): The flag to indicate if design space determination is enabled.
        nr_of_backbone_switches (int): The number of backbone switches for the INET model.
'''
def main(base_path, coordinator_config_file_path, heur_run_config_file_path, parameter_tuning, design_space_plot, nr_of_backbone_switches):
    # # Set up the general heuristic run configuration.
    if heur_run_config_file_path:
        # directories = [
        #     Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/experiment_2/metaheuristics/gravitational"),
        #     Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/experiment_2/metaheuristics/pso"),
        #     Path("/home/larry/hyper-heuristic-dse-2.0/data/raw/results/experiment_2/metaheuristics/ga")
        # ]
        # for directory in directories:
        #     visualization.multiplot_metaheuristic_runs(directory)

        # Set up the experiment_2 configuration object.
        experiment_2_config_file_path = Path(os.path.join(base_path, "config/experiment_2/experiment_2.json"))
        experiment_2_log_path = os.path.join(base_path, "data/logs/experiments/experiment_2")
        os.makedirs(experiment_2_log_path, exist_ok=True)
        config_manager_filename = f"experiment_2_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment_2_config = Config(experiment_2_config_file_path, Path(experiment_2_log_path), config_manager_filename)

        # Run Experiment 2.
        experiment_2.run_experiment(base_path, experiment_2_config, coordinator_config_file_path)

        # # Set up the experiment_1 configuration object.
        # experiment_1_config_file_path = Path(os.path.join(base_path, "config/experiment_1/experiment_1.json"))
        # experiment_1_log_path = os.path.join(base_path, "data/logs/experiments/experiment_1")
        # os.makedirs(experiment_1_log_path, exist_ok=True)
        # config_manager_filename = f"experiment_1_config_manager_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # experiment_1_config = Config(experiment_1_config_file_path, Path(experiment_1_log_path), config_manager_filename)

        # # Create the coordinator for the heuristic simulation workflow.
        # # TODO: Change this naming flow, because it goes from main, to coordinator, to data collector.
        # nr_of_agents = experiment_1_config.tryGet("hh_parameters", "num_agents")
        # run_name = "experiment_1"
        # heur_sim_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, nr_of_agents, run_name, nr_of_backbone_switches) # noqa 501

        # # Run Experiment 1.
        # min_datarate, max_datarate, min_cost, max_cost = heur_sim_coordinator.get_datarate_cost_boundaries()
        # experiment_1.run_experiment(experiment_1_config, heur_sim_coordinator, nr_of_backbone_switches)

        # heuristic_run_log_path = os.path.join(base_path, "data/logs/heuristic_run/")
        # os.makedirs(heuristic_run_log_path, exist_ok=True)
        # heuristic_run_config = Config(heur_run_config_file_path, Path(heuristic_run_log_path), "heuristic_run_config_manager")

    #     # # Obtain the configuration parameters for the heuristic run.
    #     # nr_of_agents = heuristic_run_config.tryGet("nr_of_agents")
    #     # nr_of_iterations = heuristic_run_config.tryGet("nr_of_iterations")

    # #     # Define the heuristic operators to be used.
    # #     heuristics_collection = define_heuristic_operators(heuristic_run_config)

    # #     # Visualisation of metaheuristic runs.
    # #     collect_data.vizualize_mh_runs(nr_of_backbone_switches)

    # #     # run_mh(heuristic_run_config, heur_sim_coordinator, heuristics_collection, nr_of_agents, nr_of_iterations, base_path, verbose=False)

    # #     run_hh(heuristic_run_config, heur_sim_coordinator)

    # #     # # TODO: Quick and dirty Manual removal of directories to prevent memory clogging.
    # #     # experiments_directory_path = Path("/var/scratch/lvdwater/experiments/data/campaign_custom/n1_w1_s16")
    # #     # remove_directory(experiments_directory_path)
    # #     # sims_directory_path = Path("/var/scratch/lvdwater/sims")
    # #     # remove_directory(sims_directory_path)

    # # elif parameter_tuning:
    # #     parameter_tuning_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, None, nr_of_backbone_switches) # noqa 501
    # #     parameter_tuning_coordinator.parameter_tuning_workflow()
    # # elif design_space_plot:
    # #     design_space_coordinator = coordinator.HeuristicSimulationCoordinator(base_path, coordinator_config_file_path, None, nr_of_backbone_switches) # noqa 501
    # #     design_space_coordinator.determine_design_space()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the heuristic simulation workflow.")
    parser.add_argument("--nr_sw", type=int, required=False, help="The number of backbone switches.")
    parser.add_argument("--base_path", type=str, required=True, help="Root of the project.")
    parser.add_argument("--coordinator_config", type=str, required=True, help="Path to the coordinator configuration file.")
    parser.add_argument("--heur_run_config", type=str, required=False, help="Path to the heuristic run configuration file.")
    parser.add_argument("--param_tune", action="store_true", required=False, help="Flag that enables parameter tuning.")
    parser.add_argument("--design_space_plot", action="store_true", required=False, help="Flag that enables determining the design space.")
    args = parser.parse_args()

    # Convert the arguments to Path objects.
    nr_of_backbone_switches = args.nr_sw
    base_path = Path(args.base_path)
    coordinator_config_file_path = Path(args.coordinator_config)
    heur_run_config_file_path = Path(args.heur_run_config) if args.heur_run_config else None
    parameter_tuning = args.param_tune
    design_space_plot = args.design_space_plot

    # Check if the base path exists.
    if not base_path.exists():
        raise FileNotFoundError(f"The base path {base_path} does not exist.")

    # Check if the coordinator configuration file exists
    if not coordinator_config_file_path.exists():
        raise FileNotFoundError(f"The coordinator configuration file {coordinator_config_file_path} does not exist.")

    # Check if no heuristic config file path is provided nor parameter tuning is enabled.
    if not heur_run_config_file_path and not parameter_tuning and not design_space_plot:
        raise ValueError("Please provide a heuristic run configuration file or enable parameter tuning or design space determination.")
    # Check if both heuristic config file path is provided and parameter tuning is enabled.
    elif heur_run_config_file_path and parameter_tuning or heur_run_config_file_path and design_space_plot or parameter_tuning and design_space_plot:
        raise ValueError("Multiple functionalities enabled; Please provide only a heuristic run configuration file, or only enable parameter tuning or design space configuration.")
    # Check if the heuristic config file path exists.
    elif heur_run_config_file_path and not heur_run_config_file_path.exists():
        raise FileNotFoundError(f"The heuristic run configuration file {heur_run_config_file_path} does not exist.")
    # TODO: Check if parameter tuning configurations are provided if parameter tuning is enabled.
    pass
    # TODO: Check if design space configurations are provided if design space is enabled.
    pass

    main(base_path, coordinator_config_file_path, heur_run_config_file_path, parameter_tuning, design_space_plot, nr_of_backbone_switches)
