import os
import shutil

from src.siminstance import Siminstance
from src.utils.config_creator import OmnetSimConfig, SiminstanceConfig

def create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, pdes=None, ini="omnetpp.ini", libraries=None, ned_paths=None, time_limit=None, metadata=None):
    dummy_sim_folder = os.path.join(dummy_sim_path, dummy_sim)
    print(dummy_sim_folder)

    ##### Create copy of a dummy sim
    sim_folder = os.path.join(workflow_config["sims_path"], str(id))

    if os.path.exists(sim_folder):
        shutil.rmtree(sim_folder)
    shutil.copytree(dummy_sim_folder, sim_folder)

    with open(os.path.join(sim_folder, "id.txt"), "w") as fp:
        fp.write("id: {}".format(id))
    ##### Create copy of a dummy sim

    config_file = workflow_config["local_sim_config"] + ".json"
    config_path = os.path.join(sim_folder, config_file)

    omnet_config = OmnetSimConfig(True, True, ini, config, "Cmdenv", time_limit=time_limit, pdes=pdes, libraries=libraries, ned_paths=ned_paths, metadata=None)
    sim_instance_config = SiminstanceConfig("omnet", omnet_config)
    sim_instance_config.write_conf(config_path)

    return Siminstance(sim_folder, workflow_config)

def create_sim_inet_lans(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_lans"
    config = "LargeNet"
    ini = "largeNet.ini"

    libraries = [os.path.join(inet_base, "src", "INET")]

    ned_paths = [os.path.join(inet_base, "src"), os.path.join(inet_base, "examples"),
                 os.path.join(inet_base, "showcases"), os.path.join(inet_base, "tests", "validation"),
                 os.path.join(inet_base, "tests", "networks"), os.path.join(inet_base, "tutorials")]

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, time_limit=None, ini=ini, libraries=libraries, ned_paths=ned_paths)


def create_sim_pdes(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes"
    dummy_sim_pdes = {"num_lps": 3}
    config = "LargeLookahead"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, pdes=dummy_sim_pdes)

def create_sim_pdes_comm(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_communicate"
    dummy_sim_pdes = {"num_lps": 8}
    config = "General"
    ini = "communicate_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, pdes=dummy_sim_pdes, ini=ini)

def create_sim_pdes_comp(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_compute"
    dummy_sim_pdes = {"num_lps": 8}
    config = "General"
    ini = "compute_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, pdes=dummy_sim_pdes, ini=ini)

def create_sim_pdes_sub(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_subsystem"
    dummy_sim_pdes = {"num_lps": 8}
    config = "General"
    ini = "subsystem_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, pdes=dummy_sim_pdes, ini=ini)

def create_sim_seq_comm(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_communicate"
    config = "General"
    ini = "communicate_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, ini=ini)

def create_sim_seq_comp(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_compute"
    config = "General"
    ini = "compute_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, ini=ini)

def create_sim_seq_sub(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "dummy_sim_pdes_subsystem"
    config = "General"
    ini = "subsystem_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, ini=ini)

# -------------------------------------------------------------------------------------------
# NOTE: The following functions are added by me (Lars) and does not exist in the original code.
# -------------------------------------------------------------------------------------------
def create_tictoc(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "tictoc"
    config = "General"
    ini = "omnetpp.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, ini=ini)

def create_sim_custom_dummy(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "custom_dummy"
    config = "General"
    ini = "communicate_intensive.ini"

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, ini=ini)

def create_sim_inet_lans_dummy(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "custom_dummy"
    config = "LargeNet"
    ini = "largeNet.ini"

    libraries = [os.path.join(inet_base, "src", "INET")]

    ned_paths = [os.path.join(inet_base, "src"), os.path.join(inet_base, "examples"),
                 os.path.join(inet_base, "showcases"), os.path.join(inet_base, "tests", "validation"),
                 os.path.join(inet_base, "tests", "networks"), os.path.join(inet_base, "tutorials")]

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, time_limit=None, ini=ini, libraries=libraries, ned_paths=ned_paths)

def create_sim_inet_lans_dummy_parallel(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "custom_dummy" + f"_{str(id)}"
    config = "LargeNet"
    ini = "largeNet.ini"

    libraries = [os.path.join(inet_base, "src", "INET")]

    ned_paths = [os.path.join(inet_base, "src"), os.path.join(inet_base, "examples"),
                 os.path.join(inet_base, "showcases"), os.path.join(inet_base, "tests", "validation"),
                 os.path.join(inet_base, "tests", "networks"), os.path.join(inet_base, "tutorials")]

    print(dummy_sim_path, dummy_sim)

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, time_limit=None, ini=ini, libraries=libraries, ned_paths=ned_paths)


def create_sim_asml(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "custom_faezeh"
    config = "General"
    ini = "omnetpp.ini"

    libraries = [os.path.join(inet_base, "src", "INET")]

    ned_paths = [os.path.join(inet_base, "src"), os.path.join(inet_base, "examples"),
                 os.path.join(inet_base, "showcases"), os.path.join(inet_base, "tests", "validation"),
                 os.path.join(inet_base, "tests", "networks"), os.path.join(inet_base, "tutorials")]

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, time_limit=None, ini=ini, libraries=libraries, ned_paths=ned_paths)

def create_sim_asml_parallel(workflow_config, dummy_sim_path, id, inet_base):
    dummy_sim = "custom_faezeh" + f"_{str(id)}"
    config = "General"
    ini = "omnetpp.ini"

    libraries = [os.path.join(inet_base, "src", "INET")]

    ned_paths = [os.path.join(inet_base, "src"), os.path.join(inet_base, "examples"),
                 os.path.join(inet_base, "showcases"), os.path.join(inet_base, "tests", "validation"),
                 os.path.join(inet_base, "tests", "networks"), os.path.join(inet_base, "tutorials")]

    return create_sim(workflow_config, id, dummy_sim_path, dummy_sim, config, time_limit=None, ini=ini, libraries=libraries, ned_paths=ned_paths)
