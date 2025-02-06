import os
import modulus.sym
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.graph import Graph
from modulus.sym import quantity

import pickle
import typing
import numpy as np

from .equations.CustomNavierStokes_noDivide_nd_ideal import CustomNavierStokes
from .equations.CustomNavierStokes_noDivide_nd_ideal_inviscid import CustomNavierStokes as CustomNavierStokes_inviscid

def create_conf_file(path: str,
                     training: dict,
                     modulus: dict):
    """_summary_

    Args:
        path (_type_): _description_
        training (_type_): _description_
        modulus (_type_): _description_
    """
    
    os.mkdir(f"{path}/conf")
    
    file = (f"defaults: \n"
            f"  - modulus_default \n"
            f"  - scheduler: {training['scheduler']} \n"
            f"  - optimizer: {training['optimizer']} \n" 
            f"  - loss: {training['loss']} \n"
            f"  - _self_ \n"
            f"custom: \n"
            f"    viscous: {training['viscous']} \n"
            f"    path: {path} \n"
            f"scheduler: \n"
            f"    decay_rate: {training['decay_rate']} \n"
            f"    decay_steps: {training['decay_steps']} \n"
            f"training: \n"
            f"    rec_results_freq: {modulus['rec_results_freq']} \n"
            f"    max_steps : {modulus['max_steps']} \n"
            f"    rec_inference_freq: {modulus['rec_inference_freq']} \n"
            f"    rec_monitor_freq: {modulus['rec_monitor_freq']} \n"
            f"    rec_constraint_freq: {modulus['rec_constraint_freq']} \n"
            f"optimizer:  \n"
            f"    lr: {training['lr']} \n"       
            f"save_filetypes: {'vtp'} \n"
            f"run_mode: {'train'}")
    
    with open("conf/conf.yaml", "w") as f_out:
        f_out.write(file)
        
def create_run_file(path: str):
    """_summary_

    Args:
        path (_type_): _description_
    """    
    file = f"""
import modulus.sym
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.graph import Graph
from modulus.sym import quantity

import pickle

import numpy as np

from pypinn.equations.CustomNavierStokes_noDivide_nd_ideal import CustomNavierStokes
from pypinn.equations.CustomNavierStokes_noDivide_nd_ideal_inviscid import CustomNavierStokes as CustomNavierStokes_inviscid

from pypinn.Problem_Domain import Problem_Domain

@modulus.sym.main(config_path="conf", config_name="conf")
def run(cfg: ModulusConfig) -> None:
    
    file = open(fr"{{cfg.custom.path}}/problem_domain.obj",'rb')
    problem_domain = pickle.load(file)
    file.close()        
        
    # make flow domain
    flow_domain = Domain()
    
    #######################################################################
    #######                Create Nodes for Networks                 ######
    #######################################################################
    
    for network in problem_domain.networks:
        network.create_network_nodes(viscous = cfg.custom.viscous)
    
    #######################################################################
    #######                Constraints                               ######
    #######################################################################
    
    for constraint in problem_domain.constraints:
        constraint.create_modObj(mod_domain = flow_domain)
                                                                 
    #######################################################################
    #######                PLOTTTING                                 ######
    #######################################################################
    
    for plot in problem_domain.plots:
        plot.create_modObj(mod_domain = flow_domain)
                                 
    #######################################################################
    #######                SOLVING                                   ######
    #######################################################################
            
    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()

if __name__ == "__main__":
    run()        
    """
    
    with open("run.py", "w") as f_out:
        f_out.write(file)
